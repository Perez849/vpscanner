"""
model.py — Modelo de probabilidad de acierto ("meta-etiquetado").

Dado un evento (un setup que se dispara) estima P(la operación termina con ganancia neta).
Es una regresión logística con L2 sobre las variables discretizadas por cuantiles (captura
no-linealidades simples sin riesgo de sobreajuste de un árbol profundo) + una constante por
setup + un efecto por grupo de activos. Solo NumPy: los coeficientes se guardan en JSON y
el scanner diario los lee sin depender de scikit-learn.

Calibración: tabla monótona (isotónica por bins) entre la probabilidad cruda y la
frecuencia real observada FUERA de muestra (validación walk-forward).
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

# variables usadas por el modelo (subconjunto de research.FEATURES)
MODEL_FEATURES = ['rsi2', 'ibs', 'ret1', 'ret5', 'ret10', 'atrp', 'atr_rel', 'dist200', 'dist50', 'slope200',
                  'dd20', 'dd52', 'pos52', 'bbz', 'vol_ratio', 'dn_streak', 'gap',
                  'max_gap5', 'shock5', 'max_vr5', 'rel5',
                  'spy_up', 'spy_dist200', 'spy_rsi2', 'spy_dd60', 'vix', 'vix_z', 'vix_chg5',
                  'b_up200', 'b_os', 'b_ret1', 'b_ret5',
                  'vp_poc_atr', 'vp_val_atr', 'vp_pos']
NB = 5


class LogitModel:
    def __init__(self, feat_idx: List[int], flag_names: List[str], groups: List[str], nb: int = NB, l2: float = 30.0):
        self.nb = nb
        self.l2 = l2
        self.feat_idx = feat_idx            # columnas de X que usa
        self.flag_names = flag_names        # banderas binarias (qué patrones se han disparado)
        self.groups = groups
        self.edges: List[np.ndarray] = []
        self.w: Optional[np.ndarray] = None
        self.b = 0.0
        self.calib: List[List[float]] = []  # [[p_crudo, p_calibrado], ...]

    # ── diseño ────────────────────────────────────────────────────────────
    def fit_bins(self, X: np.ndarray):
        self.edges = []
        for j in self.feat_idx:
            col = X[:, j]
            col = col[np.isfinite(col)]
            if len(col) < 50:
                self.edges.append(np.array([]))
                continue
            q = np.unique(np.quantile(col, np.linspace(0, 1, self.nb + 1)[1:-1]))
            self.edges.append(q)

    def design(self, X: np.ndarray, FL: np.ndarray, gid: np.ndarray) -> np.ndarray:
        n = len(X)
        cols = []
        for k, j in enumerate(self.feat_idx):
            col = X[:, j]
            e = self.edges[k]
            nan = ~np.isfinite(col)
            b = np.searchsorted(e, np.where(nan, 0, col), side='right')
            b = np.where(nan, len(e) + 1, b)
            oh = np.zeros((n, len(e) + 2), dtype=np.float32)
            oh[np.arange(n), b] = 1.0
            cols.append(oh)
        go = np.zeros((n, len(self.groups)), dtype=np.float32)
        go[np.arange(n), gid] = 1.0
        return np.concatenate(cols + [FL.astype(np.float32), go, np.ones((n, 1), np.float32)], axis=1)

    # ── ajuste ────────────────────────────────────────────────────────────
    def fit(self, X, FL, gid, y, l2: float | None = None, iters: int = 15, chunk: int = 150000, max_rows: int = 600000, seed: int = 0):
        l2 = self.l2 if l2 is None else l2
        self.fit_bins(X)
        if len(X) > max_rows:       # eventos muy correlacionados: submuestreo sin pérdida relevante
            sel = np.sort(np.random.default_rng(seed).choice(len(X), max_rows, replace=False))
            X, FL, gid, y = X[sel], FL[sel], gid[sel], y[sel]
        n = len(X)
        d = self.design(X[:2], FL[:2], gid[:2]).shape[1]
        w = np.zeros(d)
        pri = np.full(d, l2); pri[-1] = 1e-3           # el término constante casi sin penalizar
        yy = y.astype(np.float64)
        for _ in range(iters):
            g = pri * w
            H = np.diag(pri)
            for a0 in range(0, n, chunk):
                Z = self.design(X[a0:a0 + chunk], FL[a0:a0 + chunk], gid[a0:a0 + chunk]).astype(np.float64)
                p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
                g += Z.T @ (p - yy[a0:a0 + chunk])
                H += (Z * (p * (1 - p) + 1e-6)[:, None]).T @ Z
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-5:
                break
        self.w = w
        return self

    def predict_raw(self, X, FL, gid) -> np.ndarray:
        out = np.empty(len(X))
        for a0 in range(0, len(X), 200000):
            Z = self.design(X[a0:a0 + 200000], FL[a0:a0 + 200000], gid[a0:a0 + 200000]).astype(np.float64)
            out[a0:a0 + 200000] = 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w, -30, 30)))
        return out

    def set_calibration(self, p_oos: np.ndarray, y_oos: np.ndarray, nbins: int = 12):
        """Isotónica por bins sobre predicciones fuera de muestra."""
        if len(p_oos) < 500:
            self.calib = []
            return
        order = np.argsort(p_oos)
        p, y = p_oos[order], y_oos[order].astype(float)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, nbins + 1)))
        xs, ys, ws = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (p >= a) & (p <= b) if b == edges[-1] else (p >= a) & (p < b)
            if m.sum() >= 30:
                xs.append(float(p[m].mean())); ys.append(float(y[m].mean())); ws.append(int(m.sum()))
        # monotonía (pool-adjacent-violators)
        ys_ = list(ys); ws_ = list(ws); i = 0
        while i < len(ys_) - 1:
            if ys_[i] > ys_[i + 1]:
                tot = ws_[i] + ws_[i + 1]
                ys_[i] = (ys_[i] * ws_[i] + ys_[i + 1] * ws_[i + 1]) / tot
                ws_[i] = tot
                del ys_[i + 1]; del ws_[i + 1]; del xs[i + 1]
                i = max(i - 1, 0)
            else:
                i += 1
        self.calib = [[x, y] for x, y in zip(xs, ys_)]

    def predict(self, X, FL, gid) -> np.ndarray:
        p = self.predict_raw(X, FL, gid)
        if len(self.calib) >= 2:
            xs = np.array([c[0] for c in self.calib]); ys = np.array([c[1] for c in self.calib])
            p = np.interp(p, xs, ys)
        return p

    # ── JSON ──────────────────────────────────────────────────────────────
    def to_json(self) -> Dict:
        return {'feat_idx': self.feat_idx, 'flag_names': self.flag_names, 'groups': self.groups,
                'edges': [e.round(6).tolist() for e in self.edges], 'w': self.w.round(5).tolist(),
                'calib': [[round(a, 4), round(b, 4)] for a, b in self.calib]}

    @staticmethod
    def from_json(d: Dict) -> 'LogitModel':
        m = LogitModel(d['feat_idx'], d['flag_names'], d['groups'])
        m.edges = [np.array(e) for e in d['edges']]
        m.w = np.array(d['w'])
        m.calib = d.get('calib', [])
        return m


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """AUC por rangos (Mann-Whitney)."""
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = np.argsort(np.argsort(p)) + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class RidgeModel(LogitModel):
    """Misma discretización que LogitModel pero regresión lineal (ridge) sobre el resultado neto de la operación."""
    def fit(self, X, FL, gid, y, l2: float = 300.0, chunk: int = 150000, max_rows: int = 600000, seed: int = 0):
        self.fit_bins(X)
        if len(X) > max_rows:
            sel = np.sort(np.random.default_rng(seed).choice(len(X), max_rows, replace=False))
            X, FL, gid, y = X[sel], FL[sel], gid[sel], y[sel]
        d = self.design(X[:2], FL[:2], gid[:2]).shape[1]
        A = np.diag(np.full(d, l2)); A[-1, -1] = 1e-3
        b = np.zeros(d)
        yy = np.clip(y.astype(np.float64), -15, 15)
        for a0 in range(0, len(X), chunk):
            Z = self.design(X[a0:a0 + chunk], FL[a0:a0 + chunk], gid[a0:a0 + chunk]).astype(np.float64)
            A += Z.T @ Z
            b += Z.T @ yy[a0:a0 + chunk]
        self.w = np.linalg.solve(A, b)
        return self

    def predict_raw(self, X, FL, gid) -> np.ndarray:
        out = np.empty(len(X))
        for a0 in range(0, len(X), 200000):
            Z = self.design(X[a0:a0 + 200000], FL[a0:a0 + 200000], gid[a0:a0 + 200000]).astype(np.float64)
            out[a0:a0 + 200000] = Z @ self.w
        return out
