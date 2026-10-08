"""
feats.py — Indicadores 100% CAUSALES: el valor en la barra t usa solo datos hasta t.

Es el cambio de fondo respecto a la versión anterior: nada de perfiles calculados con
barras futuras ni de pivotes que necesitan 20 barras posteriores para existir.
El Volume Profile se conserva como "valor justo" pero calculado con una ventana móvil
de las últimas W barras (hasta hoy incluido), que es exactamente lo que se puede
saber en directo.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

Bars = Dict[str, np.ndarray]


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Media de Wilder (RMA) con NaN inicial tolerado."""
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _rsi_parts(c: np.ndarray, n: int):
    d = np.diff(c, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = np.full_like(c, np.nan)
    rd = np.full_like(c, np.nan)
    ru[1:] = _wilder(up[1:], n)
    rd[1:] = _wilder(dn[1:], n)
    return ru, rd


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = np.nan
    # arrancamos en la barra 1 para que el primer NaN no contamine el EWM
    ru = np.full_like(c, np.nan)
    rd = np.full_like(c, np.nan)
    ru[1:] = _wilder(up[1:], n)
    rd[1:] = _wilder(dn[1:], n)
    with np.errstate(invalid='ignore', divide='ignore'):
        rs = ru / rd
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where((rd == 0) & np.isfinite(ru), 100.0, out)
    return out


def _streak(cond: np.ndarray) -> np.ndarray:
    """Nº de barras consecutivas (hasta t) en que cond es True."""
    n = len(cond)
    idx = np.arange(n)
    last_false = np.maximum.accumulate(np.where(~cond, idx, -1))
    return (idx - last_false).astype(np.float64)


def build(b: Bars) -> Dict[str, np.ndarray]:
    t, o, h, l, c, v = b['t'], b['o'], b['h'], b['l'], b['c'], b['v']
    n = len(c)
    F: Dict[str, np.ndarray] = {'t': t, 'o': o, 'h': h, 'l': l, 'c': c, 'v': v}
    cp = np.concatenate([[np.nan], c[:-1]])

    for k in (5, 10, 20, 50, 100, 200):
        F[f'sma{k}'] = _sma(c, k)
    F['rsi2'] = _rsi(c, 2)
    F['rsi2_ru'], F['rsi2_rd'] = _rsi_parts(c, 2)       # medias de Wilder: permiten calcular el cierre que daría RSI(2)=70
    F['rsi3'] = _rsi(c, 3)
    F['rsi14'] = _rsi(c, 14)

    tr = np.maximum.reduce([h - l, np.abs(h - cp), np.abs(l - cp)])
    tr[0] = h[0] - l[0]
    F['atr'] = _wilder(tr, 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        F['atrp'] = F['atr'] / c * 100.0
        rng = h - l
        F['ibs'] = np.where(rng > 0, (c - l) / rng, 0.5)
        F['ret1'] = (c / cp - 1.0) * 100.0
        for k in (2, 3, 5, 10, 20, 60, 120):
            ck = np.concatenate([np.full(k, np.nan), c[:-k]])
            F[f'ret{k}'] = (c / ck - 1.0) * 100.0
        F['gap'] = (o / cp - 1.0) * 100.0
        F['dist200'] = (c / F['sma200'] - 1.0) * 100.0
        F['dist50'] = (c / F['sma50'] - 1.0) * 100.0
        F['dist20'] = (c / F['sma20'] - 1.0) * 100.0
        s200 = F['sma200']
        s200_20 = np.concatenate([np.full(20, np.nan), s200[:-20]])
        F['slope200'] = (s200 / s200_20 - 1.0) * 100.0
        hi20 = pd.Series(h).rolling(20, min_periods=20).max().to_numpy()
        F['dd20'] = (c / hi20 - 1.0) * 100.0
        for k in (252,):
            hk = pd.Series(h).rolling(k, min_periods=60).max().to_numpy()
            lk = pd.Series(l).rolling(k, min_periods=60).min().to_numpy()
            F['pos52'] = np.where(hk > lk, (c - lk) / (hk - lk), np.nan)
            F['dd52'] = (c / hk - 1.0) * 100.0
        sd20 = pd.Series(c).rolling(20, min_periods=20).std(ddof=0).to_numpy()
        F['bbz'] = np.where(sd20 > 0, (c - F['sma20']) / sd20, 0.0)
        vavg = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
        vprev = np.concatenate([[np.nan], vavg[:-1]])
        F['vol_ratio'] = np.where(vprev > 0, v / vprev, np.nan)
        F['dvol20'] = pd.Series(c * v).rolling(20, min_periods=10).mean().to_numpy()

    # choque reciente (proxy de noticias/resultados): huecos, volumen y movimientos extremos de las últimas 5 barras
    with np.errstate(invalid='ignore', divide='ignore'):
        gap_atr = np.abs(F['gap']) / F['atrp']
        shock = np.abs(F['ret1']) / F['atrp']
        F['max_gap5'] = pd.Series(gap_atr).rolling(5, min_periods=3).max().to_numpy()
        F['shock5'] = pd.Series(shock).rolling(5, min_periods=3).max().to_numpy()
        F['max_vr5'] = pd.Series(F['vol_ratio']).rolling(5, min_periods=3).max().to_numpy()
    for k in (3, 5, 10, 20, 55, 252):
        F[f'lowest{k}'] = (c <= pd.Series(c).rolling(k, min_periods=k).min().to_numpy()).astype(np.float64)
        F[f'highest{k}'] = (c >= pd.Series(c).rolling(k, min_periods=k).max().to_numpy()).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        F['rng_atr'] = (h - l) / F['atr']
    F['dn_streak'] = _streak(c < cp)
    F['up_streak'] = _streak(c > cp)
    # velas con cierre bajo la mínima previa (impulso bajista) y semana/volatilidad relativa
    atr100 = _sma(F['atrp'], 100)
    with np.errstate(invalid='ignore', divide='ignore'):
        F['atr_rel'] = F['atrp'] / atr100
    return F


# ── Perfil de volumen causal (ventana móvil) ──────────────────────────────
def vp_levels(F: Dict[str, np.ndarray], idx: np.ndarray, W: int = 60, B: int = 24,
              va_pct: float = 0.68):
    """
    Perfil de volumen de las últimas W barras terminando en cada índice de `idx`
    (hasta t inclusive). Devuelve (poc, vah, val) arrays del mismo tamaño.
    El volumen de cada barra se reparte proporcionalmente al solape con cada nivel.
    """
    h, l, v = F['h'], F['l'], F['v']
    m = len(idx)
    poc = np.full(m, np.nan)
    vah = np.full(m, np.nan)
    val = np.full(m, np.nan)
    for q in range(m):
        i = int(idx[q])
        if i + 1 < W:
            continue
        hh = h[i + 1 - W:i + 1]
        ll = l[i + 1 - W:i + 1]
        vv = v[i + 1 - W:i + 1]
        ph, pl = hh.max(), ll.min()
        if ph <= pl:
            continue
        step = (ph - pl) / B
        edges_lo = pl + np.arange(B) * step            # (B,)
        edges_hi = edges_lo + step
        rng = np.maximum(hh - ll, 1e-12)[:, None]      # (W,1)
        ov = np.clip(np.minimum(hh[:, None], edges_hi[None, :]) - np.maximum(ll[:, None], edges_lo[None, :]), 0, None)
        w = np.where(vv.sum() > 0, vv, 1.0)[:, None]   # sin volumen (FX): reparto uniforme
        vol = (w * ov / rng).sum(axis=0)
        k = int(vol.argmax())
        tot = vol.sum()
        target = tot * va_pct
        acc, a, bq = vol[k], k, k
        while acc < target:
            up = vol[a + 1] if a < B - 1 else 0.0
            dn = vol[bq - 1] if bq > 0 else 0.0
            if up == 0 and dn == 0:
                break
            if up >= dn:
                acc += up
                a += 1
            else:
                acc += dn
                bq -= 1
        poc[q] = pl + (k + 0.5) * step
        vah[q] = pl + (a + 1.0) * step
        val[q] = pl + bq * step
    return poc, vah, val


# ── Régimen de mercado (SPY / VIX) alineado por fecha ─────────────────────
def regime_series(spy: Dict[str, np.ndarray], vix: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Features de régimen indexadas por el timestamp de SPY."""
    c = spy['c']
    s200 = _sma(c, 200)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = {
            't': spy['t'],
            'spy_up': np.where(np.isfinite(s200), (c > s200).astype(np.float64), np.nan),
            'spy_dist200': (c / s200 - 1.0) * 100.0,
            'spy_rsi2': _rsi(c, 2),
            'spy_ret5': np.concatenate([np.full(5, np.nan), (c[5:] / c[:-5] - 1.0) * 100.0]),
            'spy_ret60': np.concatenate([np.full(60, np.nan), (c[60:] / c[:-60] - 1.0) * 100.0]),
            'spy_ret120': np.concatenate([np.full(120, np.nan), (c[120:] / c[:-120] - 1.0) * 100.0]),
            'spy_dd60': (c / pd.Series(spy['h']).rolling(60, min_periods=20).max().to_numpy() - 1.0) * 100.0,
        }
    # VIX alineado a las fechas de SPY (as-of)
    vt, vc = vix['t'], vix['c']
    j = np.searchsorted(vt, spy['t'] + 86400 - 1, side='right') - 1     # último VIX con fecha <= día de SPY
    j = np.clip(j, 0, len(vc) - 1)
    vix_al = vc[j]
    vm = pd.Series(vix_al).rolling(252, min_periods=60)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['vix'] = vix_al
        out['vix_z'] = ((pd.Series(vix_al) - vm.mean()) / vm.std()).to_numpy()
        out['vix_chg5'] = pd.Series(vix_al).pct_change(5).to_numpy() * 100.0
    return out


def align_regime(reg: Dict[str, np.ndarray], t_sym: np.ndarray) -> Dict[str, np.ndarray]:
    """Para cada barra del símbolo, el régimen del último día de SPY con fecha <= la de la barra."""
    day_sym = (t_sym // 86400).astype(np.int64)
    day_spy = (reg['t'] // 86400).astype(np.int64)
    j = np.searchsorted(day_spy, day_sym, side='right') - 1
    valid = j >= 0
    j = np.clip(j, 0, len(day_spy) - 1)
    out = {}
    for k, arr in reg.items():
        if k == 't':
            continue
        a = arr[j].astype(np.float64)
        a[~valid] = np.nan
        out[k] = a
    return out


# ── Amplitud de mercado (calculada con todo el universo, por día) ─────────
class Breadth:
    """Acumula, día a día, qué fracción del universo está sobre su SMA200, sobreventa (RSI2<10) y su rentabilidad media."""
    def __init__(self, d0: int = 15000, d1: int = 24000):
        self.d0 = d0
        n = d1 - d0
        self.cnt = np.zeros(n); self.up = np.zeros(n); self.os = np.zeros(n)
        self.r1 = np.zeros(n); self.r5 = np.zeros(n); self.n_r = np.zeros(n)

    def add(self, F: Dict[str, np.ndarray]) -> None:
        day = (F['t'] // 86400).astype(np.int64) - self.d0
        ok = (day >= 0) & (day < len(self.cnt)) & np.isfinite(F['sma200']) & (F['dvol20'] > 2e6 if 'dvol20' in F else True)
        d = day[ok]
        self.cnt[d] += 1
        self.up[d] += (F['c'][ok] > F['sma200'][ok])
        self.os[d] += (F['rsi2'][ok] < 10)
        r1 = F['ret1'][ok]; r5 = F['ret5'][ok]
        fin = np.isfinite(r1) & np.isfinite(r5)
        self.r1[d[fin]] += np.clip(r1[fin], -25, 25); self.r5[d[fin]] += np.clip(r5[fin], -50, 50); self.n_r[d[fin]] += 1

    def finalize(self) -> Dict[str, np.ndarray]:
        with np.errstate(invalid='ignore', divide='ignore'):
            ok = self.cnt >= 100
            out = {'day': np.arange(len(self.cnt)) + self.d0,
                   'b_up200': np.where(ok, self.up / self.cnt, np.nan) * 100,
                   'b_os': np.where(ok, self.os / self.cnt, np.nan) * 100,
                   'b_ret1': np.where(self.n_r >= 100, self.r1 / self.n_r, np.nan),
                   'b_ret5': np.where(self.n_r >= 100, self.r5 / self.n_r, np.nan)}
        return out


def align_breadth(bd: Optional[Dict[str, np.ndarray]], t_sym: np.ndarray, F: Dict[str, np.ndarray]) -> None:
    keys = ('b_up200', 'b_os', 'b_ret1', 'b_ret5')
    if bd is None:
        for k in keys:
            F[k] = np.full(len(t_sym), np.nan)
        return
    day = (t_sym // 86400).astype(np.int64)
    # último día con amplitud disponible <= día de la barra (as-of), sin mirar al futuro
    j = np.searchsorted(bd['day'], day, side='right') - 1
    j = np.clip(j, 0, len(bd['day']) - 1)
    for k in keys:
        F[k] = bd[k][j]


# ── Matriz de características por evento (idéntica en investigación y en producción) ──
FEATURES = ['rsi2', 'rsi3', 'rsi14', 'ibs', 'ret1', 'ret2', 'ret3', 'ret5', 'ret10', 'ret20', 'atrp', 'atr_rel',
            'dist200', 'dist50', 'dist20', 'slope200', 'dd20', 'dd52', 'pos52', 'bbz', 'vol_ratio',
            'dn_streak', 'up_streak', 'gap',
            'spy_up', 'spy_dist200', 'spy_rsi2', 'spy_ret5', 'spy_dd60', 'vix', 'vix_z', 'vix_chg5',
            'max_gap5', 'shock5', 'max_vr5', 'b_up200', 'b_os', 'b_ret1', 'b_ret5', 'rel5',
            'ret60', 'ret120', 'rs60', 'rs120',
            'vp_poc_atr', 'vp_val_atr', 'vp_pos']
GROUPS = ['us_large', 'us_mid', 'us_small', 'thematic', 'etf', 'crypto', 'fx', 'futures', 'eu', 'index']
REGIME_KEYS = ('spy_up', 'spy_dist200', 'spy_rsi2', 'spy_ret5', 'spy_ret60', 'spy_ret120', 'spy_dd60', 'vix', 'vix_z', 'vix_chg5')


def event_matrix(F: Dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """X[n_eventos, len(FEATURES)] en las barras `idx` (causal: solo mira hasta cada barra)."""
    idx = np.asarray(idx, dtype=np.int64)
    poc, vah, val = vp_levels(F, idx)
    atr = F['atr'][idx]
    c = F['c'][idx]
    with np.errstate(invalid='ignore', divide='ignore'):
        extra = [(poc - c) / atr, (c - val) / atr, (c - val) / (vah - val)]
    n = len(F['c'])
    base = []
    for f in FEATURES[:-3]:
        if f == 'rel5':           # caída propia frente al mercado (idiosincrática vs. sistemática)
            base.append(F['ret5'][idx] - F['b_ret5'][idx] if 'b_ret5' in F else np.full(len(idx), np.nan))
        elif f in ('rs60', 'rs120'):   # fuerza relativa frente al S&P 500
            k = f[2:]
            base.append(F['ret' + k][idx] - F['spy_ret' + k][idx] if ('spy_ret' + k) in F else np.full(len(idx), np.nan))
        else:
            base.append(F[f][idx] if f in F else np.full(len(idx), np.nan))
    return np.stack(base + extra, 1).astype(np.float32)


def ensure_regime(F: Dict[str, np.ndarray], reg: Optional[Dict[str, np.ndarray]]) -> None:
    if reg is not None:
        F.update(align_regime(reg, F['t']))
    else:
        for k in REGIME_KEYS:
            F[k] = np.full(len(F['c']), np.nan)
