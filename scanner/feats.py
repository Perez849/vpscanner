"""
feats.py — Indicadores 100% CAUSALES: el valor en la barra t usa solo datos hasta t.

Es el cambio de fondo respecto a la versión anterior: nada de perfiles calculados con
barras futuras ni de pivotes que necesitan 20 barras posteriores para existir.
El Volume Profile se conserva como "valor justo" pero calculado con una ventana móvil
de las últimas W barras (hasta hoy incluido), que es exactamente lo que se puede
saber en directo.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

Bars = Dict[str, np.ndarray]


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Media de Wilder (RMA) con NaN inicial tolerado."""
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


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
        for k in (2, 3, 5, 10, 20):
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

    for k in (3, 5, 10, 20):
        F[f'lowest{k}'] = (c <= pd.Series(c).rolling(k, min_periods=k).min().to_numpy()).astype(np.float64)
        F[f'highest{k}'] = (c >= pd.Series(c).rolling(k, min_periods=k).max().to_numpy()).astype(np.float64)
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
