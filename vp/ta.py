"""Indicadores técnicos con NumPy (sin pandas). Todos son causales: el valor en la barra i usa solo datos hasta i."""
from __future__ import annotations
from typing import Dict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

NAN = np.nan


def shift(x: np.ndarray, k: int) -> np.ndarray:
    """Valor de hace k barras (NaN al principio)."""
    if k <= 0:
        return x
    out = np.full(len(x), NAN)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), NAN)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def rmax(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), NAN)
    if len(x) >= n:
        out[n - 1:] = swv(x, n).max(axis=1)
    return out


def rmin(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), NAN)
    if len(x) >= n:
        out[n - 1:] = swv(x, n).min(axis=1)
    return out


def rstd(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), NAN)
    if len(x) >= n:
        out[n - 1:] = swv(x, n).std(axis=1)
    return out


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Media móvil de Wilder (la de ATR y RSI): arranca con la media simple de las n primeras barras."""
    out = np.full(len(x), NAN)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = shift(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    return tr


def rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ru, rd = wilder(up, n), wilder(dn, n)
    with np.errstate(invalid='ignore', divide='ignore'):
        rs = ru / rd
        return np.where(rd == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))


def band_pctile(w: np.ndarray, n: int = 120) -> np.ndarray:
    """Percentil de cada valor de w dentro de sus n-1 valores anteriores (0 = el más bajo)."""
    out = np.full(len(w), NAN)
    if len(w) >= n:
        win = swv(w, n)
        frac = (win[:, :-1] < win[:, -1:]).mean(axis=1)
        out[n - 1:] = np.where(np.isnan(win).any(axis=1), NAN, frac)
    return out


def prepare(b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Barras (t, o, h, l, c, v) → diccionario de series: las propias más ATR, medias, RSI y distancias. Hacen falta ≥ 60 barras."""
    o, h, l, c, v = (np.asarray(b[k], dtype=np.float64) for k in ('o', 'h', 'l', 'c', 'v'))
    F: Dict[str, np.ndarray] = {'t': np.asarray(b['t']), 'o': o, 'h': h, 'l': l, 'c': c, 'v': v}
    tr = true_range(h, l, c)
    F['tr'] = tr
    F['atr'] = wilder(tr, 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        F['atrp'] = F['atr'] / c * 100.0
        F['sma20'], F['sma50'], F['sma200'] = sma(c, 20), sma(c, 50), sma(c, 200)
        F['sd20'] = rstd(c, 20)
        F['rsi2'], F['rsi14'] = rsi(c, 2), rsi(c, 14)
        F['ret5'] = (c / shift(c, 5) - 1.0) * 100.0
        F['ret20'] = (c / shift(c, 20) - 1.0) * 100.0
        F['dist200'] = (c / F['sma200'] - 1.0) * 100.0
        hi, lo = rmax(h, 252), rmin(l, 252)
        F['pos52'] = np.where(hi > lo, (c - lo) / (hi - lo), NAN)
    return F
