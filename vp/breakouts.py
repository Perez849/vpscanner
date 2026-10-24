"""Rupturas y compresiones (13): máximos/mínimos de 55, 100 y 252 sesiones, Bollinger y ATR comprimidos, caja de Darvas y base junto al máximo anual.
Cada una devuelve un vector booleano (True en la barra de ruptura) y, aparte, la geometría de la barra para dibujarla."""
from __future__ import annotations
from typing import Dict

import numpy as np

from .ta import band_pctile, rmax, rmin, rstd, shift, sma, true_range, wilder


def detect(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    h, l, c, atr = F['h'], F['l'], F['c'], F['atr']
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for p in (55, 100, 252):
            out[f'donch{p}_up'] = c > shift(rmax(h, p), 1)
            out[f'donch{p}_dn'] = c < shift(rmin(l, p), 1)
        s20, sd = sma(c, 20), rstd(c, 20)
        bbp = band_pctile(4 * sd / s20, 120)
        sq = np.nan_to_num(np.maximum.reduce([shift((bbp <= 0.10).astype(float), k) for k in range(1, 6)]), nan=0.0) > 0.5     # estuvo comprimido en las 5 barras previas
        out['bb_squeeze_up'] = sq & (c > s20 + 2 * sd)
        out['bb_squeeze_dn'] = sq & (c < s20 - 2 * sd)
        trf = true_range(h, l, c)
        a5, a50 = wilder(trf, 5), wilder(trf, 50)
        cont = np.minimum.reduce([shift((a5 / a50 < 0.6).astype(float), k) for k in range(1, 6)]) > 0.5
        out['atr_squeeze_up'] = cont & (c > shift(rmax(h, 20), 1))
        out['atr_squeeze_dn'] = cont & (c < shift(rmin(l, 20), 1))
        bh, bl = shift(rmax(h, 15), 1), shift(rmin(l, 15), 1)
        tight = (bh - bl) <= 3.0 * atr
        out['box_up'] = tight & (c > bh)
        out['box_dn'] = tight & (c < bl)
        hi252 = shift(rmax(h, 252), 1)
        r20 = (shift(rmax(h, 20), 1) - shift(rmin(l, 20), 1)) / c
        out['base_near_high'] = (shift(c, 1) >= 0.95 * hi252) & (r20 <= 0.08) & (c > shift(rmax(h, 20), 1))
    return {k: np.nan_to_num(v.astype(np.float64)).astype(bool) for k, v in out.items()}


def geometry(name: str, F: Dict[str, np.ndarray], i: int) -> Dict:
    """Niveles que dibuja la web para una ruptura en la barra i."""
    h, l = F['h'], F['l']
    if name.startswith('donch'):
        p = int(name[5:].split('_')[0])
        return {'level': float(h[i - p:i].max() if name.endswith('_up') else l[i - p:i].min())}
    if name.startswith('box_'):
        return {'hi': float(h[i - 15:i].max()), 'lo': float(l[i - 15:i].min()), 'i0': i - 15}
    if name == 'base_near_high':
        return {'hi': float(h[i - 20:i].max()), 'lo': float(l[i - 20:i].min()), 'i0': i - 20, 'hi52': float(h[max(i - 252, 0):i].max())}
    return {'hi': float(h[i - 20:i].max()), 'lo': float(l[i - 20:i].min())}        # compresiones de Bollinger y de ATR
