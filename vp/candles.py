"""Patrones de velas (30). Cada uno devuelve un vector booleano: True en la barra que completa el patrón. Causal: solo mira esa barra y las 2 anteriores."""
from __future__ import annotations
from typing import Dict

import numpy as np

from .ta import shift


def detect(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    o, h, l, c, atr, atrp, sma20 = F['o'], F['h'], F['l'], F['c'], F['atr'], F['atrp'], F['sma20']
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        rng, body = h - l, np.abs(c - o)
        hi_b, lo_b = np.maximum(o, c), np.minimum(o, c)
        up_sh, lo_sh = h - hi_b, lo_b - l
        green, red = c > o, c < o
        # tendencia previa: caída (subida) de más de 1,5 ATR en 5 sesiones y precio bajo (sobre) su media de 20
        dn_state = (c / shift(c, 5) - 1.0 < -1.5 * atrp / 100.0) & (c < sma20)
        up_state = (c / shift(c, 5) - 1.0 > 1.5 * atrp / 100.0) & (c > sma20)
        b = lambda a, k: shift(a.astype(np.float64), k) > 0.5
        dn1, dn2, dn3 = b(dn_state, 1), b(dn_state, 2), b(dn_state, 3)
        up1, up2, up3 = b(up_state, 1), b(up_state, 2), b(up_state, 3)
        o1, c1, h1, l1, b1 = shift(o, 1), shift(c, 1), shift(h, 1), shift(l, 1), shift(body, 1)
        o2, c2, b2 = shift(o, 2), shift(c, 2), shift(body, 2)
        g1, r1, g2, r2 = b(green, 1), b(red, 1), b(green, 2), b(red, 2)
        atr1, atr2 = shift(atr, 1), shift(atr, 2)
        big = rng >= 0.8 * atr
        small_body = body <= 0.35 * rng
        hammer_shape = big & small_body & (lo_sh >= 0.6 * rng) & (up_sh <= 0.25 * rng)
        inv_shape = big & small_body & (up_sh >= 0.6 * rng) & (lo_sh <= 0.25 * rng)
        out['hammer'] = dn1 & hammer_shape
        out['inv_hammer'] = dn1 & inv_shape
        out['shooting_star'] = up1 & inv_shape
        out['hanging_man'] = up1 & hammer_shape
        out['bull_engulf'] = dn2 & r1 & green & (o <= c1) & (c >= o1) & (body > b1) & (b1 >= 0.3 * atr1)
        out['bear_engulf'] = up2 & g1 & red & (o >= c1) & (c <= o1) & (body > b1) & (b1 >= 0.3 * atr1)
        mid1 = (o1 + c1) / 2.0
        out['piercing'] = dn2 & r1 & green & ((o < l1) | (o < c1)) & (c > mid1) & (c < o1) & (b1 >= 0.8 * atr1)
        out['dark_cloud'] = up2 & g1 & red & ((o > h1) | (o > c1)) & (c < mid1) & (c > o1) & (b1 >= 0.8 * atr1)
        mid2 = (o2 + c2) / 2.0
        star_small = b1 <= 0.4 * b2
        out['morning_star'] = dn3 & r2 & (b2 >= atr2) & star_small & (np.maximum(o1, c1) < c2) & green & (c > mid2)
        out['evening_star'] = up3 & g2 & (b2 >= atr2) & star_small & (np.minimum(o1, c1) > c2) & red & (c < mid2)
        out['soldiers'] = dn3 & green & g1 & g2 & (c > c1) & (c1 > c2) & (o > o1) & (o < c1) & (o1 > o2) & (o1 < c2) & (up_sh <= 0.3 * body)
        out['crows'] = up3 & red & r1 & r2 & (c < c1) & (c1 < c2) & (o < o1) & (o > c1) & (o1 < o2) & (o1 > c2) & (lo_sh <= 0.3 * body)
        out['harami_bull'] = dn2 & r1 & (b1 >= atr1) & green & (o > c1) & (c < o1) & (body <= 0.5 * b1)
        out['harami_bear'] = up2 & g1 & (b1 >= atr1) & red & (o < c1) & (c > o1) & (body <= 0.5 * b1)
        out['tweezer_bottom'] = dn2 & r1 & green & (np.abs(l - l1) <= 0.1 * atr)
        out['tweezer_top'] = up2 & g1 & red & (np.abs(h - h1) <= 0.1 * atr)
        doji = big & (body <= 0.1 * rng)
        out['dragonfly'] = dn1 & doji & (lo_sh >= 0.6 * rng) & (up_sh <= 0.1 * rng)
        out['gravestone'] = up1 & doji & (up_sh >= 0.6 * rng) & (lo_sh <= 0.1 * rng)
        out['marubozu_bull'] = green & (body >= 0.9 * rng) & (body >= 1.5 * atr)
        out['marubozu_bear'] = red & (body >= 0.9 * rng) & (body >= 1.5 * atr)
        out['keyrev_bull'] = dn1 & (l < l1) & (c > h1)
        out['keyrev_bear'] = up1 & (h > h1) & (c < l1)
        # doji y peonzas con mechas a ambos lados (los libélula y lápida ya están arriba)
        dj = big & (body <= 0.1 * rng) & (up_sh >= 0.2 * rng) & (lo_sh >= 0.2 * rng)
        sp = big & (body <= 0.25 * rng) & (body > 0.1 * rng) & (up_sh >= 0.25 * rng) & (lo_sh >= 0.25 * rng)
        out['doji_bull'], out['doji_bear'] = dn1 & dj, up1 & dj
        out['spin_bull'], out['spin_bear'] = dn1 & sp, up1 & sp
        dj1 = (b1 <= 0.1 * shift(rng, 1)) & (shift(rng, 1) >= 0.5 * atr1)
        out['doji_star_bull'] = dn3 & r2 & (b2 >= atr2) & dj1 & (np.maximum(o1, c1) < c2) & green & (c > mid2)
        out['doji_star_bear'] = up3 & g2 & (b2 >= atr2) & dj1 & (np.minimum(o1, c1) > c2) & red & (c < mid2)
        # tres interiores: harami + cierre de confirmación más allá de la apertura de la primera vela
        out['three_inside_up'] = dn3 & r2 & (b2 >= atr2) & g1 & (o1 > c2) & (c1 < o2) & (c > o2) & green
        out['three_inside_dn'] = up3 & g2 & (b2 >= atr2) & r1 & (o1 < c2) & (c1 > o2) & (c < o2) & red
    return {k: np.nan_to_num(v.astype(np.float64)).astype(bool) for k, v in out.items()}
