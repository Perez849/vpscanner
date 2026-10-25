"""Patrones de velas (30 clásicos + 31 ampliados + 4 de consenso). Cada uno devuelve un vector booleano: True en la barra que completa el patrón. Causal: solo mira esa barra y las 4 anteriores."""
from __future__ import annotations
from typing import Dict

import numpy as np

from .catalog import CATALOG
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
        # ── ampliación: más formaciones de 1 a 5 velas ────────────────────────────
        o3, c3, h3, l3 = shift(o, 3), shift(c, 3), shift(h, 3), shift(l, 3)
        o4, c4, h4, l4 = shift(o, 4), shift(c, 4), shift(h, 4), shift(l, 4)
        h2, l2 = shift(h, 2), shift(l, 2)
        rng1, rng2, body3 = shift(rng, 1), shift(rng, 2), shift(body, 3)
        g3, r3 = b(green, 3), b(red, 3)
        atr3 = shift(atr, 3)
        dn4, up4 = b(dn_state, 4), b(up_state, 4)
        up_sh1, lo_sh1 = shift(up_sh, 1), shift(lo_sh, 1)
        long_b = body >= 1.0 * atr
        # cinturón: abre en un extremo y cierra casi en el otro, a favor de la reversión
        out['belt_bull'] = dn1 & green & long_b & (lo_sh <= 0.05 * rng) & (up_sh <= 0.3 * rng)
        out['belt_bear'] = up1 & red & long_b & (up_sh <= 0.05 * rng) & (lo_sh <= 0.3 * rng)
        # kicker: la vela contraria abre con hueco más allá de la apertura anterior y no la devuelve
        out['kicker_bull'] = r1 & green & (b1 >= 0.8 * atr1) & long_b & (o >= o1 + 0.1 * atr) & (l >= c1)
        out['kicker_bear'] = g1 & red & (b1 >= 0.8 * atr1) & long_b & (o <= o1 - 0.1 * atr) & (h <= c1)
        # contraataque: hueco en contra y cierre en el mismo nivel que la vela anterior
        out['counter_bull'] = dn1 & r1 & green & (b1 >= 0.8 * atr1) & long_b & (np.abs(c - c1) <= 0.1 * atr) & (o <= c1 - 0.5 * atr)
        out['counter_bear'] = up1 & g1 & red & (b1 >= 0.8 * atr1) & long_b & (np.abs(c - c1) <= 0.1 * atr) & (o >= c1 + 0.5 * atr)
        # harami con doji, harami del mismo color (paloma / halcón)
        dj_now = (body <= 0.1 * rng) & (rng >= 0.4 * atr)
        out['harami_cross_bull'] = dn2 & r1 & (b1 >= atr1) & dj_now & (hi_b < o1) & (lo_b > c1)
        out['harami_cross_bear'] = up2 & g1 & (b1 >= atr1) & dj_now & (lo_b > o1) & (hi_b < c1)
        out['homing_pigeon'] = dn2 & r1 & red & (b1 >= atr1) & (o < o1) & (c > c1) & (body <= 0.6 * b1)
        out['descending_hawk'] = up2 & g1 & green & (b1 >= atr1) & (o > o1) & (c < c1) & (body <= 0.6 * b1)
        # cierres iguales
        out['matching_low'] = dn2 & r1 & red & (np.abs(c - c1) <= 0.05 * atr) & (b1 >= 0.5 * atr1) & (body >= 0.5 * atr)
        out['matching_high'] = up2 & g1 & green & (np.abs(c - c1) <= 0.05 * atr) & (b1 >= 0.5 * atr1) & (body >= 0.5 * atr)
        # tres exteriores: envolvente confirmada
        eng_up = r2 & g1 & (o1 <= c2) & (c1 >= o2) & (shift(body, 1) > b2)
        eng_dn = g2 & r1 & (o1 >= c2) & (c1 <= o2) & (shift(body, 1) > b2)
        out['three_outside_up'] = dn3 & eng_up & green & (c > c1)
        out['three_outside_dn'] = up3 & eng_dn & red & (c < c1)
        # tres golpes: tres velas seguidas y una grande en contra que las borra (continuación)
        out['three_strike_bull'] = g3 & g2 & g1 & (c1 > c2) & (c2 > c3) & red & (o >= c1) & (c <= o3) & up1 & up2
        out['three_strike_bear'] = r3 & r2 & r1 & (c1 < c2) & (c2 < c3) & green & (o <= c1) & (c >= o3) & dn1 & dn2
        # bebé abandonado: estrella doji con huecos a ambos lados (sombras incluidas)
        dj1b = (b1 <= 0.1 * rng1) & (rng1 >= 0.4 * atr1)
        out['abandoned_bull'] = dn3 & r2 & (b2 >= atr2) & dj1b & (h1 < l2) & green & (l > h1) & (c > mid2)
        out['abandoned_bear'] = up3 & g2 & (b2 >= atr2) & dj1b & (l1 > h2) & red & (h < l1) & (c < mid2)
        # agotamiento de tres velas alcistas y duda con vela pequeña
        out['advance_block'] = up3 & green & g1 & g2 & (c > c1) & (c1 > c2) & (o < c1) & (o > o1) & (o1 < c2) & (o1 > o2) & (body < b1) & (b1 < b2) & (up_sh > up_sh1) & (up_sh >= 0.3 * rng)
        out['deliberation'] = up3 & green & g1 & g2 & (c > c1) & (b2 >= atr2) & (b1 >= atr1) & (body <= 0.4 * b1) & (o >= c1 - 0.1 * atr)
        # métodos de tres (cinco velas): pausa dentro del rango de la vela larga y continuación
        mid_in_up = (shift(h, 3) <= h4) & (shift(l, 3) >= l4) & (shift(h, 2) <= h4) & (shift(l, 2) >= l4) & (h1 <= h4) & (l1 >= l4)
        out['rising_methods'] = up4 & b(green, 4) & (shift(body, 4) >= atr) & mid_in_up & r3 & r2 & green & (c > c4) & (body >= atr)
        out['falling_methods'] = dn4 & b(red, 4) & (shift(body, 4) >= atr) & mid_in_up & g3 & g2 & red & (c < c4) & (body >= atr)
        # tasuki: hueco en la tendencia y vela contraria que entra en el hueco sin cerrarlo
        out['tasuki_up'] = up3 & g2 & g1 & (l1 > h2) & red & (o > o1) & (o < c1) & (c < o1) & (c > h2)
        out['tasuki_dn'] = dn3 & r2 & r1 & (h1 < l2) & green & (o < o1) & (o > c1) & (c > o1) & (c < l2)
        # dos cuervos con hueco
        out['two_crows_gap'] = up3 & g2 & (b2 >= atr2) & r1 & (o1 > c2) & (c1 > c2) & red & (o > o1) & (c < c1) & (c > c2)
        # sándwich: dos velas del mismo color con una contraria en medio y cierres iguales
        out['stick_sandwich_bull'] = dn3 & r2 & g1 & (c1 > o2) & red & (np.abs(c - c2) <= 0.05 * atr) & (body >= 0.5 * atr)
        out['stick_sandwich_bear'] = up3 & g2 & r1 & (c1 < o2) & green & (np.abs(c - c2) <= 0.05 * atr) & (body >= 0.5 * atr)
        # vela exterior (su rango envuelve el anterior) y ruptura de una vela interior
        out['outside_up'] = dn1 & green & (h > h1) & (l < l1) & (c > h1) & long_b
        out['outside_dn'] = up1 & red & (h > h1) & (l < l1) & (c < l1) & long_b
        inside1 = (h1 < h2) & (l1 > l2)
        out['inside_break_up'] = inside1 & (c > h2) & green & (rng2 >= 0.8 * atr2) & up2
        out['inside_break_dn'] = inside1 & (c < l2) & red & (rng2 >= 0.8 * atr2) & dn2
        # consenso: cuántas formaciones de velas del mismo sentido se completan en la misma barra
        keys = [k for k in out if k in CATALOG and CATALOG[k].fam == 'velas']
        res = {k: np.nan_to_num(v.astype(np.float64)).astype(bool) for k, v in out.items()}
        n_up = sum(res[k].astype(np.int16) for k in keys if CATALOG[k].dir > 0)
        n_dn = sum(res[k].astype(np.int16) for k in keys if CATALOG[k].dir < 0)
    res['consenso_alcista_2'] = (n_up >= 2) & (n_dn == 0)
    res['consenso_alcista_3'] = (n_up >= 3) & (n_dn == 0)
    res['consenso_bajista_2'] = (n_dn >= 2) & (n_up == 0)
    res['consenso_bajista_3'] = (n_dn >= 3) & (n_up == 0)
    return res
