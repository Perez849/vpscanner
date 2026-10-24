#!/usr/bin/env python3
"""
charts.py — CHARTISMO: velas, rupturas/compresiones y figuras (banderas, doble suelo/techo, hombro-cabeza-hombro, triángulos, cuñas, taza con asa, diamante…)
(comando `charts`; NO interviene en producción).

Cada patrón se detecta de forma CAUSAL: la señal existe en el cierre de la barra i usando solo datos hasta i (los pivotes de las figuras se confirman K barras
después de formarse, así que nunca se mira al futuro). Se entra en la apertura siguiente y se compara con entrar al azar en las mismas acciones y el mismo año.

Universo: las N acciones más negociadas EN CADA MOMENTO (dinero negociado medio de 60 sesiones, calculado con datos hasta ese día): «valores grandes y populares» sin
elegirlos mirando cuáles han subido después.
"""
from __future__ import annotations
import math
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

import feats

K_PIV = 4                    # un pivote en j se confirma en j + K_PIV
COOLDOWN = 5                 # barras mínimas entre dos señales del mismo patrón en el mismo valor

# nombre → (dirección, familia, descripción)
PATTERNS: Dict[str, Tuple[int, str, str]] = {}


def _reg(name: str, d: int, fam: str, desc: str):
    PATTERNS[name] = (d, fam, desc)


# velas
for _n, _d, _s in (('hammer', 1, 'martillo tras caída'), ('inv_hammer', 1, 'martillo invertido tras caída'), ('shooting_star', -1, 'estrella fugaz tras subida'),
                   ('hanging_man', -1, 'hombre colgado tras subida'), ('bull_engulf', 1, 'envolvente alcista'), ('bear_engulf', -1, 'envolvente bajista'),
                   ('piercing', 1, 'línea penetrante'), ('dark_cloud', -1, 'nube oscura'), ('morning_star', 1, 'estrella de la mañana'), ('evening_star', -1, 'estrella de la tarde'),
                   ('soldiers', 1, 'tres soldados blancos tras caída'), ('crows', -1, 'tres cuervos negros tras subida'), ('harami_bull', 1, 'harami alcista'),
                   ('harami_bear', -1, 'harami bajista'), ('tweezer_bottom', 1, 'pinzas en suelo'), ('tweezer_top', -1, 'pinzas en techo'),
                   ('dragonfly', 1, 'doji libélula tras caída'), ('gravestone', -1, 'doji lápida tras subida'), ('marubozu_bull', 1, 'marubozu alcista largo'),
                   ('marubozu_bear', -1, 'marubozu bajista largo'), ('keyrev_bull', 1, 'reversión clave alcista'), ('keyrev_bear', -1, 'reversión clave bajista')):
    _reg(_n, _d, 'velas', _s)
# rupturas y compresiones
for _p in (20, 55, 100, 252):
    _reg(f'donch{_p}_up', 1, 'rupturas', f'máximo de {_p} sesiones'); _reg(f'donch{_p}_dn', -1, 'rupturas', f'mínimo de {_p} sesiones')
for _n, _d, _s in (('bb_squeeze_up', 1, 'Bollinger comprimido y ruptura al alza'), ('bb_squeeze_dn', -1, 'Bollinger comprimido y ruptura a la baja'),
                   ('nr7_up', 1, 'NR7 y ruptura al alza'), ('nr7_dn', -1, 'NR7 y ruptura a la baja'), ('inside_up', 1, 'vela interior y ruptura al alza'),
                   ('inside_dn', -1, 'vela interior y ruptura a la baja'), ('atr_squeeze_up', 1, 'ATR comprimido y ruptura al alza'), ('atr_squeeze_dn', -1, 'ATR comprimido y ruptura a la baja'),
                   ('box_up', 1, 'caja estrecha (Darvas) y ruptura al alza'), ('box_dn', -1, 'caja estrecha y ruptura a la baja'),
                   ('base_near_high', 1, 'base apretada junto al máximo anual y ruptura')):
    _reg(_n, _d, 'rupturas', _s)
# figuras
for _n, _d, _s in (('double_bottom', 1, 'doble suelo (ruptura del cuello)'), ('double_top', -1, 'doble techo (ruptura del cuello)'), ('hs_top', -1, 'hombro-cabeza-hombro'),
                   ('hs_inverse', 1, 'hombro-cabeza-hombro invertido'), ('asc_triangle', 1, 'triángulo ascendente'), ('desc_triangle', -1, 'triángulo descendente'),
                   ('sym_tri_up', 1, 'triángulo simétrico, ruptura al alza'), ('sym_tri_dn', -1, 'triángulo simétrico, ruptura a la baja'), ('rising_wedge', -1, 'cuña ascendente (bajista)'),
                   ('falling_wedge', 1, 'cuña descendente (alcista)'), ('trendline_up', 1, 'ruptura de la directriz bajista'), ('trendline_dn', -1, 'ruptura de la directriz alcista'),
                   ('support_bounce', 1, 'rebote en soporte de 3 toques'), ('resistance_reject', -1, 'rechazo en resistencia de 3 toques'), ('vcp', 1, 'contracción de volatilidad (VCP)'),
                   ('cup_handle', 1, 'taza con asa'), ('diamond_bull', 1, 'diamante de suelo'), ('diamond_bear', -1, 'diamante de techo'), ('bull_flag', 1, 'bandera alcista'),
                   ('bear_flag', -1, 'bandera bajista'), ('bull_pennant', 1, 'banderín alcista'), ('bear_pennant', -1, 'banderín bajista'), ('island_bull', 1, 'isla de suelo'),
                   ('island_bear', -1, 'isla de techo'), ('triple_bottom', 1, 'triple suelo'), ('triple_top', -1, 'triple techo'), ('rect_up', 1, 'rectángulo, ruptura al alza'),
                   ('rect_dn', -1, 'rectángulo, ruptura a la baja')):
    _reg(_n, _d, 'figuras', _s)
for _n in ('double_bottom', 'double_top', 'triple_bottom', 'triple_top', 'hs_top', 'hs_inverse', 'asc_triangle', 'desc_triangle', 'sym_tri_up', 'sym_tri_dn', 'rising_wedge', 'falling_wedge',
           'rect_up', 'rect_dn', 'bull_flag', 'bear_flag', 'bull_pennant', 'bear_pennant', 'cup_handle', 'trendline_up', 'trendline_dn'):
    _reg(_n + '_hq', PATTERNS[_n][0], 'calidad', PATTERNS[_n][2] + ' (solo calidad ≥ 75)')


def _sh(x: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k, np.nan), x[:-k]]) if k > 0 else x


def _rmax(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()


def _rmin(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()


def candles(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    o, h, l, c, atr, atrp = F['o'], F['h'], F['l'], F['c'], F['atr'], F['atrp']
    n = len(c)
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        rng, body = h - l, np.abs(c - o)
        hi_b, lo_b = np.maximum(o, c), np.minimum(o, c)
        up_sh, lo_sh = h - hi_b, lo_b - l
        green, red = c > o, c < o
        sma20 = F['sma20']
        dn_state = (c / _sh(c, 5) - 1.0 < -1.5 * atrp / 100.0) & (c < sma20)           # tendencia previa: caída de >1,5 ATR en 5 sesiones y bajo su media
        up_state = (c / _sh(c, 5) - 1.0 > 1.5 * atrp / 100.0) & (c > sma20)
        s = lambda a, k: _sh(a.astype(np.float64), k) > 0.5 if a.dtype == bool else _sh(a, k)
        dn1, dn2, dn3 = s(dn_state, 1), s(dn_state, 2), s(dn_state, 3)
        up1, up2, up3 = s(up_state, 1), s(up_state, 2), s(up_state, 3)
        o1, c1, h1, l1, b1 = _sh(o, 1), _sh(c, 1), _sh(h, 1), _sh(l, 1), _sh(body, 1)
        o2, c2, b2 = _sh(o, 2), _sh(c, 2), _sh(body, 2)
        g1, r1, g2, r2 = _sh(green.astype(float), 1) > 0.5, _sh(red.astype(float), 1) > 0.5, _sh(green.astype(float), 2) > 0.5, _sh(red.astype(float), 2) > 0.5
        atr1 = _sh(atr, 1)
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
        out['morning_star'] = dn3 & r2 & (b2 >= 1.0 * _sh(atr, 2)) & star_small & (np.maximum(o1, c1) < c2) & green & (c > mid2)
        out['evening_star'] = up3 & g2 & (b2 >= 1.0 * _sh(atr, 2)) & star_small & (np.minimum(o1, c1) > c2) & red & (c < mid2)
        gg = green & g1 & g2 & (c > c1) & (c1 > c2) & (o > o1) & (o < c1) & (o1 > o2) & (o1 < c2) & (up_sh <= 0.3 * body)
        rr = red & r1 & r2 & (c < c1) & (c1 < c2) & (o < o1) & (o > c1) & (o1 < o2) & (o1 > c2) & (lo_sh <= 0.3 * body)
        out['soldiers'] = dn3 & gg
        out['crows'] = up3 & rr
        out['harami_bull'] = dn2 & r1 & (b1 >= 1.0 * atr1) & green & (o > c1) & (c < o1) & (body <= 0.5 * b1)
        out['harami_bear'] = up2 & g1 & (b1 >= 1.0 * atr1) & red & (o < c1) & (c > o1) & (body <= 0.5 * b1)
        out['tweezer_bottom'] = dn2 & r1 & green & (np.abs(l - l1) <= 0.1 * atr)
        out['tweezer_top'] = up2 & g1 & red & (np.abs(h - h1) <= 0.1 * atr)
        doji = big & (body <= 0.1 * rng)
        out['dragonfly'] = dn1 & doji & (lo_sh >= 0.6 * rng) & (up_sh <= 0.1 * rng)
        out['gravestone'] = up1 & doji & (up_sh >= 0.6 * rng) & (lo_sh <= 0.1 * rng)
        out['marubozu_bull'] = green & (body >= 0.9 * rng) & (body >= 1.5 * atr)
        out['marubozu_bear'] = red & (body >= 0.9 * rng) & (body >= 1.5 * atr)
        out['keyrev_bull'] = dn1 & (l < l1) & (c > h1)
        out['keyrev_bear'] = up1 & (h > h1) & (c < l1)
    return {k: np.nan_to_num(v.astype(np.float64)).astype(bool) if v.dtype != bool else v for k, v in out.items()}


def breakouts(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    o, h, l, c, atr, v = F['o'], F['h'], F['l'], F['c'], F['atr'], F['v']
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for p in (20, 55, 100, 252):
            hh = _sh(_rmax(h, p), 1)
            ll = _sh(_rmin(l, p), 1)
            out[f'donch{p}_up'] = c > hh
            out[f'donch{p}_dn'] = c < ll
        s20 = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy()
        sd = pd.Series(c).rolling(20, min_periods=20).std(ddof=0).to_numpy()
        bbw = 4 * sd / s20
        bbp = pd.Series(bbw).rolling(120, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True).to_numpy() if len(c) > 130 else np.full(len(c), np.nan)
        sq = pd.Series((bbp <= 0.10).astype(float)).rolling(5, min_periods=5).max().to_numpy() > 0.5
        out['bb_squeeze_up'] = _sh(sq.astype(float), 1) > 0.5
        out['bb_squeeze_dn'] = out['bb_squeeze_up'] & (c < s20 - 2 * sd)
        out['bb_squeeze_up'] = out['bb_squeeze_up'] & (c > s20 + 2 * sd)
        rng = h - l
        nr7 = rng <= _rmin(rng, 7)
        n1 = _sh(nr7.astype(float), 1) > 0.5
        out['nr7_up'] = n1 & (c > _sh(h, 1))
        out['nr7_dn'] = n1 & (c < _sh(l, 1))
        ib = (h < _sh(h, 1)) & (l > _sh(l, 1))
        i1 = _sh(ib.astype(float), 1) > 0.5
        out['inside_up'] = i1 & (c > _sh(h, 1))
        out['inside_dn'] = i1 & (c < _sh(l, 1))
        trf = np.maximum.reduce([h - l, np.abs(h - _sh(c, 1)), np.abs(l - _sh(c, 1))])
        a5, a50 = feats._wilder(trf, 5), feats._wilder(trf, 50)
        cont = pd.Series((a5 / a50 < 0.6).astype(float)).rolling(5, min_periods=5).min().to_numpy() > 0.5
        c1 = _sh(cont.astype(float), 1) > 0.5
        out['atr_squeeze_up'] = c1 & (c > _sh(_rmax(h, 20), 1))
        out['atr_squeeze_dn'] = c1 & (c < _sh(_rmin(l, 20), 1))
        bh, bl = _sh(_rmax(h, 15), 1), _sh(_rmin(l, 15), 1)
        tight = (bh - bl) <= 3.0 * atr
        out['box_up'] = tight & (c > bh)
        out['box_dn'] = tight & (c < bl)
        hi252 = _sh(_rmax(h, 252), 1)
        r20 = (_sh(_rmax(h, 20), 1) - _sh(_rmin(l, 20), 1)) / c
        out['base_near_high'] = (_sh(c, 1) >= 0.95 * hi252) & (r20 <= 0.08) & (c > _sh(_rmax(h, 20), 1))
    return {k: np.nan_to_num(v.astype(np.float64)).astype(bool) if v.dtype != bool else v for k, v in out.items()}


def pivots(h: np.ndarray, l: np.ndarray, k: int = K_PIV) -> Tuple[np.ndarray, np.ndarray]:
    """Marcas de pivote en el índice j (máximo/mínimo de 2k+1 barras). Un pivote en j solo es conocido desde la barra j+k."""
    n = len(h)
    ph, pl = np.zeros(n, bool), np.zeros(n, bool)
    if n > 2 * k + 1:
        ph[k:n - k] = np.argmax(swv(h, 2 * k + 1), axis=1) == k
        pl[k:n - k] = np.argmin(swv(l, 2 * k + 1), axis=1) == k
    return ph, pl


def figures(F: Dict[str, np.ndarray], rec: Dict[str, Dict[int, dict]] | None = None) -> Dict[str, np.ndarray]:
    """Figuras (causales). Si se pasa `rec`, guarda en rec[nombre][barra] la geometría de cada señal (pivotes, cuello, líneas…) para dibujarla."""
    o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
    sma50, sma200 = F['sma50'], F['sma200']
    n = len(c)
    names = [k for k, v in PATTERNS.items() if v[1] in ('figuras', 'calidad')]
    out = {k: np.zeros(n, bool) for k in names}
    ph, pl = pivots(h, l)
    H: List[Tuple[int, float]] = []
    L: List[Tuple[int, float]] = []
    used: Dict[str, tuple] = {}
    atrp = F['atrp']
    with np.errstate(invalid='ignore', divide='ignore'):
        dn_state = (c / _sh(c, 5) - 1.0 < -1.5 * atrp / 100.0) & (c < F['sma20'])
        up_state = (c / _sh(c, 5) - 1.0 > 1.5 * atrp / 100.0) & (c > F['sma20'])

    def fire(name: str, key, i: int, **g) -> None:
        if used.get(name) != key:
            used[name] = key
            out[name][i] = True
            if rec is not None:
                rec.setdefault(name, {})[i] = g

    def line(p1, p2, x):
        return p2[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (x - p2[0])

    for i in range(max(K_PIV + 1, 30), n):
        j = i - K_PIV
        if ph[j]:
            H.append((j, h[j]))
        if pl[j]:
            L.append((j, l[j]))
        a = atr[i]
        if not (a == a) or a <= 0:
            continue
        ci, cp = c[i], c[i - 1]
        # ── doble suelo / techo
        if len(L) >= 2:
            (j1, v1), (j2, v2) = L[-2], L[-1]
            if 8 <= j2 - j1 <= 80 and abs(v1 - v2) <= max(0.03 * min(v1, v2), 1.5 * a):
                neck = h[j1:j2 + 1].max()
                if neck - max(v1, v2) >= 2.5 * a and ci > neck >= cp and i - j2 <= 40:
                    fire('double_bottom', (j1, j2), i, p1=(j1, v1), p2=(j2, v2), neck=float(neck), mid=(j1 + int(np.argmax(h[j1:j2 + 1])), float(neck)))
        if len(H) >= 2:
            (j1, v1), (j2, v2) = H[-2], H[-1]
            if 8 <= j2 - j1 <= 80 and abs(v1 - v2) <= max(0.03 * min(v1, v2), 1.5 * a):
                neck = l[j1:j2 + 1].min()
                if min(v1, v2) - neck >= 2.5 * a and ci < neck <= cp and i - j2 <= 40:
                    fire('double_top', (j1, j2), i, p1=(j1, v1), p2=(j2, v2), neck=float(neck), mid=(j1 + int(np.argmin(l[j1:j2 + 1])), float(neck)))
        # ── hombro-cabeza-hombro
        if len(H) >= 3 and len(L) >= 2:
            (ja, va), (jb, vb), (jc, vc) = H[-3:]
            if vb - max(va, vc) >= 1.0 * a and abs(va - vc) <= 0.25 * (vb - min(va, vc)) and 10 <= jc - ja <= 120 and i - jc <= 30:
                k1, k2 = ja + int(np.argmin(l[ja:jb + 1])), jb + int(np.argmin(l[jb:jc + 1]))
                t1, t2 = l[k1], l[k2]
                neck = (t1 + t2) / 2.0
                if ci < neck <= cp:
                    fire('hs_top', (ja, jb, jc), i, ls=(ja, va), hd=(jb, vb), rs=(jc, vc), t1=(k1, float(t1)), t2=(k2, float(t2)), neck=float(neck))
        if len(L) >= 3 and len(H) >= 2:
            (ja, va), (jb, vb), (jc, vc) = L[-3:]
            if min(va, vc) - vb >= 1.0 * a and abs(va - vc) <= 0.25 * (max(va, vc) - vb) and 10 <= jc - ja <= 120 and i - jc <= 30:
                k1, k2 = ja + int(np.argmax(h[ja:jb + 1])), jb + int(np.argmax(h[jb:jc + 1]))
                t1, t2 = h[k1], h[k2]
                neck = (t1 + t2) / 2.0
                if ci > neck >= cp:
                    fire('hs_inverse', (ja, jb, jc), i, ls=(ja, va), hd=(jb, vb), rs=(jc, vc), t1=(k1, float(t1)), t2=(k2, float(t2)), neck=float(neck))
        # ── triángulos, cuñas, directrices
        if len(H) >= 3 and len(L) >= 2:
            H3 = H[-3:]
            top = max(v for _, v in H3)
            if top - min(v for _, v in H3) <= 0.8 * a and 12 <= H3[-1][0] - H3[0][0] <= 90 and L[-1][0] > H3[0][0] and L[-2][0] > H3[0][0] - 10 and L[-1][1] > L[-2][1] + 0.3 * a:
                if ci > top >= cp and i - H3[-1][0] <= 40:
                    fire('asc_triangle', tuple(x for x, _ in H3), i, highs=[tuple(x) for x in H3], q1=L[-2], q2=L[-1], top=float(top))
        if len(L) >= 3 and len(H) >= 2:
            L3 = L[-3:]
            bot = min(v for _, v in L3)
            if max(v for _, v in L3) - bot <= 0.8 * a and 12 <= L3[-1][0] - L3[0][0] <= 90 and H[-1][0] > L3[0][0] and H[-2][0] > L3[0][0] - 10 and H[-1][1] < H[-2][1] - 0.3 * a:
                if ci < bot <= cp and i - L3[-1][0] <= 40:
                    fire('desc_triangle', tuple(x for x, _ in L3), i, lows=[tuple(x) for x in L3], p1=H[-2], p2=H[-1], bot=float(bot))
        if len(H) >= 2 and len(L) >= 2:
            p1, p2, q1, q2 = H[-2], H[-1], L[-2], L[-1]
            if p2[0] - p1[0] >= 6 and q2[0] - q1[0] >= 6 and abs(p2[0] - q2[0]) <= 40 and i - max(p2[0], q2[0]) <= 40:
                su, sl = (p2[1] - p1[1]) / (p2[0] - p1[0]), (q2[1] - q1[1]) / (q2[0] - q1[0])
                up_i, lo_i = line(p1, p2, i), line(q1, q2, i)
                up_p, lo_p = line(p1, p2, i - 1), line(q1, q2, i - 1)
                if up_i > lo_i and (up_i - lo_i) < (line(p1, p2, max(p1[0], q1[0])) - line(q1, q2, max(p1[0], q1[0]))):   # estrechándose
                    g = dict(p1=p1, p2=p2, q1=q1, q2=q2)
                    if su < -0.02 * a and sl > 0.02 * a:                                     # simétrico
                        if ci > up_i and cp <= up_p:
                            fire('sym_tri_up', (p1[0], p2[0], q1[0], q2[0]), i, **g)
                        elif ci < lo_i and cp >= lo_p:
                            fire('sym_tri_dn', (p1[0], p2[0], q1[0], q2[0]), i, **g)
                    elif su > 0 and sl > 0 and su < sl:                                      # cuña ascendente
                        if ci < lo_i and cp >= lo_p:
                            fire('rising_wedge', (p1[0], p2[0], q1[0], q2[0]), i, **g)
                    elif su < 0 and sl < 0 and su < sl:                                      # cuña descendente
                        if ci > up_i and cp <= up_p:
                            fire('falling_wedge', (p1[0], p2[0], q1[0], q2[0]), i, **g)
        if len(H) >= 2:
            p1, p2 = H[-2], H[-1]
            if p2[1] < p1[1] - 0.5 * a and 8 <= p2[0] - p1[0] <= 90 and i - p2[0] <= 40:
                if ci > line(p1, p2, i) and cp <= line(p1, p2, i - 1):
                    fire('trendline_up', (p1[0], p2[0]), i, p1=p1, p2=p2)
        if len(L) >= 2:
            q1, q2 = L[-2], L[-1]
            if q2[1] > q1[1] + 0.5 * a and 8 <= q2[0] - q1[0] <= 90 and i - q2[0] <= 40:
                if ci < line(q1, q2, i) and cp >= line(q1, q2, i - 1):
                    fire('trendline_dn', (q1[0], q2[0]), i, p1=q1, p2=q2)
        # ── soporte / resistencia de 3 toques
        if len(L) >= 3:
            L3 = L[-3:]
            sup = float(np.mean([v for _, v in L3]))
            if max(v for _, v in L3) - min(v for _, v in L3) <= 0.8 * a and L3[-1][0] - L3[0][0] >= 15 and i - L3[-1][0] >= 3 and i - L3[-1][0] <= 60:
                if sup - 0.8 * a <= l[i] <= sup + 0.3 * a and ci > sup and ci > o[i]:
                    fire('support_bounce', (L3[-1][0], i // 3), i, level=sup, touches=[tuple(x) for x in L3])
        if len(H) >= 3:
            H3 = H[-3:]
            res = float(np.mean([v for _, v in H3]))
            if max(v for _, v in H3) - min(v for _, v in H3) <= 0.8 * a and H3[-1][0] - H3[0][0] >= 15 and i - H3[-1][0] >= 3 and i - H3[-1][0] <= 60:
                if res - 0.3 * a <= h[i] <= res + 0.8 * a and ci < res and ci < o[i]:
                    fire('resistance_reject', (H3[-1][0], i // 3), i, level=res, touches=[tuple(x) for x in H3])
        # ── VCP
        if len(H) >= 3 and ci > sma50[i] > sma200[i] and H[-1][0] - H[-3][0] <= 120:
            depths, lows = [], []
            for k in (-3, -2, -1):
                jh = H[k][0]
                jn = H[k + 1][0] if k < -1 else i
                kk = jh + int(np.argmin(l[jh:jn + 1]))
                depths.append((H[k][1] - l[kk]) / H[k][1])
                lows.append((kk, float(l[kk])))
            if depths[0] > depths[1] > depths[2] and depths[2] <= 0.08 and depths[0] <= 0.40 and ci > H[-1][1] >= cp:
                fire('vcp', (H[-3][0], H[-1][0]), i, highs=[tuple(x) for x in H[-3:]], lows=lows, depths=[float(d) for d in depths])
        # ── taza con asa
        if ci > sma200[i] * 0.9:
            for W in (60, 100):
                if i < W + 5:
                    continue
                hh12 = h[i - 12:i].max()
                if not ci > hh12:
                    continue
                lj = i - W + int(np.argmax(h[i - W:i - W + W // 4]))
                left = h[lj]
                seg = l[i - W:i]
                bot = seg.min()
                bi = int(np.argmin(seg))
                depth = (left - bot) / left
                if 0.12 <= depth <= 0.35 and 0.3 * W <= bi <= 0.75 * W:
                    rj = i - W // 4 + int(np.argmax(h[i - W // 4:i]))
                    right = h[rj]
                    if 0.92 * left <= right <= 1.06 * left and ci >= right * 0.995:
                        hj = i - 12 + int(np.argmin(l[i - 12:i]))
                        handle_low = l[hj]
                        if (right - handle_low) <= 0.4 * (right - bot) and ci > hh12:
                            xs = np.arange(W)
                            cs = c[i - W:i]
                            cf = np.polyfit(xs, cs, 2)
                            if cf[0] > 0:
                                fit = np.polyval(cf, xs)
                                r2 = 1.0 - ((cs - fit) ** 2).sum() / ((cs - cs.mean()) ** 2).sum()
                                if r2 >= 0.5:
                                    fire('cup_handle', (i - W, bi), i, left=(lj, float(left)), bottom=(i - W + bi, float(bot)), right=(rj, float(right)), handle=(hj, float(handle_low)),
                                         fit=[float(x) for x in cf], w0=i - W, W=W)
                                    break
        # ── diamante
        if i >= 60:
            W = 40
            hs_, ls_ = h[i - W:i], l[i - W:i]
            w = np.array([hs_[b * 5:(b + 1) * 5].max() - ls_[b * 5:(b + 1) * 5].min() for b in range(8)])
            pk = int(np.argmax(w))
            if 2 <= pk <= 5 and w[0] <= 0.6 * w[pk] and w[-1] <= 0.55 * w[pk] and w[:pk + 1].argmax() == pk and (np.diff(w[:pk + 1]) > -0.15 * w[pk]).all() and (np.diff(w[pk:]) < 0.15 * w[pk]).all():
                top10, bot10 = h[i - 10:i].max(), l[i - 10:i].min()
                env_hi = [(i - W + b * 5 + 2, float(hs_[b * 5:(b + 1) * 5].max())) for b in range(8)]
                env_lo = [(i - W + b * 5 + 2, float(ls_[b * 5:(b + 1) * 5].min())) for b in range(8)]
                if ci > top10 and dn_state[i - W - 1]:
                    fire('diamond_bull', (i // 10,), i, hi=env_hi, lo=env_lo, top=float(top10), bot=float(bot10), width=float(w.max()))
                elif ci < bot10 and up_state[i - W - 1]:
                    fire('diamond_bear', (i // 10,), i, hi=env_hi, lo=env_lo, top=float(top10), bot=float(bot10), width=float(w.max()))
        # ── banderas y banderines
        for f in (5, 8, 12):
            pe = i - f
            fh, fl = h[pe + 1:i], l[pe + 1:i]
            if len(fh) < 3:
                continue
            for p in (5, 8, 12):
                ps = pe - p
                if ps < 0:
                    continue
                pole = c[pe] - c[ps]
                if abs(pole) < 5.0 * atr[pe]:
                    continue
                zr = fh.max() - fl.min()
                half = max(len(fh) // 2, 1)
                contraction = (fh[:half].max() - fl[:half].min()) > 1.3 * (fh[half:].max() - fl[half:].min()) if len(fh) - half >= 2 else False
                slope = np.polyfit(np.arange(len(fh)), c[pe + 1:i], 1)[0]
                g = dict(ps=(ps, float(c[ps])), pe=(pe, float(c[pe])), fh=float(fh.max()), fl=float(fl.min()), z0=pe + 1, pole=float(pole))
                if pole > 0 and zr <= 0.6 * pole and ci > fh.max() >= cp - 1e-12 and c[i - 1] <= fh.max():
                    if contraction:
                        fire('bull_pennant', (pe, p), i, **g)
                    elif slope <= 0.1 * a:
                        fire('bull_flag', (pe, p), i, **g)
                elif pole < 0 and zr <= 0.6 * (-pole) and ci < fl.min() and c[i - 1] >= fl.min():
                    if contraction:
                        fire('bear_pennant', (pe, p), i, **g)
                    elif slope >= -0.1 * a:
                        fire('bear_flag', (pe, p), i, **g)
        # ── isla
        for aidx in range(i - 5, i):
            if aidx < 2:
                continue
            if h[aidx] < l[aidx - 1] and dn_state[aidx - 1] and l[i] > h[aidx:i].max():
                fire('island_bull', (aidx,), i, a=aidx, gap_from=float(l[aidx - 1]), lo=float(l[aidx:i + 1].min()), hi=float(h[aidx:i].max()))
                break
            if l[aidx] > h[aidx - 1] and up_state[aidx - 1] and h[i] < l[aidx:i].min():
                fire('island_bear', (aidx,), i, a=aidx, gap_from=float(h[aidx - 1]), lo=float(l[aidx:i].min()), hi=float(h[aidx:i + 1].max()))
                break
    # el motor geométrico (shapes.py) sustituye a los detectores de dos pivotes de las figuras de pivotes/rectas
    import shapes
    r2: Dict = {} if rec is not None else None
    new = shapes.figures(F, r2)
    for k, v in new.items():
        out[k] = v
        if rec is not None:
            rec[k] = r2.get(k, {})
    if rec is not None and '_forming' in r2:
        rec['_forming'] = r2['_forming']
    return out


def signals(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Todas las señales de patrones (booleanos por barra, causales)."""
    s: Dict[str, np.ndarray] = {}
    s.update(candles(F))
    s.update(breakouts(F))
    s.update(figures(F))
    return {k: v for k, v in s.items() if k in PATTERNS}


def cooldown(idx: np.ndarray, gap: int = COOLDOWN) -> np.ndarray:
    """Se queda con la primera señal de cada racha: ignora las que llegan a menos de `gap` barras de la última conservada."""
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=np.int64)


# ═════════════════════════════════════════════════════════════════════════
#  Estudio
# ═════════════════════════════════════════════════════════════════════════
STOCK_GROUPS = ('us_large', 'us_mid', 'us_small', 'thematic')
HORIZONS = (5, 10, 20)
COST = 0.15                      # % ida y vuelta para acciones grandes
KS = (50, 300)
PRIMARY = 10


def membership(data: Dict, syms: List[str], ks=KS, window: int = 60):
    """Para cada día, las K acciones más negociadas (dinero medio de `window` sesiones con datos hasta ese día). Devuelve (días, {K: matriz bool [días, símbolos]})."""
    ds = {s: (data[s]['t'] // 86400).astype(np.int64) for s in syms}
    days = np.unique(np.concatenate(list(ds.values())))
    dv = np.full((len(days), len(syms)), np.nan, dtype=np.float32)
    for j, s in enumerate(syms):
        b = data[s]
        x = pd.Series(b['c'].astype(np.float64) * b['v'].astype(np.float64)).rolling(window, min_periods=window).mean().to_numpy()
        dv[np.searchsorted(days, ds[s]), j] = x
    order = np.argsort(-np.nan_to_num(dv, nan=-1.0), axis=1)
    out = {}
    for K in ks:
        m = np.zeros(dv.shape, bool)
        kk = min(K, dv.shape[1])
        top = order[:, :kk]
        rows = np.repeat(np.arange(len(days)), kk)
        m[rows, top.ravel()] = True
        out[K] = m & np.isfinite(dv)
    return days, out


def _t_cluster(x: np.ndarray, day: np.ndarray) -> float:
    n = len(x)
    if n < 5:
        return float('nan')
    u, inv = np.unique(day, return_inverse=True)
    r = np.bincount(inv, weights=x - x.mean())
    se = math.sqrt(float((r ** 2).sum())) / n
    return float(x.mean() / (se + 1e-12))


class Panel:
    """Todas las barras de todos los valores estudiados en un único vector, con retornos futuros, pertenencia al universo y señales por patrón."""
    def __init__(self):
        self.sym: List[np.ndarray] = []
        self.day: List[np.ndarray] = []
        self.R: Dict[Tuple[str, int], List[np.ndarray]] = {}
        self.mem: Dict[int, List[np.ndarray]] = {K: [] for K in KS}
        self.events: Dict[str, List[np.ndarray]] = {k: [] for k in PATTERNS}
        self.off = [0]
        self.names: List[str] = []

    def add(self, name: str, sid: int, F: Dict[str, np.ndarray], mem: Dict[int, np.ndarray], bracket: bool = True) -> None:
        import simulate as SM
        n = len(F['c'])
        o, c = F['o'], F['c']
        off = self.off[-1]
        entry = np.full(n, np.nan); entry[:-1] = o[1:]
        for hz in HORIZONS:
            ex = np.full(n, np.nan); ex[:n - hz] = c[hz:]
            ret = (ex / entry - 1.0) * 100.0
            self.R.setdefault(('long', hz), []).append(ret - COST)
            self.R.setdefault(('short', hz), []).append(-ret - COST)
        v = SM.Variant('atr', 2.0, 2.0, 10)
        for d, key in ((1, 'long'), (-1, 'short')):
            pn = np.full(n, np.nan)
            idx = np.arange(60, max(n - 12, 60))
            if bracket and len(idx):
                pn[idx] = SM.simulate(F, idx, d, [v], COST)[v.name].pnl
            self.R.setdefault((key, 0), []).append(pn)
        self.sym.append(np.full(n, sid, np.int32))
        self.day.append((F['t'] // 86400).astype(np.int64))
        for K in KS:
            self.mem[K].append(mem[K])
        for name_, sig in signals(F).items():
            idx = cooldown(np.flatnonzero(sig))
            if len(idx):
                self.events[name_].append(idx + off)
        self.off.append(off + n)
        self.names.append(name)

    def finish(self) -> None:
        self.symv = np.concatenate(self.sym)
        self.dayv = np.concatenate(self.day)
        self.year = (np.datetime64('1970-01-01') + self.dayv.astype('timedelta64[D]')).astype('datetime64[Y]').astype(int) + 1970
        self.Rv = {k: np.concatenate(v) for k, v in self.R.items()}
        self.memv = {K: np.concatenate(v) for K, v in self.mem.items()}
        self.ev = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in self.events.items()}
        self.offa = np.array(self.off[:-1], dtype=np.int64)
        self.nbar = np.diff(np.array(self.off, dtype=np.int64))
        y0 = int(self.year.min())
        self.y0 = y0
        self.base: Dict[Tuple[int, str, int], np.ndarray] = {}
        ny = int(self.year.max()) - y0 + 1
        for K in KS:
            for key, r in self.Rv.items():
                ok = self.memv[K] & np.isfinite(r)
                s = np.bincount(self.year[ok] - y0, weights=r[ok], minlength=ny)
                c = np.bincount(self.year[ok] - y0, minlength=ny)
                self.base[(K,) + key] = np.where(c > 0, s / np.maximum(c, 1), np.nan)

    def stat(self, idx: np.ndarray, K: int, key: Tuple[str, int]) -> Dict:
        r = self.Rv[key][idx]
        ok = self.memv[K][idx] & np.isfinite(r)
        if ok.sum() < 5:
            return {'n': int(ok.sum())}
        i2 = idx[ok]
        x = r[ok]
        ex = x - self.base[(K,) + key][self.year[i2] - self.y0]
        yrs = self.year[i2]
        pos = tot = 0
        for Y in np.unique(yrs):
            m = yrs == Y
            if m.sum() >= 8:
                tot += 1
                pos += ex[m].mean() > 0
        return {'n': int(ok.sum()), 'wr': float((x > 0).mean() * 100), 'mean': float(x.mean()), 'med': float(np.median(x)), 'ex': float(ex.mean()),
                't': _t_cluster(ex, self.dayv[i2]), 'pos': int(pos), 'tot': int(tot)}

    def null_max_t(self, K: int, key: Tuple[str, int], reps: int, seed: int = 5) -> np.ndarray:
        """Distribución del mayor |t| entre todos los patrones cuando las señales de cada valor se desplazan en el tiempo (aleatoriamente, conservando su agrupación)."""
        rng = np.random.default_rng(seed)
        r_all, mem = self.Rv[key], self.memv[K]
        base = self.base[(K,) + key]
        evs = [(k, v, self.symv[v]) for k, v in self.ev.items() if len(v) >= 20]
        out = np.empty(reps)
        for q in range(reps):
            sh = np.array([rng.integers(30, max(nb - 30, 31)) for nb in self.nbar])
            best = 0.0
            for k, v, sv in evs:
                o = self.offa[sv]
                nb = self.nbar[sv]
                j = o + ((v - o + sh[sv]) % nb)
                r = r_all[j]
                ok = mem[j] & np.isfinite(r)
                if ok.sum() < 20:
                    continue
                ex = r[ok] - base[self.year[j[ok]] - self.y0]
                t = abs(_t_cluster(ex, self.dayv[j[ok]]))
                if t > best:
                    best = t
            out[q] = best
        return out


def run(uni: Dict, data: Dict, args, out_dir: str):
    import pit
    uni2, data2, row_ok, info = pit.prepare(uni, data, args)
    syms = [s for s in uni2 if uni2[s].get('group') in STOCK_GROUPS and not uni2[s].get('aux') and s in data2 and len(data2[s]['c']) > 300 and '.' not in s]      # sin cotizaciones extranjeras (Londres cotiza en peniques: su «dinero negociado» sale inflado ×100)
    print(f'\n######## CHARTISMO · {len(syms)} acciones con datos · {len(PATTERNS)} patrones (velas {sum(1 for v in PATTERNS.values() if v[1] == "velas")}, '
          f'rupturas {sum(1 for v in PATTERNS.values() if v[1] == "rupturas")}, figuras {sum(1 for v in PATTERNS.values() if v[1] == "figuras")})', flush=True)
    days, mem = membership(data2, syms, KS)
    ever = np.zeros(len(syms), bool)
    for K in KS:
        ever |= mem[K].sum(0) >= 40
    work = [j for j in range(len(syms)) if ever[j]]
    print(f'  universo móvil: las {KS[0]} y las {KS[1]} acciones más negociadas en cada fecha · {len(work)} valores han estado en alguno · {len(days)} sesiones', flush=True)
    dvm = np.full((len(days), len(syms)), np.nan, dtype=np.float32)
    for j, s_ in enumerate(syms):
        b_ = data2[s_]
        dvm[np.searchsorted(days, (b_['t'] // 86400).astype(np.int64)), j] = pd.Series(b_['c'].astype(np.float64) * b_['v'].astype(np.float64)).rolling(60, min_periods=60).mean().to_numpy()
    for d_str in ('2018-06-29', '2021-06-30', '2024-06-28', '2026-09-30'):
        r = int(np.searchsorted(days, (np.datetime64(d_str) - np.datetime64('1970-01-01')).astype(int)))
        r = min(r, len(days) - 1)
        o_ = np.argsort(-np.nan_to_num(dvm[r], nan=-1.0))[:12]
        print(f'  las 12 más negociadas el {d_str} (dinero medio de 60 sesiones, M$): ' + ' '.join(f'{syms[j]}({dvm[r, j] / 1e6:,.0f})' for j in o_), flush=True)
    valid = np.isfinite(dvm).sum(1)
    ok_days = valid >= KS[0]
    print(f'  miembros por día del universo de {KS[0]}: mediana {int(np.median(mem[KS[0]][ok_days].sum(1)))} · días con < {KS[0]} valores con dato: {int((~ok_days).sum())} de {len(days)}', flush=True)
    P = Panel()
    t0 = __import__('time').time()
    for q, j in enumerate(work):
        s = syms[j]
        F = feats.build(data2[s])
        d = (F['t'] // 86400).astype(np.int64)
        ix = np.searchsorted(days, d)
        P.add(s, q, F, {K: mem[K][ix, j] for K in KS})
        if (q + 1) % 100 == 0:
            print(f'    {q + 1}/{len(work)} valores · {__import__("time").time() - t0:.0f}s', flush=True)
    P.finish()
    key = ('long', PRIMARY)
    stats_out: Dict[str, Dict] = {nm: {'dir': d, 'fam': fam, 'desc': desc} for nm, (d, fam, desc) in PATTERNS.items()}
    thr_out: Dict[str, Dict] = {}
    for K in KS:
        n_mem = int(P.memv[K].sum())
        print(f'\n================ UNIVERSO: {K} ACCIONES MÁS NEGOCIADAS EN CADA FECHA · {n_mem:,} barras ================', flush=True)
        for dname, dirn in (('largo', 'long'), ('corto', 'short')):
            b = P.base[(K, dirn, PRIMARY)]
            print(f'  entrar al azar a {PRIMARY} sesiones ({dname}, neto de {COST} % de coste): ' + ' '.join(f'{P.y0 + i}:{v:+.2f}' for i, v in enumerate(b) if np.isfinite(v)), flush=True)
        null = P.null_max_t(K, key, 150)
        thr95, thr99 = float(np.quantile(null, 0.95)), float(np.quantile(null, 0.99))
        thr_out[f'K{K}'] = {'p95': round(thr95, 2), 'p99': round(thr99, 2), 'bars': n_mem}
        for nm_, (d_, fam_, desc_) in PATTERNS.items():
            k_ = (('long' if d_ > 0 else 'short'), PRIMARY)
            st_ = P.stat(P.ev[nm_], K, k_)
            if st_.get('n', 0) >= 1:
                stats_out[nm_][f'K{K}'] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st_.items()}
        print(f'  bajo el azar, el MEJOR |t| de {len(P.ev)} patrones es ≥ {thr95:.1f} el 5 % de las veces y ≥ {thr99:.1f} el 1 % (150 desplazamientos temporales de las señales de cada valor)', flush=True)
        rows = []
        for nm, (d, fam, desc) in PATTERNS.items():
            k = (('long' if d > 0 else 'short'), PRIMARY)
            st = P.stat(P.ev[nm], K, k)
            if st.get('n', 0) >= 20:
                rows.append((nm, d, fam, st))
        rows.sort(key=lambda r: -abs(r[3]['t']))
        print('     patrón'.ljust(24) + 'familia'.ljust(10) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'mediana'.rjust(9) + 'exceso'.rjust(8) + 't'.rjust(6) + 'años+'.rjust(7))
        for nm, d, fam, st in rows[:30]:
            star = '★' if abs(st['t']) >= thr95 else ' '
            print('     ' + (('▲ ' if d > 0 else '▼ ') + nm + star).ljust(23) + fam.ljust(10) + f"{st['n']:6d}{st['wr']:8.1f}%{st['mean']:+7.2f}%{st['med']:+8.2f}%{st['ex']:+7.2f}%{st['t']:+6.1f}{st['pos']:4d}/{st['tot']:<2d}", flush=True)
        sig = [r[0] for r in rows if abs(r[3]['t']) >= thr95]
        print(f'     superan el umbral del mejor de {len(rows)}: {len(sig)} ({", ".join(sig) if sig else "ninguno"}) · por azar se esperarían ≈ {0.05 * len(rows):.1f} si se miraran sueltos')
        # resumen por familia y dirección
        print('     resumen (exceso medio sobre entrar al azar, a 10 sesiones; % de patrones con exceso positivo):')
        for fam in ('velas', 'rupturas', 'figuras'):
            for d, dn in ((1, 'alcistas (largo)'), (-1, 'bajistas (corto)')):
                sel = [r for r in rows if r[2] == fam and r[1] == d]
                if sel:
                    print(f'       {fam:9s} {dn:17s} {len(sel):2d} patrones · exceso medio {np.mean([r[3]["ex"] for r in sel]):+.3f}% · positivos {np.mean([r[3]["ex"] > 0 for r in sel]) * 100:.0f}%')
        # los 8 mejores y 8 peores con todos los horizontes y la salida con objetivo y stop
        print('     horizontes y salida con objetivo/stop 2×ATR para los 8 de mayor |t| (exceso % / t):')
        print('       ' + 'patrón'.ljust(20) + '5 ses.'.rjust(16) + '10 ses.'.rjust(16) + '20 ses.'.rjust(16) + 'obj/stop 2×ATR'.rjust(18))
        for nm, d, fam, st in rows[:8]:
            side = 'long' if d > 0 else 'short'
            cells = []
            for hz in (5, 10, 20, 0):
                s2 = P.stat(P.ev[nm], K, (side, hz))
                cells.append(f"{s2['ex']:+6.2f}% /{s2['t']:+5.1f}" if 'ex' in s2 else '       n/d')
            print('       ' + nm.ljust(20) + ''.join(c.rjust(16 if i < 3 else 18) for i, c in enumerate(cells)), flush=True)
    import json as _json
    import os as _os
    _os.makedirs(out_dir, exist_ok=True)
    _json.dump({'primary': PRIMARY, 'cost': COST, 'thr': thr_out, 'ks': list(KS), 'patterns': stats_out}, open(_os.path.join(out_dir, 'pattern_stats.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════
#  ¿Mejoran los patrones el sistema de rebotes? (los 65 patrones como señales en las barras de cada candidato)
# ═════════════════════════════════════════════════════════════════════════
def overlay(data: Dict, EV: Dict, rows: np.ndarray, look: int = 3) -> Tuple[np.ndarray, List[str], List[str]]:
    """Matriz [candidatos × patrones] con 1 si el patrón se completó en alguna de las últimas `look` barras (incluida la del candidato)."""
    sym, bi = EV['sym'][rows], EV['bi'][rows].astype(np.int64)
    names = list(PATTERNS)
    M = np.zeros((len(rows), len(names)), dtype=np.float32)
    order = np.argsort(sym, kind='stable')
    cuts = np.flatnonzero(np.diff(sym[order])) + 1
    for q, grp in enumerate(np.split(order, cuts)):
        if not len(grp):
            continue
        F = feats.build(data[EV['sym_names'][sym[grp[0]]]])
        S = signals(F)
        i = bi[grp]
        for k, nm in enumerate(names):
            sg = S[nm]
            if not sg.any():
                continue
            hit = np.zeros(len(sg), bool)
            for L in range(look):
                hit[L:] |= sg[:len(sg) - L]
            M[grp, k] = hit[i]
        if (q + 1) % 300 == 0:
            print(f'    {q + 1} valores', flush=True)
    return M, names, [PATTERNS[n][1] for n in names]


def run2(uni: Dict, data: Dict, args, out_dir: str):
    import osc
    import pit
    import research as RS
    import setups as SU
    import simulate as SM
    import stops as ST
    uni2, data2, row_ok, info = pit.prepare(uni, data, args)
    EV = RS.build_events(data2, uni2, SU.SETUPS, SM.default_variants())
    rows, FL, names = RS.union_events(EV, +1)
    k = row_ok(EV, rows)
    rows, FL = rows[k], FL[k]
    last_day = int(EV['day'].max())
    day, gid = EV['day'][rows], EV['grp'][rows]
    year = RS.year_of(day)
    print(f'\n######## CHARTISMO COMO AYUDA AL SISTEMA DE REBOTES · {len(rows):,} candidatos', flush=True)
    exp: Dict = {}
    osc._quiet(RS.build_plan, EV, rows, FL, names, ST.plan_of(4.0), last_day, spy=data2['SPY'], export=exp)
    allowed = exp['allowed']
    M, cols, fam = overlay(data2, EV, rows)
    print(f'  patrones completados en las últimas 3 barras de cada candidato: {np.mean(M.sum(1) > 0) * 100:.1f}% de los candidatos tienen alguno', flush=True)
    vi = EV['variants'].index(ST.vname(4.0))
    pnl = EV['pnl'][rows, vi]
    atrp = EV['X'][rows][:, RS.FEATURES.index('atrp')]
    elig = np.isfinite(pnl) & (day <= last_day) & np.isfinite(atrp) & (atrp >= RS.MIN_ATR_PCT) & (atrp < RS.MAX_ATR_PCT) & np.isin(gid, allowed)
    print(f'  fondo: {int(elig.sum()):,} candidatos elegibles · media {pnl[elig].mean():+.2f}% · acierto {np.mean(pnl[elig] > 0) * 100:.1f}%', flush=True)
    print('\n  A.', end='')
    sc = osc.screen(M, cols, elig, day, pnl, year, noun='patrones', fam=fam, top=25)
    # grupos con y sin patrón, frente al resto
    sym = EV['sym'][rows]
    print('\n  B. CANDIDATOS CON EL PATRÓN FRENTE AL RESTO (solo patrones con ≥ 150 candidatos elegibles)')
    out = []
    for j, nm in enumerate(cols):
        m = elig & (M[:, j] > 0)
        if m.sum() >= 150:
            a = RS.stats(pnl[m], day[m], sym[m])
            b = RS.stats(pnl[elig & ~(M[:, j] > 0)], day[elig & ~(M[:, j] > 0)], sym[elig & ~(M[:, j] > 0)])
            dif = a['mean'] - b['mean']
            out.append((nm, a['n'], a['wr'], b['wr'], a['mean'], b['mean'], dif, dif / ((a['se'] ** 2 + b['se'] ** 2) ** 0.5 + 1e-12)))
    out.sort(key=lambda r: -abs(r[7]))
    for nm, n_, w1, w2, m1, m2, dif, t in out[:15]:
        print(f'     {nm:18s} n={n_:5d} · acierto {w1:5.1f}% vs {w2:5.1f}% · media {m1:+6.2f}% vs {m2:+6.2f}% · dif {dif:+5.2f} (t={t:+4.1f})', flush=True)
    famc = {f: [nm for nm, ff in zip(cols, fam) if ff == f] for f in dict.fromkeys(fam)}
    bull = [nm for nm in cols if PATTERNS[nm][0] > 0]
    arms = [('modelo actual (28 variables)', [])] + [(f'+ {f} ({len(l)})', l) for f, l in famc.items()] + [(f'+ alcistas ({len(bull)})', bull), (f'+ los {len(cols)} patrones', list(cols)),
                                                                                                         ('+ los que superan el umbral del mejor de N', sc['sig'])]
    osc.model_arms(EV, rows, FL, names, M, cols, arms, last_day, data2['SPY'], 'C. SISTEMA ENTERO REENTRENADO CON LOS PATRONES COMO VARIABLES (AUC fuera de muestra)')
    sys.stdout.flush()
