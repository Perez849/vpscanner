#!/usr/bin/env python3
"""
chartlab.py — LABORATORIO DE GRÁFICOS: dibuja sobre cada valor los patrones que detecta `charts.py` (figuras, rupturas, velas), con sus soportes y resistencias,
y con la entrada, el stop y el objetivo medidos con la regla clásica de cada figura. Para los patrones pasados simula qué ocurrió después (¿objetivo o stop?).

Salida (JSON por valor, lo lee `charts.html`):
  bars     : t (día), o, h, l, c, v de las últimas NB barras  + sma50/sma200
  sr       : soportes y resistencias (zonas con ≥ 2 toques)
  patterns : lista de patrones; cada uno con puntos (pts), líneas, polígonos, zonas, entrada/stop/objetivo (rr), nota y resultado simulado (out)
No interviene en producción: es una herramienta de visualización y NO una recomendación; el estudio `charts` muestra que estos patrones no baten al azar.
"""
from __future__ import annotations
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import charts as CH
import feats
from shapes import at

NB = 340            # barras exportadas
WARM = 260          # barras previas que solo sirven para las medias
HOLD = 40           # sesiones máximas para el resultado simulado
ACTIVE = 12         # un patrón es «activo» si se completó en las últimas 12 sesiones y aún no resolvió

# Patrones solo de visualización (el estudio no los incluye)
EXTRA: Dict[str, Tuple[int, str, str]] = {
    'doji_bull': (1, 'velas', 'doji tras una caída (indecisión en el suelo)'),
    'doji_bear': (-1, 'velas', 'doji tras una subida (indecisión en el techo)'),
    'spin_bull': (1, 'velas', 'peonza tras una caída'),
    'spin_bear': (-1, 'velas', 'peonza tras una subida'),
    'doji_star_bull': (1, 'velas', 'estrella doji alcista (estrella de la mañana con doji)'),
    'doji_star_bear': (-1, 'velas', 'estrella doji bajista (estrella de la tarde con doji)'),
    'three_inside_up': (1, 'velas', 'tres velas interiores alcistas'),
    'three_inside_dn': (-1, 'velas', 'tres velas interiores bajistas'),
}
# Los que se calculan pero no se dibujan por ser ruido (muchísimas señales sin forma): se dejan fuera de la salida
SKIP = {'nr7_up', 'nr7_dn', 'inside_up', 'inside_dn', 'donch20_up', 'donch20_dn'}      # las variantes «_hq» (calidad alta) solo existen para el estudio: se saltan abajo

LABEL = {
    'hammer': 'Martillo', 'inv_hammer': 'Martillo invertido', 'shooting_star': 'Estrella fugaz', 'hanging_man': 'Hombre colgado', 'bull_engulf': 'Envolvente alcista',
    'bear_engulf': 'Envolvente bajista', 'piercing': 'Línea penetrante', 'dark_cloud': 'Nube oscura', 'morning_star': 'Estrella de la mañana', 'evening_star': 'Estrella de la tarde',
    'soldiers': 'Tres soldados', 'crows': 'Tres cuervos', 'harami_bull': 'Harami alcista', 'harami_bear': 'Harami bajista', 'tweezer_bottom': 'Pinzas de suelo',
    'tweezer_top': 'Pinzas de techo', 'dragonfly': 'Doji libélula', 'gravestone': 'Doji lápida', 'marubozu_bull': 'Marubozu alcista', 'marubozu_bear': 'Marubozu bajista',
    'keyrev_bull': 'Reversión clave ↑', 'keyrev_bear': 'Reversión clave ↓', 'doji_bull': 'Doji ↑ (tras caída)', 'doji_bear': 'Doji ↓ (tras subida)', 'spin_bull': 'Peonza ↑', 'spin_bear': 'Peonza ↓', 'doji_star_bull': 'Estrella doji ↑', 'doji_star_bear': 'Estrella doji ↓',
    'three_inside_up': 'Tres interiores ↑', 'three_inside_dn': 'Tres interiores ↓',
    'donch55_up': 'Ruptura máx. 55', 'donch55_dn': 'Ruptura mín. 55', 'donch100_up': 'Ruptura máx. 100', 'donch100_dn': 'Ruptura mín. 100', 'donch252_up': 'Ruptura máx. anual',
    'donch252_dn': 'Ruptura mín. anual', 'bb_squeeze_up': 'Squeeze Bollinger ↑', 'bb_squeeze_dn': 'Squeeze Bollinger ↓', 'atr_squeeze_up': 'Compresión ATR ↑', 'atr_squeeze_dn': 'Compresión ATR ↓',
    'box_up': 'Caja (Darvas) ↑', 'box_dn': 'Caja (Darvas) ↓', 'base_near_high': 'Base junto a máximos',
    'double_bottom': 'Doble suelo', 'double_top': 'Doble techo', 'hs_top': 'Hombro-cabeza-hombro', 'hs_inverse': 'H-C-H invertido', 'asc_triangle': 'Triángulo ascendente',
    'desc_triangle': 'Triángulo descendente', 'sym_tri_up': 'Triángulo simétrico ↑', 'sym_tri_dn': 'Triángulo simétrico ↓', 'rising_wedge': 'Cuña ascendente', 'falling_wedge': 'Cuña descendente',
    'trendline_up': 'Ruptura de directriz bajista', 'trendline_dn': 'Ruptura de directriz alcista', 'support_bounce': 'Rebote en soporte', 'resistance_reject': 'Rechazo en resistencia',
    'triple_bottom': 'Triple suelo', 'triple_top': 'Triple techo', 'rect_up': 'Rectángulo ↑', 'rect_dn': 'Rectángulo ↓',
    'vcp': 'VCP (contracción)', 'cup_handle': 'Taza con asa', 'diamond_bull': 'Diamante de suelo', 'diamond_bear': 'Diamante de techo', 'bull_flag': 'Bandera alcista',
    'bear_flag': 'Bandera bajista', 'bull_pennant': 'Banderín alcista', 'bear_pennant': 'Banderín bajista', 'island_bull': 'Isla de suelo', 'island_bear': 'Isla de techo',
}
CANDLE_BARS = {'hammer': 1, 'inv_hammer': 1, 'shooting_star': 1, 'hanging_man': 1, 'dragonfly': 1, 'gravestone': 1, 'marubozu_bull': 1, 'marubozu_bear': 1, 'doji_bull': 1, 'doji_bear': 1, 'spin_bull': 1, 'spin_bear': 1,
               'bull_engulf': 2, 'bear_engulf': 2, 'piercing': 2, 'dark_cloud': 2, 'harami_bull': 2, 'harami_bear': 2, 'tweezer_bottom': 2, 'tweezer_top': 2, 'keyrev_bull': 2, 'keyrev_bear': 2,
               'morning_star': 3, 'evening_star': 3, 'soldiers': 3, 'crows': 3, 'doji_star_bull': 3, 'doji_star_bear': 3, 'three_inside_up': 3, 'three_inside_dn': 3}


def _r(x: float) -> float:
    x = float(x)
    return round(x, 4 if abs(x) < 10 else 2)


def pat_info(name: str) -> Tuple[int, str, str]:
    return CH.PATTERNS[name] if name in CH.PATTERNS else EXTRA[name]


def extra_candles(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    o, h, l, c, atr, atrp = F['o'], F['h'], F['l'], F['c'], F['atr'], F['atrp']
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        rng, body = h - l, np.abs(c - o)
        up_sh, lo_sh = h - np.maximum(o, c), np.minimum(o, c) - l
        big = rng >= 0.8 * atr
        dn_state = (c / CH._sh(c, 5) - 1.0 < -1.5 * atrp / 100.0) & (c < F['sma20'])
        up_state = (c / CH._sh(c, 5) - 1.0 > 1.5 * atrp / 100.0) & (c > F['sma20'])
        s = lambda a, k: CH._sh(a.astype(np.float64), k) > 0.5
        dj = big & (body <= 0.1 * rng) & (up_sh >= 0.2 * rng) & (lo_sh >= 0.2 * rng)          # doji con ambas mechas (los libélula/lápida ya están en las velas)
        sp = big & (body <= 0.25 * rng) & (body > 0.1 * rng) & (up_sh >= 0.25 * rng) & (lo_sh >= 0.25 * rng)
        out['doji_bull'], out['doji_bear'] = s(dn_state, 1) & dj, s(up_state, 1) & dj
        out['spin_bull'], out['spin_bear'] = s(dn_state, 1) & sp, s(up_state, 1) & sp
        o1, c1, o2, c2 = CH._sh(o, 1), CH._sh(c, 1), CH._sh(o, 2), CH._sh(c, 2)
        b1, b2 = CH._sh(body, 1), CH._sh(body, 2)
        r1 = CH._sh(rng, 1)
        dj1 = (b1 <= 0.1 * r1) & (r1 >= 0.5 * CH._sh(atr, 1))
        mid2 = (o2 + c2) / 2.0
        out['doji_star_bull'] = s(dn_state, 3) & (c2 < o2) & (b2 >= CH._sh(atr, 2)) & dj1 & (np.maximum(o1, c1) < c2) & (c > o) & (c > mid2)
        out['doji_star_bear'] = s(up_state, 3) & (c2 > o2) & (b2 >= CH._sh(atr, 2)) & dj1 & (np.minimum(o1, c1) > c2) & (c < o) & (c < mid2)
        # tres interiores: harami + cierre de confirmación más allá de la apertura de la primera vela
        out['three_inside_up'] = s(dn_state, 3) & (c2 < o2) & (b2 >= CH._sh(atr, 2)) & (c1 > o1) & (o1 > c2) & (c1 < o2) & (c > o2) & (c > o)
        out['three_inside_dn'] = s(up_state, 3) & (c2 > o2) & (b2 >= CH._sh(atr, 2)) & (c1 < o1) & (o1 < c2) & (c1 > o2) & (c < o2) & (c < o)
    return {k: np.nan_to_num(v.astype(np.float64)).astype(bool) for k, v in out.items()}


# ── soportes y resistencias ───────────────────────────────────────────────
def support_resistance(F: Dict[str, np.ndarray], s0: int, k: int = 5, max_levels: int = 10) -> List[Dict]:
    """Zonas con ≥ 2 toques: agrupa los pivotes de las últimas barras (tolerancia 0,6 ATR) y puntúa por nº de toques y cercanía en el tiempo."""
    h, l, c, atr = F['h'], F['l'], F['c'], F['atr']
    n = len(c)
    ph, pl = CH.pivots(h, l, k)
    pts = [(j, float(h[j]), 'R') for j in np.flatnonzero(ph) if j >= s0 - 30 and j + k < n] + [(j, float(l[j]), 'S') for j in np.flatnonzero(pl) if j >= s0 - 30 and j + k < n]
    pts.sort(key=lambda x: x[1])
    a = float(atr[-1]) if np.isfinite(atr[-1]) else float(np.nanmedian(atr))
    tol = 0.6 * a
    clusters: List[List[Tuple[int, float, str]]] = []
    for p in pts:
        if clusters and p[1] - np.mean([q[1] for q in clusters[-1]]) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    last = float(c[-1])
    out = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        price = float(np.mean([q[1] for q in cl]))
        if abs(price / last - 1.0) > 0.20:
            continue
        idxs = [q[0] for q in cl]
        rec = np.mean([math.exp(-(n - 1 - j) / 120.0) for j in idxs])
        out.append({'p': _r(price), 'lo': _r(price - 0.25 * a), 'hi': _r(price + 0.25 * a), 'touches': len(cl), 'i0': int(min(idxs)) - s0, 'i1': int(max(idxs)) - s0,
                    'kind': 'S' if price < last else 'R', 'score': float(len(cl) * (0.5 + rec))})
    out.sort(key=lambda x: -x['score'])
    out = out[:max_levels]
    for x in out:
        x['score'] = round(x['score'], 2)
    return out


# ── resultado simulado ────────────────────────────────────────────────────
def outcome(F: Dict[str, np.ndarray], i: int, d: int, stop: float, target: float, maxb: int = HOLD) -> Dict:
    o, h, l, c = F['o'], F['h'], F['l'], F['c']
    n = len(c)
    e = i + 1
    if e >= n:
        return {'res': 'open', 'bars': 0, 'ret': None}
    entry = float(o[e])
    if (d > 0 and (entry <= stop or entry >= target)) or (d < 0 and (entry >= stop or entry <= target)):
        return {'res': 'na', 'bars': 0, 'ret': None}
    for k in range(e, min(n, e + maxb)):
        if d > 0:
            if l[k] <= stop:
                px = min(stop, float(o[k])) if k == e or o[k] <= stop else stop
                return {'res': 'stop', 'bars': k - e + 1, 'ret': _r(d * (px / entry - 1) * 100), 'xi': k, 'xp': _r(px)}
            if h[k] >= target:
                px = max(target, float(o[k])) if o[k] >= target else target
                return {'res': 'target', 'bars': k - e + 1, 'ret': _r(d * (px / entry - 1) * 100), 'xi': k, 'xp': _r(px)}
        else:
            if h[k] >= stop:
                px = max(stop, float(o[k])) if o[k] >= stop else stop
                return {'res': 'stop', 'bars': k - e + 1, 'ret': _r(d * (px / entry - 1) * 100), 'xi': k, 'xp': _r(px)}
            if l[k] <= target:
                px = min(target, float(o[k])) if o[k] <= target else target
                return {'res': 'target', 'bars': k - e + 1, 'ret': _r(d * (px / entry - 1) * 100), 'xi': k, 'xp': _r(px)}
    k = min(n, e + maxb) - 1
    res = 'open' if k >= n - 1 and e + maxb > n else 'timeout'
    return {'res': res, 'bars': k - e + 1, 'ret': _r(d * (float(c[k]) / entry - 1) * 100), 'xi': k, 'xp': _r(c[k])}


# ── niveles de entrada, stop y objetivo (reglas clásicas) ─────────────────
def _generic(entry: float, a: float, d: int, risk: float, rmult: float = 2.0) -> Tuple[float, float]:
    risk = max(risk, 0.6 * a)
    return entry - d * risk, entry + d * rmult * risk


def levels(name: str, g: Dict, F: Dict[str, np.ndarray], i: int, d: int, entry: Optional[float] = None) -> Tuple[float, float, str]:
    """(stop, objetivo, nota) con la regla clásica de cada patrón. `entry` permite calcularlo para una ruptura futura (figuras en formación)."""
    o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
    a = float(atr[i])
    entry = float(c[i]) if entry is None else float(entry)
    if name in ('double_bottom', 'triple_bottom'):
        low = min(p[1] for p in g['pts'])
        return low - 0.25 * a, g['neck'] + g['hgt'], 'Objetivo = altura de la figura (cuello − mínimos) proyectada sobre el cuello. Stop 0,25 ATR bajo el mínimo más bajo.'
    if name in ('double_top', 'triple_top'):
        hi = max(p[1] for p in g['pts'])
        return hi + 0.25 * a, g['neck'] - g['hgt'], 'Objetivo = altura de la figura (máximos − cuello) proyectada bajo el cuello. Stop 0,25 ATR sobre el máximo más alto.'
    if name == 'hs_top':
        return g['rs'][1] + 0.25 * a, g['neck'] - g['hgt'], 'Objetivo = distancia cabeza–línea de cuello proyectada bajo la ruptura. Stop sobre el hombro derecho.'
    if name == 'hs_inverse':
        return g['rs'][1] - 0.25 * a, g['neck'] + g['hgt'], 'Objetivo = distancia cabeza–línea de cuello proyectada sobre la ruptura. Stop bajo el hombro derecho.'
    if name in ('asc_triangle', 'desc_triangle', 'sym_tri_up', 'sym_tri_dn', 'rising_wedge', 'falling_wedge', 'rect_up', 'rect_dn'):
        U, D, x0 = g['U'], g['D'], g['x0']
        w0 = max(at(U, x0) - at(D, x0), 0.5 * a)                      # anchura de la figura en su inicio: la medida clásica
        lastlo = min(at(D, i), g['tl'][-1][1]) if g.get('tl') else at(D, i)
        lasthi = max(at(U, i), g['tu'][-1][1]) if g.get('tu') else at(U, i)
        if name in ('asc_triangle', 'sym_tri_up', 'rect_up'):
            txt = {'asc_triangle': 'Objetivo = altura de la base (en el inicio) proyectada sobre la resistencia.', 'sym_tri_up': 'Objetivo = anchura del triángulo en su inicio proyectada desde la ruptura.',
                   'rect_up': 'Objetivo = altura del rectángulo proyectada desde la ruptura.'}[name]
            return lastlo - 0.25 * a, entry + w0, txt + ' Stop bajo el último mínimo / la base.'
        if name in ('desc_triangle', 'sym_tri_dn', 'rect_dn'):
            txt = {'desc_triangle': 'Objetivo = altura de la base (en el inicio) proyectada bajo el soporte.', 'sym_tri_dn': 'Objetivo = anchura del triángulo en su inicio proyectada desde la ruptura.',
                   'rect_dn': 'Objetivo = altura del rectángulo proyectada desde la ruptura.'}[name]
            return lasthi + 0.25 * a, entry - w0, txt + ' Stop sobre el último máximo / el techo.'
        if name == 'falling_wedge':
            start = at(U, x0)
            return lastlo - 0.25 * a, (start if start > entry + 0.5 * a else entry + w0), 'Objetivo = inicio de la cuña (la recta superior en su arranque). Stop bajo el último mínimo de la cuña.'
        start = at(D, x0)
        return lasthi + 0.25 * a, (start if start < entry - 0.5 * a else entry - w0), 'Objetivo = inicio de la cuña (la recta inferior en su arranque). Stop sobre el último máximo de la cuña.'
    if name == 'cup_handle':
        rim = max(g['left'][1], g['right'][1])
        return g['handle'][1] - 0.25 * a, rim + (rim - g['bottom'][1]), 'Objetivo = profundidad de la taza proyectada sobre el borde. Stop bajo el mínimo del asa.'
    if name == 'vcp':
        stop = g['lows'][-1][1] - 0.25 * a
        return stop, entry + 2.0 * max(entry - stop, 0.6 * a), 'Objetivo genérico de 2R. Stop bajo el mínimo de la última contracción.'
    if name in ('bull_flag', 'bull_pennant'):
        return g['fl'] - 0.25 * a, g['fh'] + abs(g['pole']), 'Objetivo = altura del asta proyectada desde la ruptura de la bandera. Stop bajo el mínimo de la bandera.'
    if name in ('bear_flag', 'bear_pennant'):
        return g['fh'] + 0.25 * a, g['fl'] - abs(g['pole']), 'Objetivo = altura del asta proyectada desde la ruptura de la bandera. Stop sobre el máximo de la bandera.'
    if name in ('diamond_bull', 'diamond_bear'):
        if d > 0:
            return g['bot'] - 0.25 * a, entry + g['width'], 'Objetivo = anchura máxima del diamante proyectada desde la ruptura. Stop bajo el rango final.'
        return g['top'] + 0.25 * a, entry - g['width'], 'Objetivo = anchura máxima del diamante proyectada desde la ruptura. Stop sobre el rango final.'
    if name == 'island_bull':
        return g['lo'] - 0.25 * a, max(g['gap_from'], entry + 1.5 * a), 'Objetivo = cierre del hueco (nivel previo a la isla). Stop bajo la isla.'
    if name == 'island_bear':
        return g['hi'] + 0.25 * a, min(g['gap_from'], entry - 1.5 * a), 'Objetivo = cierre del hueco (nivel previo a la isla). Stop sobre la isla.'
    if name == 'support_bounce':
        stop, tgt = _generic(entry, a, 1, entry - min(float(l[i]), g['level']) + 0.5 * a)
        return stop, tgt, 'Objetivo genérico de 2R. Stop bajo el soporte y la mecha de rebote.'
    if name == 'resistance_reject':
        stop, tgt = _generic(entry, a, -1, max(float(h[i]), g['level']) - entry + 0.5 * a)
        return stop, tgt, 'Objetivo genérico de 2R. Stop sobre la resistencia y la mecha de rechazo.'
    if name in ('trendline_up', 'trendline_dn'):
        stop, tgt = _generic(entry, a, d, 1.5 * a)
        return stop, tgt, 'Objetivo genérico de 2R con stop de 1,5 ATR (la ruptura de una directriz no tiene medida clásica).'
    if name in ('box_up', 'box_dn'):
        bh, bl = g['hi'], g['lo']
        height = max(bh - bl, 0.5 * a)
        return ((bl - 0.25 * a, bh + height) if d > 0 else (bh + 0.25 * a, bl - height)) + ('Objetivo = altura de la caja proyectada desde la ruptura. Stop al otro lado de la caja.',)
    if name == 'base_near_high':
        bh, bl = g['hi'], g['lo']
        return bl - 0.25 * a, bh + max(bh - bl, 0.5 * a), 'Objetivo = altura de la base proyectada desde la ruptura. Stop bajo la base.'
    if name.startswith('donch'):
        lvl = g['level']
        stop, tgt = _generic(entry, a, d, abs(entry - lvl) + 1.5 * a)
        return stop, tgt, 'Objetivo genérico de 2R. Stop 1,5 ATR más allá del nivel roto.'
    if name.startswith('bb_squeeze') or name.startswith('atr_squeeze'):
        mid = float(F['sma20'][i])
        stop, tgt = _generic(entry, a, d, abs(entry - mid))
        return stop, tgt, 'Objetivo genérico de 2R. Stop en la media de 20 sesiones.'
    # velas
    m = CANDLE_BARS.get(name, 1)
    lo_, hi_ = float(l[i - m + 1:i + 1].min()), float(h[i - m + 1:i + 1].max())
    if d >= 0:
        stop, tgt = _generic(entry, a, 1, entry - lo_ + 0.1 * a)
    else:
        stop, tgt = _generic(entry, a, -1, hi_ - entry + 0.1 * a)
    return stop, tgt, 'Objetivo genérico de 2R. Stop más allá del extremo del patrón de velas.'


# ── geometría por patrón ──────────────────────────────────────────────────
def geometry(name: str, g: Dict, F: Dict[str, np.ndarray], i: int) -> Dict:
    h, l, c = F['h'], F['l'], F['c']
    lines: List[Dict] = []
    pts: List[List] = []
    polys: List[Dict] = []
    zones: List[Dict] = []
    ann: List[Dict] = []

    def ln(a, b, k='trend', dash=False, n=None):
        d_ = {'a': [int(a[0]), _r(a[1])], 'b': [int(b[0]), _r(b[1])], 'k': k, 'dash': dash}
        if n:
            d_['n'] = int(n)
        lines.append(d_)

    def pt(p, tag=''):
        pts.append([int(p[0]), _r(p[1]), tag])

    def ext(p1, p2, x):
        return p2[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (x - p2[0])

    if name in ('double_bottom', 'double_top', 'triple_bottom', 'triple_top'):
        P, M = g['pts'], g['mids']
        seq = []
        for k, p in enumerate(P):
            seq.append(p)
            if k < len(M):
                seq.append(M[k])
        polys.append({'pts': [[int(x), _r(y)] for x, y in seq], 'k': 'zig'})
        for k, p in enumerate(P):
            pt(p, str(k + 1))
        for m in M:
            pt(m, '')
        ln((P[0][0], g['neck']), (i, g['neck']), 'neck', True)
        ext_ = min(P, key=lambda q: q[1]) if name.endswith('bottom') else max(P, key=lambda q: q[1])
        ann.append({'k': 'height', 'x': int(P[-1][0]), 'y0': _r(g['neck']), 'y1': _r(ext_[1]), 't': 'altura'})
    elif name in ('hs_top', 'hs_inverse'):
        pts_ = [g['ls'], g['t1'], g['hd'], g['t2'], g['rs']]
        polys.append({'pts': [[int(p[0]), _r(p[1])] for p in pts_], 'k': 'zig'})
        for p, t in zip(pts_, ['Hombro', '', 'Cabeza', '', 'Hombro']):
            pt(p, t)
        ln(g['t1'], (i, ext(g['t1'], g['t2'], i)), 'neck', True)
        ann.append({'k': 'height', 'x': int(g['hd'][0]), 'y0': _r(ext(g['t1'], g['t2'], g['hd'][0])), 'y1': _r(g['hd'][1]), 't': 'altura'})
    elif name in ('asc_triangle', 'desc_triangle', 'sym_tri_up', 'sym_tri_dn', 'rising_wedge', 'falling_wedge', 'rect_up', 'rect_dn') or name.startswith('forming_'):
        U, D, x0 = g['U'], g['D'], g['x0']
        ln((x0, at(U, x0)), (i, at(U, i)), 'upper', False, len(g['tu']))
        ln((x0, at(D, x0)), (i, at(D, i)), 'lower', False, len(g['tl']))
        polys.append({'pts': [[int(x0), _r(at(U, x0))], [int(i), _r(at(U, i))], [int(i), _r(at(D, i))], [int(x0), _r(at(D, x0))]], 'k': 'fill'})
        for p in g['tu']:
            pt(p, '')
        for p in g['tl']:
            pt(p, '')
        ap = g.get('apex')
        if ap is not None and i < ap <= i + 45:
            ln((i, at(U, i)), (ap, at(U, ap)), 'ext', True)
            ln((i, at(D, i)), (ap, at(D, ap)), 'ext', True)
            pt((ap, at(U, ap)), 'ápice')
        ann.append({'k': 'height', 'x': int(x0), 'y0': _r(at(U, x0)), 'y1': _r(at(D, x0)), 't': 'base'})
    elif name in ('trendline_up', 'trendline_dn'):
        L, tch = g['L'], g['tch']
        ln((tch[0][0], at(L, tch[0][0])), (i, at(L, i)), 'trend', False, len(tch))
        for k, p in enumerate(tch):
            pt(p, str(k + 1))
    elif name in ('support_bounce', 'resistance_reject'):
        t0 = g['touches'][0][0]
        ln((t0, g['level']), (i, g['level']), 'level')
        for p in g['touches']:
            pt(p, '')
    elif name == 'vcp':
        zz = []
        for k, hi_ in enumerate(g['highs']):
            zz.append([hi_[0], _r(hi_[1])])
            zz.append([g['lows'][k][0], _r(g['lows'][k][1])])
            pt(hi_, f'{k + 1}')
        polys.append({'pts': zz + [[i, _r(c[i])]], 'k': 'zig'})
        ln((g['highs'][0][0], g['highs'][-1][1]), (i, g['highs'][-1][1]), 'neck', True)
    elif name == 'cup_handle':
        cf = g['fit']
        xs = np.arange(g['W'])
        ys = np.polyval(cf, xs)
        step = max(g['W'] // 28, 1)
        polys.append({'pts': [[int(g['w0'] + x), _r(y)] for x, y in zip(xs[::step], ys[::step])] + [[int(g['w0'] + xs[-1]), _r(ys[-1])]], 'k': 'curve'})
        rim = max(g['left'][1], g['right'][1])
        ln((g['left'][0], rim), (i, rim), 'neck', True)
        zones.append({'i0': g['right'][0], 'i1': i, 'lo': _r(g['handle'][1]), 'hi': _r(g['right'][1]), 'k': 'box'})
        pt(g['left'], 'Borde'); pt(g['bottom'], 'Fondo'); pt(g['right'], 'Borde'); pt(g['handle'], 'Asa')
        ann.append({'k': 'height', 'x': int(g['bottom'][0]), 'y0': _r(rim), 'y1': _r(g['bottom'][1]), 't': 'profundidad'})
    elif name in ('diamond_bull', 'diamond_bear'):
        hi_, lo_ = g['hi'], g['lo']
        poly = [[p[0], _r(p[1])] for p in hi_] + [[p[0], _r(p[1])] for p in lo_[::-1]]
        polys.append({'pts': poly, 'k': 'fill'})
        polys.append({'pts': [[p[0], _r(p[1])] for p in hi_], 'k': 'zig'})
        polys.append({'pts': [[p[0], _r(p[1])] for p in lo_], 'k': 'zig'})
    elif name in ('bull_flag', 'bear_flag', 'bull_pennant', 'bear_pennant'):
        ps, pe = g['ps'], g['pe']
        ln(ps, pe, 'pole')
        z0 = g['z0']
        seg_h, seg_l = h[z0:i], l[z0:i]
        xs = np.arange(z0, i)
        if len(xs) >= 3:
            uh = np.polyfit(xs, seg_h, 1)
            ul = np.polyfit(xs, seg_l, 1)
            if not g.get('pen'):                                       # bandera: canal paralelo que contiene toda la consolidación
                sl_ = float(np.polyfit(xs, c[z0:i], 1)[0])
                up_b = float((seg_h - sl_ * xs).max())
                lo_b = float((seg_l - sl_ * xs).min())
                uh, ul = np.array([sl_, up_b]), np.array([sl_, lo_b])
            ln((z0, np.polyval(uh, z0)), (i - 1, np.polyval(uh, i - 1)), 'upper')
            ln((z0, np.polyval(ul, z0)), (i - 1, np.polyval(ul, i - 1)), 'lower')
            polys.append({'pts': [[z0, _r(np.polyval(uh, z0))], [i - 1, _r(np.polyval(uh, i - 1))], [i - 1, _r(np.polyval(ul, i - 1))], [z0, _r(np.polyval(ul, z0))]], 'k': 'fill'})
        pt(ps, 'Asta'); pt(pe, '')
        ann.append({'k': 'label', 'x': int((ps[0] + pe[0]) / 2), 'y': _r((ps[1] + pe[1]) / 2), 't': f'asta {abs(pe[1] - ps[1]):.2f} · {abs(pe[1] / ps[1] - 1) * 100:.1f} %'})
    elif name in ('island_bull', 'island_bear'):
        zones.append({'i0': g['a'] - 1, 'i1': i, 'lo': _r(g['lo']), 'hi': _r(g['hi']), 'k': 'box'})
        ln((g['a'] - 1, g['gap_from']), (i, g['gap_from']), 'level', True)
    elif name.startswith('donch'):
        ln((max(i - 40, 0), g['level']), (i, g['level']), 'level', True)
    elif name.startswith(('box_', 'base_near')):
        zones.append({'i0': g['i0'], 'i1': i - 1, 'lo': _r(g['lo']), 'hi': _r(g['hi']), 'k': 'box'})
        if name == 'base_near_high':
            ln((max(i - 60, 0), g['hi52']), (i, g['hi52']), 'level', True)
    elif name.startswith(('bb_squeeze', 'atr_squeeze')):
        zones.append({'i0': i - 20, 'i1': i - 1, 'lo': _r(g['lo']), 'hi': _r(g['hi']), 'k': 'box'})
    return {'pts': pts, 'lines': lines, 'polys': polys, 'zones': zones, 'ann': ann}


def vector_geom(name: str, F: Dict[str, np.ndarray], i: int) -> Dict:
    """Geometría de los patrones vectorizados (rupturas y compresiones) en la barra i."""
    h, l, c = F['h'], F['l'], F['c']
    g: Dict = {}
    if name.startswith('donch'):
        p = int(name[5:].split('_')[0])
        g['level'] = float(h[i - p:i].max() if name.endswith('_up') else l[i - p:i].min())
    elif name.startswith('box_'):
        g.update(hi=float(h[i - 15:i].max()), lo=float(l[i - 15:i].min()), i0=i - 15)
    elif name == 'base_near_high':
        g.update(hi=float(h[i - 20:i].max()), lo=float(l[i - 20:i].min()), i0=i - 20, hi52=float(h[max(i - 252, 0):i].max()))
    elif name.startswith(('bb_squeeze', 'atr_squeeze')):
        g.update(hi=float(h[i - 20:i].max()), lo=float(l[i - 20:i].min()))
    return g


# ── análisis de un valor ──────────────────────────────────────────────────
MIN_STOP = 1.0       # el stop nunca queda a menos de 1 ATR de la entrada (un stop pegado da R:R absurdos y salta con el ruido)
FAR_STOP = 3.0       # por encima de 3 ATR el estructural se considera lejano: se ofrece además un stop táctico de 1,5 ATR

FORM_LABEL = {'sym_tri': 'Triángulo simétrico', 'asc_triangle': 'Triángulo ascendente', 'desc_triangle': 'Triángulo descendente', 'rising_wedge': 'Cuña ascendente',
              'falling_wedge': 'Cuña descendente', 'rect': 'Rectángulo'}
FORM_UP = {'sym_tri': 'sym_tri_up', 'asc_triangle': 'asc_triangle', 'falling_wedge': 'falling_wedge', 'rect': 'rect_up', 'desc_triangle': 'sym_tri_up', 'rising_wedge': 'sym_tri_up'}
FORM_DN = {'sym_tri': 'sym_tri_dn', 'desc_triangle': 'desc_triangle', 'rising_wedge': 'rising_wedge', 'rect': 'rect_dn', 'asc_triangle': 'sym_tri_dn', 'falling_wedge': 'sym_tri_dn'}
FORM_BIAS = {'asc_triangle': 1, 'falling_wedge': 1, 'desc_triangle': -1, 'rising_wedge': -1}


def rr_box(F: Dict[str, np.ndarray], i: int, dd: int, entry: float, stop: float, tgt: float) -> Dict:
    """Entrada, stop, objetivo (y objetivo parcial T1 al 50 %) con un stop mínimo de 1 ATR y, si el estructural es muy lejano, un stop táctico."""
    a = float(F['atr'][i])
    if dd > 0 and not (stop < entry < tgt):
        stop, tgt = _generic(entry, a, 1, max(abs(entry - stop), 0.6 * a))
    if dd < 0 and not (tgt < entry < stop):
        stop, tgt = _generic(entry, a, -1, max(abs(entry - stop), 0.6 * a))
    stop_min = abs(entry - stop) < MIN_STOP * a
    if stop_min:
        stop = entry - dd * MIN_STOP * a
    risk, reward = abs(entry - stop), abs(tgt - entry)
    far = risk > FAR_STOP * a
    stop_t = entry - dd * 1.5 * a if far else None
    return {'entry': _r(entry), 'stop': _r(stop), 'target': _r(tgt), 't1': _r(entry + dd * 0.5 * reward), 'rr': float(round(reward / max(risk, 1e-9), 2)),
            'risk_pct': float(round(risk / entry * 100, 2)), 'reward_pct': float(round(reward / entry * 100, 2)), 'stop_min': bool(stop_min), 'far': bool(far),
            'stop_t': None if stop_t is None else _r(stop_t), 'rr_t': None if stop_t is None else float(round(reward / max(abs(entry - stop_t), 1e-9), 2)), '_stop': stop, '_stop_t': stop_t, '_tgt': tgt}


def analyze(sym: str, bars: Dict[str, np.ndarray], name: str = '', nb: int = NB, stats: Optional[Dict] = None) -> Dict:
    F = feats.build(bars)
    n = len(F['c'])
    s0 = max(n - nb, 0)
    rec: Dict[str, Dict[int, dict]] = {}
    figs = CH.figures(F, rec)
    sigs: Dict[str, np.ndarray] = {}
    sigs.update(CH.candles(F)); sigs.update(CH.breakouts(F)); sigs.update(figs); sigs.update(extra_candles(F))
    sr = support_resistance(F, s0)
    items: List[Dict] = []
    sh = lambda x: x - s0

    def xmin(geo: Dict) -> int:
        xs = [p[0] for p in geo['pts']] + [x for L_ in geo['lines'] for x in (L_['a'][0], L_['b'][0])] + [p[0] for P in geo['polys'] for p in P['pts']] + [Z['i0'] for Z in geo['zones']]
        return min(xs) if xs else s0

    def shift(geo: Dict) -> None:
        for p in geo['pts']:
            p[0] = sh(p[0])
        for L_ in geo['lines']:
            L_['a'][0] = sh(L_['a'][0]); L_['b'][0] = sh(L_['b'][0])
        for P in geo['polys']:
            for p in P['pts']:
                p[0] = sh(p[0])
        for Z in geo['zones']:
            Z['i0'] = sh(Z['i0']); Z['i1'] = sh(Z['i1'])
        for A in geo['ann']:
            A['x'] = sh(A['x'])

    for pname, sig in sigs.items():
        if pname in SKIP or pname.endswith('_hq'):
            continue
        d, fam, desc = pat_info(pname)
        if pname in ('support_bounce', 'resistance_reject'):
            fam = 'rebotes'
        if pname in ('trendline_up', 'trendline_dn'):
            fam = 'directrices'
        for i in CH.cooldown(np.flatnonzero(sig)):
            i = int(i)
            if i < s0 or i >= n:
                continue
            g = rec.get(pname, {}).get(i) or vector_geom(pname, F, i)
            try:
                geo = geometry(pname, g, F, i)
                if fam == 'velas':
                    m_ = CANDLE_BARS.get(pname, 1)
                    geo['zones'].append({'i0': i - m_ + 1, 'i1': i, 'lo': _r(F['l'][i - m_ + 1:i + 1].min()), 'hi': _r(F['h'][i - m_ + 1:i + 1].max()), 'k': 'candle'})
                stop, tgt, note = levels(pname, g, F, i, d if d != 0 else (1 if F['c'][i] > F['sma50'][i] else -1))
            except Exception as exc:                       # un patrón mal formado no debe romper el valor
                print(f'  [{sym}] {pname}@{i}: {type(exc).__name__}: {exc}', flush=True)
                continue
            dd = d if d != 0 else (1 if tgt > F['c'][i] else -1)
            entry = float(F['c'][i])
            bx = rr_box(F, i, dd, entry, stop, tgt)
            stop, tgt, stop_t = bx.pop('_stop'), bx.pop('_tgt'), bx.pop('_stop_t')
            if bx['stop_min']:
                note += ' El stop estructural quedaba a menos de 1 ATR: se ha ampliado a 1 ATR.'
            if xmin(geo) < s0:                              # la figura empieza antes de la ventana exportada: no se dibuja cortada
                continue
            out = outcome(F, i, dd, stop, tgt)
            out_t = outcome(F, i, dd, stop_t, tgt) if stop_t is not None else None
            active = (n - 1 - i) <= ACTIVE and out['res'] in ('open',)
            shift(geo)
            for o_ in (out, out_t):
                if o_ and 'xi' in o_:
                    o_['xi'] = sh(o_['xi'])
            items.append({'id': f'{pname}@{i - s0}', 'name': pname, 'label': LABEL.get(pname, pname), 'desc': desc, 'dir': dd, 'fam': fam, 'i': i - s0, 'state': 'done', 'q': g.get('q') if isinstance(g, dict) else None, **geo,
                          'rr': bx, 'note': note, 'out': out, 'out_t': out_t, 'active': bool(active), 'ev': (stats or {}).get(pname), 'ev_hq': (stats or {}).get(pname + '_hq')})
    keep = {}
    for it in items:                                                   # varias rupturas de Donchian el mismo día y en el mismo sentido: se queda la de mayor periodo
        if it['name'].startswith('donch'):
            k = (it['i'], it['dir'])
            per = int(it['name'][5:].split('_')[0])
            if k not in keep or per > keep[k][0]:
                keep[k] = (per, it)
    items = [it for it in items if not it['name'].startswith('donch') or keep[(it['i'], it['dir'])][1] is it]
    # compresiones en formación (aún sin ruptura): se dibujan con los dos escenarios
    fm = rec.get('_forming', {})
    if fm:
        import shapes
        i, st = next(iter(fm.items()))
        if i >= s0 and st['kind'] in FORM_LABEL and xmin(geometry('forming_' + st['kind'], dict(U=st['U'], D=st['D'], tu=st['tu'], tl=st['tl'], x0=st['x0'], apex=st['apex']), F, i)) >= s0:
            a = float(F['atr'][i])
            g = dict(U=st['U'], D=st['D'], tu=st['tu'], tl=st['tl'], x0=st['x0'], apex=st['apex'], kind=st['kind'])
            nm_f = 'forming_' + st['kind']
            geo = geometry(nm_f, g, F, i)
            sc = {}
            for key, dd, nm, trig in (('up', 1, FORM_UP[st['kind']], at(st['U'], i + 1) + 0.15 * a), ('dn', -1, FORM_DN[st['kind']], at(st['D'], i + 1) - 0.15 * a)):
                stop, tgt, note = levels(nm, g, F, i, dd, entry=trig)
                bx = rr_box(F, i, dd, trig, stop, tgt)
                for k_ in ('_stop', '_tgt', '_stop_t'):
                    bx.pop(k_, None)
                bx['name'] = nm
                against = FORM_BIAS.get(st['kind'], 0) * dd < 0
                bx['label'] = ('rompe contra el sesgo habitual de la figura' if against else LABEL.get(nm, nm))
                bx['note'] = note
                bx['ev'] = (stats or {}).get(nm)
                sc[key] = bx
            shift(geo)
            q = shapes.compression_quality(st, 0.25, 1.15)
            bias = FORM_BIAS.get(st['kind'], 0)
            items.append({'id': f'{nm_f}@{i - s0}', 'name': nm_f, 'label': FORM_LABEL[st['kind']], 'desc': 'compresión sin romper: se opera la ruptura, no la figura', 'dir': 0, 'fam': 'figuras',
                          'i': i - s0, 'state': 'forming', 'q': q, 'bias': bias, **geo, 'rr': None, 'sc': sc,
                          'note': 'Aún no ha roto: el cierre sigue dentro de las dos rectas. Se opera la ruptura (cierre fuera de la recta) y el sentido no está decidido.' + (' Sesgo habitual: ' + ('alcista' if bias > 0 else 'bajista') + '.' if bias else ''),
                          'out': {'res': 'forming', 'bars': 0, 'ret': None}, 'out_t': None, 'active': True, 'ev': None})
    items.sort(key=lambda x: x['i'])
    sl = slice(s0, n)
    close = F['c']
    sma50, sma200 = F['sma50'][sl], F['sma200'][sl]
    last = float(close[-1])
    summ = {'close': _r(last), 'atr': _r(F['atr'][-1]), 'atrp': round(float(F['atrp'][-1]), 2), 'ret5': round(float(F['ret5'][-1]), 2), 'ret20': round(float(F['ret20'][-1]), 2),
            'rsi14': round(float(F['rsi14'][-1]), 1), 'rsi2': round(float(F['rsi2'][-1]), 1), 'dist200': round(float(F['dist200'][-1]), 2) if np.isfinite(F['dist200'][-1]) else None,
            'pos52': round(float(F['pos52'][-1]), 3) if np.isfinite(F['pos52'][-1]) else None}
    return {'sym': sym, 'name': name, 'asof': int(bars['t'][-1] // 86400), 'n': n - s0,
            'bars': {'t': [int(x // 86400) for x in bars['t'][sl]], 'o': [_r(x) for x in bars['o'][sl]], 'h': [_r(x) for x in bars['h'][sl]], 'l': [_r(x) for x in bars['l'][sl]],
                     'c': [_r(x) for x in bars['c'][sl]], 'v': [int(x) for x in bars['v'][sl]]},
            'sma50': [None if not np.isfinite(x) else _r(x) for x in sma50], 'sma200': [None if not np.isfinite(x) else _r(x) for x in sma200],
            'sr': sr, 'patterns': items, 'summary': summ}


def load_stats(path: str) -> Tuple[Dict, Dict]:
    """(estadísticas por patrón, metadatos: umbrales del mejor de N, horizonte, coste) del estudio `charts`."""
    try:
        d = json.load(open(path, encoding='utf-8'))
        return d.get('patterns', {}), {k: d[k] for k in ('primary', 'cost', 'thr', 'ks') if k in d}
    except Exception:
        return {}, {}


DEFAULT = ('AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX ORCL CRM ADBE QCOM INTC MU PLTR COIN SMCI ARM UBER DIS BA CAT GE JPM BAC GS V MA LLY UNH XOM CVX COST WMT HD KO PEP PFE MRK ABBV NKE SBUX '
           'SPY QQQ IWM GLD TLT XLE XLF SMH').split()


def run_cli(symbols: List[str], out_dir: str, rng: str = '3y', stats_path: Optional[str] = None, raw_dir: Optional[str] = None) -> None:
    import data as D
    stats, meta = load_stats(stats_path) if stats_path else ({}, {})
    bars, failed = D.fetch_many(symbols, rng=rng, workers=6, cache_path=None, verbose=True)
    os.makedirs(out_dir, exist_ok=True)
    if raw_dir:                                  # series completas (para afinar los detectores sin red): solo en el volcado de investigación
        os.makedirs(raw_dir, exist_ok=True)
        for s_, b_ in bars.items():
            json.dump({k: [float(x) for x in b_[k]] for k in ('t', 'o', 'h', 'l', 'c', 'v')}, open(os.path.join(raw_dir, f'{s_}.json'), 'w'), separators=(',', ':'))
    index = []
    for s in symbols:
        if s not in bars or len(bars[s]['c']) < 320:
            print(f'  {s}: sin datos suficientes', flush=True)
            continue
        res = analyze(s, bars[s], stats=stats)
        json.dump(res, open(os.path.join(out_dir, f'{s}.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
        act = [p for p in res['patterns'] if p['active']]
        index.append({'sym': s, 'close': res['summary']['close'], 'asof': res['asof'], 'n': len(res['patterns']), 'atrp': res['summary']['atrp'], 'rsi2': res['summary']['rsi2'],
                      'active': [{'id': p['id'], 'label': p['label'], 'dir': p['dir'], 'fam': p['fam'], 'rr': (p['rr'] or {}).get('rr'), 'q': p.get('q'), 'st': p.get('state')} for p in act][:10]})
    json.dump({'generated': int(__import__('time').time()), 'symbols': index, 'stats_meta': meta}, open(os.path.join(out_dir, 'index.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
    print(f'  chartlab: {len(index)} valores → {out_dir} (fallidos {failed})', flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'charts'))
    ap.add_argument('--stats', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pattern_stats.json'))
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.replace(',', ' ').split() if s.strip()] or DEFAULT
    run_cli(syms, a.out, stats_path=a.stats)
