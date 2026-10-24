"""Plan de cada patrón: entrada, stop y objetivo con la regla clásica de la figura; objetivo parcial T1, stop mínimo, obstáculos y resultado simulado."""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import shapes
from .catalog import CATALOG
from .shapes import at

HOLD = 40            # sesiones máximas del resultado simulado
MIN_STOP = 1.0       # el stop nunca queda a menos de 1 ATR de la entrada (un stop pegado da R:R absurdos y salta con el ruido)
FAR_STOP = 3.0       # por encima de 3 ATR el estructural se considera lejano: se ofrece además un stop táctico de 1,5 ATR
TACT = 1.5


def r(x: float) -> float:
    x = float(x)
    return round(x, 4 if abs(x) < 10 else 2)


def generic(entry: float, a: float, d: int, risk: float, rmult: float = 2.0) -> Tuple[float, float]:
    risk = max(risk, 0.6 * a)
    return entry - d * risk, entry + d * rmult * risk


def rule(name: str, g: Dict, F: Dict[str, np.ndarray], i: int, d: int, entry: Optional[float] = None) -> Tuple[float, float]:
    """(stop, objetivo) con la regla clásica del patrón `name` completado en la barra i. `entry` permite calcularlo para una ruptura futura."""
    h, l, c, atr = F['h'], F['l'], F['c'], F['atr']
    a = float(atr[i])
    entry = float(c[i]) if entry is None else float(entry)
    if name in ('double_bottom', 'triple_bottom'):
        return min(p[1] for p in g['pts']) - 0.25 * a, g['neck'] + g['hgt']
    if name in ('double_top', 'triple_top'):
        return max(p[1] for p in g['pts']) + 0.25 * a, g['neck'] - g['hgt']
    if name == 'hs_top':
        return g['rs'][1] + 0.25 * a, g['neck'] - g['hgt']
    if name == 'hs_inverse':
        return g['rs'][1] - 0.25 * a, g['neck'] + g['hgt']
    if name in ('asc_triangle', 'desc_triangle', 'sym_tri_up', 'sym_tri_dn', 'rising_wedge', 'falling_wedge', 'rect_up', 'rect_dn'):
        U, D, x0 = g['U'], g['D'], g['x0']
        w0 = max(at(U, x0) - at(D, x0), 0.5 * a)                     # anchura de la figura en su inicio: la medida clásica
        lastlo = min(at(D, i), g['tl'][-1][1]) if g.get('tl') else at(D, i)
        lasthi = max(at(U, i), g['tu'][-1][1]) if g.get('tu') else at(U, i)
        if name in ('asc_triangle', 'sym_tri_up', 'rect_up'):
            return lastlo - 0.25 * a, entry + w0
        if name in ('desc_triangle', 'sym_tri_dn', 'rect_dn'):
            return lasthi + 0.25 * a, entry - w0
        if name == 'falling_wedge':
            start = at(U, x0)
            return lastlo - 0.25 * a, (start if start > entry + 0.5 * a else entry + w0)
        start = at(D, x0)
        return lasthi + 0.25 * a, (start if start < entry - 0.5 * a else entry - w0)
    if name == 'cup_handle':
        rim = max(g['left'][1], g['right'][1])
        return g['handle'][1] - 0.25 * a, rim + (rim - g['bottom'][1])
    if name == 'vcp':
        stop = g['lows'][-1][1] - 0.25 * a
        return stop, entry + 2.0 * max(entry - stop, 0.6 * a)
    if name in ('bull_flag', 'bull_pennant'):
        return g['fl'] - 0.25 * a, g['fh'] + abs(g['pole'])
    if name in ('bear_flag', 'bear_pennant'):
        return g['fh'] + 0.25 * a, g['fl'] - abs(g['pole'])
    if name == 'diamond_bull':
        return g['bot'] - 0.25 * a, entry + g['width']
    if name == 'diamond_bear':
        return g['top'] + 0.25 * a, entry - g['width']
    if name == 'island_bull':
        return g['lo'] - 0.25 * a, max(g['gap_from'], entry + 1.5 * a)
    if name == 'island_bear':
        return g['hi'] + 0.25 * a, min(g['gap_from'], entry - 1.5 * a)
    if name == 'support_bounce':
        return generic(entry, a, 1, entry - min(float(l[i]), g['level']) + 0.5 * a)
    if name == 'resistance_reject':
        return generic(entry, a, -1, max(float(h[i]), g['level']) - entry + 0.5 * a)
    if name in ('trendline_up', 'trendline_dn'):
        return generic(entry, a, d, 1.5 * a)
    if name in ('box_up', 'box_dn'):
        bh, bl = g['hi'], g['lo']
        height = max(bh - bl, 0.5 * a)
        return (bl - 0.25 * a, bh + height) if d > 0 else (bh + 0.25 * a, bl - height)
    if name == 'base_near_high':
        bh, bl = g['hi'], g['lo']
        return bl - 0.25 * a, bh + max(bh - bl, 0.5 * a)
    if name.startswith('donch'):
        return generic(entry, a, d, abs(entry - g['level']) + 1.5 * a)
    if name.startswith(('bb_squeeze', 'atr_squeeze')):
        return generic(entry, a, d, abs(entry - float(F['sma20'][i])))
    m = max(CATALOG[name].bars, 1) if name in CATALOG else 1                # velas: más allá del extremo del patrón
    lo_, hi_ = float(l[i - m + 1:i + 1].min()), float(h[i - m + 1:i + 1].max())
    return generic(entry, a, 1, entry - lo_ + 0.1 * a) if d >= 0 else generic(entry, a, -1, hi_ - entry + 0.1 * a)


def box(F: Dict[str, np.ndarray], i: int, d: int, entry: float, stop: float, target: float) -> Dict:
    """Entrada, stop y objetivo (más el parcial T1 al 50 %) con un stop mínimo de 1 ATR y, si el estructural es muy lejano, un stop táctico de 1,5 ATR."""
    a = float(F['atr'][i])
    if d > 0 and not (stop < entry < target):
        stop, target = generic(entry, a, 1, max(abs(entry - stop), 0.6 * a))
    if d < 0 and not (target < entry < stop):
        stop, target = generic(entry, a, -1, max(abs(entry - stop), 0.6 * a))
    stop_min = abs(entry - stop) < MIN_STOP * a
    if stop_min:
        stop = entry - d * MIN_STOP * a
    risk, reward = abs(entry - stop), abs(target - entry)
    far = risk > FAR_STOP * a
    stop_t = entry - d * TACT * a if far else None
    return {'entry': r(entry), 'stop': r(stop), 'target': r(target), 't1': r(entry + d * 0.5 * reward), 'rr': float(round(reward / max(risk, 1e-9), 2)),
            'risk_pct': float(round(risk / entry * 100, 2)), 'reward_pct': float(round(reward / entry * 100, 2)), 'stop_min': bool(stop_min), 'far': bool(far),
            'stop_t': None if stop_t is None else r(stop_t), 'rr_t': None if stop_t is None else float(round(reward / max(abs(entry - stop_t), 1e-9), 2)),
            '_stop': stop, '_stop_t': stop_t, '_target': target}


def outcome(F: Dict[str, np.ndarray], i: int, d: int, stop: float, target: float, maxb: int = HOLD) -> Dict:
    """Qué pasó: entrada en la apertura siguiente; el stop gana si se tocan ambos el mismo día; cierre a las `maxb` sesiones."""
    o, h, l, c = F['o'], F['h'], F['l'], F['c']
    n = len(c)
    e = i + 1
    if e >= n:
        return {'res': 'open', 'bars': 0, 'ret': None}
    entry = float(o[e])
    if (d > 0 and (entry <= stop or entry >= target)) or (d < 0 and (entry >= stop or entry <= target)):
        return {'res': 'na', 'bars': 0, 'ret': None}
    for k in range(e, min(n, e + maxb)):
        hit_stop = l[k] <= stop if d > 0 else h[k] >= stop
        hit_tgt = h[k] >= target if d > 0 else l[k] <= target
        if hit_stop:
            gap = (o[k] <= stop) if d > 0 else (o[k] >= stop)
            px = float(o[k]) if gap else stop
            return {'res': 'stop', 'bars': k - e + 1, 'ret': r(d * (px / entry - 1) * 100), 'xi': k, 'xp': r(px)}
        if hit_tgt:
            gap = (o[k] >= target) if d > 0 else (o[k] <= target)
            px = float(o[k]) if gap else target
            return {'res': 'target', 'bars': k - e + 1, 'ret': r(d * (px / entry - 1) * 100), 'xi': k, 'xp': r(px)}
    k = min(n, e + maxb) - 1
    res = 'open' if k >= n - 1 and e + maxb > n else 'timeout'
    return {'res': res, 'bars': k - e + 1, 'ret': r(d * (float(c[k]) / entry - 1) * 100), 'xi': k, 'xp': r(c[k])}


def zones(F: Dict[str, np.ndarray], s0: int, k: int = 5, max_levels: int = 10) -> List[Dict]:
    """Soportes y resistencias: zonas con ≥ 2 toques (pivotes de las últimas barras agrupados a ≤ 0,6 ATR), puntuadas por toques y cercanía en el tiempo."""
    h, l, c, atr = F['h'], F['l'], F['c'], F['atr']
    n = len(c)
    ph, pl = shapes.pivot_flags(h, l, k)
    pts = [(j, float(h[j])) for j in np.flatnonzero(ph) if j >= s0 - 30 and j + k < n] + [(j, float(l[j])) for j in np.flatnonzero(pl) if j >= s0 - 30 and j + k < n]
    pts.sort(key=lambda x: x[1])
    a = float(atr[-1]) if np.isfinite(atr[-1]) else float(np.nanmedian(atr))
    clusters: List[List[Tuple[int, float]]] = []
    for p in pts:
        if clusters and p[1] - np.mean([q[1] for q in clusters[-1]]) <= 0.6 * a:
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
        out.append({'p': r(price), 'lo': r(price - 0.25 * a), 'hi': r(price + 0.25 * a), 'touches': len(cl), 'i0': int(min(idxs)) - s0, 'i1': int(max(idxs)) - s0,
                    'kind': 'S' if price < last else 'R', 'score': round(float(len(cl) * (0.5 + rec)), 2)})
    out.sort(key=lambda x: -x['score'])
    return out[:max_levels]


def obstacles(zs: List[Dict], entry: float, target: float) -> List[Dict]:
    """Soportes/resistencias fuertes (≥ 2 toques) entre la entrada y el objetivo: el precio tiene que atravesarlos para llegar."""
    lo, hi = sorted((entry, target))
    out = [{'p': z['p'], 'n': z['touches'], 'k': z['kind']} for z in zs if lo < z['p'] < hi]
    out.sort(key=lambda z: abs(z['p'] - entry))
    return out[:4]
