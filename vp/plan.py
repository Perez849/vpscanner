"""Plan de cada patrón: entrada, stop inicial, nivel de aseguramiento y resultado simulado.

Plan del operador (el que se muestra, se simula y se cuenta con TODAS las señales: ninguna se descarta):
  · entrada en la apertura siguiente a la señal;
  · stop inicial: el estructural (al otro lado de la figura) si queda entre MIN_STOP y MAX_RISK_ATR ATR de la entrada; si está más lejos se acorta a MAX_RISK_ATR ATR;
  · asegurar: cuando una sesión cierra a LOCK_PCT % o más a favor (+4 %), el stop sube a ese nivel (+4 %) y desde entonces persigue al mejor cierre a TRAIL_ATR ATR sin bajar nunca de él:
    se asegura lo ganado y se deja correr el resto, sin objetivo fijo;
  · como máximo HOLD_LOCK sesiones.
El plan clásico (stop al otro lado de la figura sin límites y objetivo por la medida clásica) se conserva como referencia (`stop_s`, `target_c`, `rr_s`, `outcome`): daba stops del 10, 26 o 43 %.
Los avisos (riesgo de más del MAX_RISK_PCT % del precio, stop más lejos que el nivel que se quiere asegurar) son informativos: no excluyen la señal."""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import shapes
from .catalog import CATALOG
from .shapes import at

HOLD = 40            # sesiones máximas del resultado simulado con el plan clásico
HOLD_LOCK = 60       # sesiones máximas del plan del operador (hay que dejar correr)
MIN_STOP = 1.0       # el stop nunca queda a menos de 1 ATR de la entrada (un stop pegado salta con el ruido)
MAX_RISK_ATR = 2.0   # el stop nunca queda a más de 2 ATR de la entrada: ~5 % en una acción de volatilidad media (ATR del 2,6 %); el estructural más lejano se acorta
LOCK_PCT = 4.0       # rentabilidad (%) que se quiere asegurar: al cerrar una sesión por encima, el stop sube a ese nivel
TRAIL_ATR = 3.0      # después el stop persigue al mejor cierre a 3 ATR (sin bajar del nivel asegurado)
MAX_RISK_PCT = 8.0   # aviso: por encima de este riesgo (% del precio) el valor es muy volátil para un stop de 2 ATR
MAX_RR = 3.0         # (plan clásico de referencia) el objetivo clásico se limita a 3 veces el riesgo
TACT = 1.5           # (estudio del plan) stop táctico de referencia
SPENT = 1.0          # si la medida clásica deja menos de 1 ATR hasta el objetivo, la ruptura ya la consumió (solo informativo)


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
    if name.startswith('channel_'):
        U, D = g['U'], g['D']
        wi = max(at(U, i) - at(D, i), 0.5 * a)                           # los canales son paralelos: la anchura actual es la medida
        if d > 0:
            return (min(at(D, i), g['tl'][-1][1]) if g.get('tl') else at(D, i)) - 0.25 * a, entry + wi
        return (max(at(U, i), g['tu'][-1][1]) if g.get('tu') else at(U, i)) + 0.25 * a, entry - wi
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
    """Plan del operador a partir del stop estructural y el objetivo clásico: stop acotado a [MIN_STOP, MAX_RISK_ATR] ATR y nivel de aseguramiento a LOCK_PCT % a favor
    (`target` es ese nivel; sin objetivo fijo). Devuelve también el plan clásico sin límites (`stop_s`, `target_c`, `rr_s`) y avisos informativos."""
    a = float(F['atr'][i])
    spent = bool(d * (target - entry) < SPENT * a)                  # medida clásica ya consumida (solo informativo)
    if d > 0 and not (stop < entry < target):
        stop, target = generic(entry, a, 1, max(abs(entry - stop), 0.6 * a))
    if d < 0 and not (target < entry < stop):
        stop, target = generic(entry, a, -1, max(abs(entry - stop), 0.6 * a))
    stop_s, target_c = stop, target
    risk_s, reward_c = abs(entry - stop_s), abs(target_c - entry)
    risk = min(max(risk_s, MIN_STOP * a), MAX_RISK_ATR * a)
    stop = entry - d * risk
    lock = entry * (1.0 + d * LOCK_PCT / 100.0)
    risk_pct = risk / entry * 100
    rr = LOCK_PCT / max(risk_pct, 1e-9)                              # lo que se asegura frente a lo que se arriesga
    return {'entry': r(entry), 'stop': r(stop), 'target': r(lock), 't1': None, 'lock_pct': LOCK_PCT, 'rr': float(round(rr, 2)), 'spent': spent, 'ok': True,
            'risk_pct': float(round(risk_pct, 2)), 'risk_atr': float(round(risk / a, 2)), 'risk_hi': bool(risk_pct > MAX_RISK_PCT), 'rr_low': bool(rr < 1.0),
            'stop_cap': bool(risk_s > MAX_RISK_ATR * a), 'stop_min': bool(risk_s < MIN_STOP * a),
            'stop_s': r(stop_s), 'target_c': r(target_c), 'rr_s': float(round(reward_c / max(risk_s, 1e-9), 2)), 'risk_s_pct': float(round(risk_s / entry * 100, 2)),
            'reward_c_pct': float(round(reward_c / entry * 100, 2)),
            '_stop': stop, '_target': lock, '_stop_s': stop_s, '_target_c': target_c}


COST = 0.15          # coste de ida y vuelta (%) que descuenta también el estudio: el resultado «neto» de la web es comparable con el de las estadísticas


def _finish(F: Dict[str, np.ndarray], d: int, e: int, k: int, entry: float, stop: float, target: float, t1: float, res: str, px: float, bars: int, touched_t1: bool) -> Dict:
    """Resultado de una operación ya cerrada en la barra k con todo lo que importa además de «llegó o no al objetivo»: ganancia neta, múltiplo del riesgo (R), recorrido máximo a favor/en contra
    y qué parte del camino hasta el objetivo se llegó a recorrer."""
    h, l = F['h'], F['l']
    gross = d * (px / entry - 1.0) * 100.0
    best = float(h[e:k + 1].max()) if d > 0 else float(l[e:k + 1].min())
    worst = float(l[e:k + 1].min()) if d > 0 else float(h[e:k + 1].max())
    mfe = max(d * (best / entry - 1.0) * 100.0, 0.0) + 0.0                  # + 0.0: sin «-0.0»
    mae = max(-d * (worst / entry - 1.0) * 100.0, 0.0) + 0.0
    risk, dist = abs(entry - stop), abs(target - entry)
    return {'res': res, 'bars': bars, 'ret': r(gross), 'net': r(gross - COST), 'r': r(d * (px - entry) / risk) if risk > 0 else None, 'mfe': r(mfe), 'mae': r(mae),
            'prog': r(min(max(d * (best - entry) / dist, 0.0), 9.99) + 0.0) if dist > 0 else None, 't1': bool(touched_t1), 'xi': k, 'xp': r(px)}


def outcome(F: Dict[str, np.ndarray], i: int, d: int, stop: float, target: float, maxb: int = HOLD, t1: Optional[float] = None) -> Dict:
    """Qué pasó: entrada en la apertura siguiente; el stop gana si se tocan ambos el mismo día; cierre a las `maxb` sesiones.
    Además del resultado (objetivo / stop / cierre por tiempo / en curso) devuelve la ganancia neta, el múltiplo del riesgo, el recorrido máximo y, con el parcial T1 (50 % en T1 y stop a la entrada
    para el resto), lo que habría dado esa gestión (`ret_m`). Quién gana o pierde lo decide el signo de `net`, no el haber llegado al objetivo."""
    o, h, l, c = F['o'], F['h'], F['l'], F['c']
    n = len(c)
    e = i + 1
    if e >= n:
        return {'res': 'open', 'bars': 0, 'ret': None}
    entry = float(o[e])
    if (d > 0 and (entry <= stop or entry >= target)) or (d < 0 and (entry >= stop or entry <= target)):
        return {'res': 'na', 'bars': 0, 'ret': None}
    if t1 is None:
        t1 = entry + d * 0.5 * abs(target - entry)
    touched = False
    for k in range(e, min(n, e + maxb)):
        hit_stop = l[k] <= stop if d > 0 else h[k] >= stop
        hit_tgt = h[k] >= target if d > 0 else l[k] <= target
        if hit_stop:
            gap = (o[k] <= stop) if d > 0 else (o[k] >= stop)
            px = float(o[k]) if gap else stop
            return _finish(F, d, e, k, entry, stop, target, t1, 'stop', px, k - e + 1, touched)
        if hit_tgt:
            gap = (o[k] >= target) if d > 0 else (o[k] <= target)
            px = float(o[k]) if gap else target
            return _finish(F, d, e, k, entry, stop, target, t1, 'target', px, k - e + 1, True)
        if (h[k] >= t1) if d > 0 else (l[k] <= t1):
            touched = True
    k = min(n, e + maxb) - 1
    res = 'open' if k >= n - 1 and e + maxb > n else 'timeout'
    return _finish(F, d, e, k, entry, stop, target, t1, res, float(c[k]), k - e + 1, touched)


def outcome_lock(F: Dict[str, np.ndarray], i: int, d: int, stop: float, lock_pct: float = LOCK_PCT, trail: float = TRAIL_ATR, maxb: int = HOLD_LOCK) -> Dict:
    """Plan del operador: entrada en la apertura siguiente con `stop`; cuando una sesión cierra a `lock_pct` % o más a favor, el stop sube a ese nivel y desde entonces persigue al mejor cierre
    a `trail` ATR (nunca por debajo del nivel asegurado). El stop de cada sesión se fija con los cierres anteriores; el stop gana si se toca el mismo día que cualquier otra cosa; un hueco
    más allá del stop sale a la apertura. `res`: 'target' = se llegó a asegurar (sale con ≥ lo asegurado salvo hueco), 'stop' = saltó el stop inicial, 'timeout' = cierre por tiempo sin asegurar."""
    o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
    n = len(c)
    e = i + 1
    if e >= n:
        return {'res': 'open', 'bars': 0, 'ret': None}
    entry = float(o[e])
    a = float(atr[i])
    if (d > 0 and entry <= stop) or (d < 0 and entry >= stop):
        return {'res': 'na', 'bars': 0, 'ret': None}
    lock = entry * (1.0 + d * lock_pct / 100.0)
    st, locked, best = stop, False, entry
    for k in range(e, min(n, e + maxb)):
        if (l[k] <= st) if d > 0 else (h[k] >= st):
            gap = (o[k] <= st) if d > 0 else (o[k] >= st)
            px = float(o[k]) if gap else st
            out = _finish(F, d, e, k, entry, stop, lock, entry, 'target' if locked else 'stop', px, k - e + 1, locked)
            out['locked'] = locked
            return out
        best = max(best, float(c[k])) if d > 0 else min(best, float(c[k]))
        if not locked and ((c[k] >= lock) if d > 0 else (c[k] <= lock)):
            locked = True
        if locked:
            st = max(lock, best - trail * a) if d > 0 else min(lock, best + trail * a)
    k = min(n, e + maxb) - 1
    res = 'open' if k >= n - 1 and e + maxb > n else ('target' if locked else 'timeout')
    out = _finish(F, d, e, k, entry, stop, lock, entry, res, float(c[k]), k - e + 1, locked)
    out['locked'] = locked
    return out


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
