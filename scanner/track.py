"""
track.py — Seguimiento en vivo (paper trading) de las alertas.

A diferencia de la versión anterior, NO guarda estado intermedio (barsSeen...): cada día
se recalcula cada operación desde su barra de señal con los precios descargados. Así es
idempotente (re-ejecutar no cambia nada) y usa EXACTAMENTE las reglas del backtest:

    señal en el cierre de la barra i  →  entrada en la apertura de i+1
    stop/objetivo = ATR(i) × múltiplos, desde el precio de entrada
    hueco de apertura más allá del nivel → se sale a la apertura
    vela que toca stop y objetivo → STOP (pesimista)
    salida por tiempo al cierre de la barra H (o 'sig': cierre cruza la SMA5 → apertura siguiente)
"""
from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional
import numpy as np


def _exit_cond(F, b: int, sgn: int, kind: str, c: float) -> bool:
    """Condición de salida por señal evaluada al cierre de la barra b."""
    if kind == 'sig':
        return bool(np.isfinite(F['sma5'][b]) and ((c > F['sma5'][b]) if sgn > 0 else (c < F['sma5'][b])))
    if kind == 'rsi':
        r = F['rsi2'][b]
        return bool(np.isfinite(r) and ((r > 70.0) if sgn > 0 else (r < 30.0)))
    if kind == 'ph':
        return bool((c > F['h'][b - 1]) if sgn > 0 else (c < F['l'][b - 1]))
    return False


def _resolve_trailing(F, i: int, sgn: int, K: float, S0: float, H: int, E: float, atr: float, cost: float) -> Dict[str, Any]:
    """Stop de seguimiento (chandelier): extremo favorable hasta la barra ANTERIOR ∓ K×ATR; solo se mueve a favor."""
    n = len(F['c'])
    out: Dict[str, Any] = {'entry': E, 'stop': E - sgn * S0 * atr, 'target': None, 'entryIdx': i + 1}
    ext = E
    last_j = min(H, n - 1 - i)
    exit_px, bars = None, 0
    for j in range(1, last_j + 1):
        b = i + j
        o, h, l, c = float(F['o'][b]), float(F['h'][b]), float(F['l'][b]), float(F['c'][b])
        bars = j
        if sgn > 0:
            stop = max(E - S0 * atr, ext - K * atr)
            if l <= stop:
                exit_px, reason = min(o, stop), 'stop'; break
            ext = max(ext, h)
        else:
            stop = min(E + S0 * atr, ext + K * atr)
            if h >= stop:
                exit_px, reason = max(o, stop), 'stop'; break
            ext = min(ext, l)
        if j == H:
            exit_px, reason = c, 'tiempo'
    if exit_px is not None:
        out.update(status='cerrada', exitPrice=exit_px, reason=reason, bars=bars, exitIdx=i + bars,
                   pnl=sgn * (exit_px - E) / E * 100.0 - cost)
    else:
        lastc = float(F['c'][min(i + last_j, n - 1)])
        trail = max(E - S0 * atr, ext - K * atr) if sgn > 0 else min(E + S0 * atr, ext + K * atr)
        out.update(status='abierta', bars=bars, curPrice=lastc, curPnl=sgn * (lastc - E) / E * 100.0 - cost, trailStop=trail)
    return out


def resolve(F: Dict[str, np.ndarray], i: int, sgn: int, ex: Dict[str, Any], cost: float, entry_mode: str = 'open') -> Dict[str, Any]:
    """
    Estado de una operación con señal en la barra i, con los datos disponibles.
    Devuelve dict: status ('pendiente'|'abierta'|'cerrada'), y según el caso entry/exit/pnl.
    """
    n = len(F['c'])
    kind, T, S, H = ex['kind'], ex.get('T', 0.0), ex['S'], int(ex['H'])
    atr = float(F['atr'][i])
    if i + 1 >= n:
        return {'status': 'pendiente'}
    E = float(F['o'][i + 1]) if entry_mode == 'open' else float(F['c'][i])
    if kind == 'trl':
        return _resolve_trailing(F, i, sgn, T, S, H, E, atr, cost)
    stop = E - sgn * S * atr
    tgt = E + sgn * T * atr if kind == 'atr' else None
    out: Dict[str, Any] = {'entry': E, 'stop': stop, 'target': tgt, 'entryIdx': i + 1}
    last_j = min(H, n - 1 - i)
    exit_px, reason, bars = None, None, 0
    for j in range(1, last_j + 1):
        b = i + j
        o, h, l, c = float(F['o'][b]), float(F['h'][b]), float(F['l'][b]), float(F['c'][b])
        bars = j
        if sgn > 0:
            if o <= stop:
                exit_px, reason = o, 'stop'; break
            if tgt is not None and o >= tgt:
                exit_px, reason = o, 'objetivo'; break
            if l <= stop:
                exit_px, reason = stop, 'stop'; break
            if tgt is not None and h >= tgt:
                exit_px, reason = tgt, 'objetivo'; break
            sig = _exit_cond(F, b, +1, kind, c)
        else:
            if o >= stop:
                exit_px, reason = o, 'stop'; break
            if tgt is not None and o <= tgt:
                exit_px, reason = o, 'objetivo'; break
            if h >= stop:
                exit_px, reason = stop, 'stop'; break
            if tgt is not None and l <= tgt:
                exit_px, reason = tgt, 'objetivo'; break
            sig = _exit_cond(F, b, -1, kind, c)
        if sig:
            if j < H:
                if b + 1 < n:
                    exit_px, reason, bars = float(F['o'][b + 1]), 'señal', j + 1
                    break
                out['pendingExit'] = True          # condición cumplida; se sale en la apertura de mañana
                break
            exit_px, reason = c, 'señal'; break
        if j == H:
            exit_px, reason = c, 'tiempo'
    if exit_px is not None:
        out.update(status='cerrada', exitPrice=exit_px, reason=reason, bars=bars,
                   exitIdx=i + bars,
                   pnl=sgn * (exit_px - E) / E * 100.0 - cost)
    else:
        lastc = float(F['c'][min(i + last_j, n - 1)])
        out.update(status='abierta', bars=bars, curPrice=lastc,
                   curPnl=sgn * (lastc - E) / E * 100.0 - cost)
    return out


def stat_block(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    closed = [t for t in trades if t['status'] == 'cerrada']
    n = len(closed)
    d: Dict[str, Any] = {'n': n, 'open': sum(1 for t in trades if t['status'] == 'abierta'),
                         'pending': sum(1 for t in trades if t['status'] == 'pendiente')}
    if n:
        pn = [t['pnl'] for t in closed]
        w = sum(1 for x in pn if x > 0)
        d.update(wr=round(w / n * 100, 1), mean=round(sum(pn) / n, 2), total=round(sum(pn), 2),
                 pExp=round(sum(t['p'] for t in closed if t.get('p') is not None) /
                            max(1, sum(1 for t in closed if t.get('p') is not None)) * 100, 1)
                 if any(t.get('p') is not None for t in closed) else None)
    else:
        d.update(wr=None, mean=None, total=0.0, pExp=None)
    return d


def update(prev: Optional[Dict[str, Any]], alerts: List[Dict[str, Any]], series_F: Dict[str, Dict[str, np.ndarray]],
           cost_of: Dict[str, float]) -> Dict[str, Any]:
    """
    prev: tracking.json anterior (o None). alerts: alertas de HOY (dicts con id, sym, setup, dir, exit, sigTs...).
    series_F: {sym: features} de los activos con operaciones (necesita o,h,l,c,atr,sma5,t).
    """
    prev = prev or {'trades': []}
    trades: List[Dict[str, Any]] = prev.get('trades', [])
    known = {t['id'] for t in trades}
    for a in alerts:
        if a['id'] in known:
            continue
        trades.append({k: a[k] for k in ('id', 'sym', 'group', 'setup', 'cell', 'dir', 'exit', 'sigTs', 'p', 'sigClose') if k in a}
                      | {'status': 'pendiente'})
    for t in trades:
        if t['status'] == 'cerrada':
            continue
        F = series_F.get(t['sym'])
        if F is None:
            # el activo ya no devuelve datos (exclusión, OPA...): tras 45 días se descarta del seguimiento
            if time.time() * 1000 - t['sigTs'] > 45 * 86400 * 1000:
                t['status'] = 'descartada'
            continue
        ts = np.asarray(F['t'])
        j = np.flatnonzero(np.abs(ts - t['sigTs'] / 1000.0) < 3600)
        if len(j) == 0:
            continue
        r = resolve(F, int(j[0]), int(t['dir']), t['exit'], cost_of.get(t['sym'], 0.30))
        for k in ('entry', 'stop', 'target', 'exitPrice', 'reason', 'bars', 'pnl', 'curPrice', 'curPnl', 'pendingExit', 'trailStop'):
            t.pop(k, None)
        t.update({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()
                  if k not in ('entryIdx', 'exitIdx')})
        t['status'] = r['status']
        if 'entryIdx' in r:
            t['entryTs'] = int(ts[r['entryIdx']] * 1000)
        if 'exitIdx' in r:
            t['exitTs'] = int(ts[r['exitIdx']] * 1000)
    by_setup: Dict[str, Any] = {}
    for sid in sorted({t['setup'] for t in trades}):
        by_setup[sid] = stat_block([t for t in trades if t['setup'] == sid])
    core = [t for t in trades if t['exit']['kind'] != 'trl']        # los pelotazos (trailing) tienen otra naturaleza: se miden aparte
    pel = [t for t in trades if t['exit']['kind'] == 'trl']
    return {'trades': trades, 'stats': {'global': stat_block(core), 'pelotazo': stat_block(pel), 'bySetup': by_setup}}
