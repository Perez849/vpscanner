# -*- coding: utf-8 -*-
"""
seguimiento.py — Seguimiento EN VIVO de las alertas élite.

Cada día que corre el robot:
  1. Registra las alertas nuevas (congela entrada=cierre de hoy, stop, objetivo).
  2. Sigue las vivas día a día con la MISMA lógica que el backtest:
     gaps en apertura, desempate por proximidad si toca stop y objetivo el mismo
     día (marcado como 'ambiguo' para revisión), timeout a 60 sesiones, coste 0.30%.
  3. Recalcula estadísticas reales (WR/PnL en vivo) global, por tipo y por clase.

Persiste todo en seguimiento_vivo.json. No depende del navegador ni de tokens:
lo escribe el propio robot (GitHub Actions) junto al resto de JSON.

Comparable 1:1 con el backtest porque mide igual.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

COST_ROUNDTRIP_PCT = 0.30
MAX_BARS = 60  # timeout, igual que el backtest


def _is_long(signal: str) -> bool:
    return signal in ('LONG', 'RANGE_LONG')


def _resolve_open_trade(tr: Dict[str, Any], bars_after: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resuelve una operación viva con las barras posteriores a su entrada (las que
    aún no se habían visto). Réplica EXACTA de la resolución del backtest.

    bars_after: lista de barras OHLC (dict con open/high/low/close) que han ocurrido
    DESPUÉS de la barra de entrada, en orden cronológico, empezando por la primera
    sesión posterior a la entrada que todavía no se había procesado.

    Devuelve el trade actualizado (sigue 'abierta' o pasa a 'cerrada').
    """
    is_long = _is_long(tr['signal'])
    entry = tr['entry']; stop = tr['stop']; target = tr['target']
    bars_seen = tr.get('barsSeen', 0)

    for bar in bars_after:
        bars_seen += 1
        o, hi, lo = bar['open'], bar['high'], bar['low']
        stop_hit = (lo <= stop) if is_long else (hi >= stop)
        target_hit = (hi >= target) if is_long else (lo <= target)
        gap_stop = (o <= stop) if is_long else (o >= stop)
        gap_target = (o >= target) if is_long else (o <= target)

        outcome = None; exitp = None; ambiguous = False
        if gap_stop:
            outcome, exitp = 'loss', o
        elif gap_target:
            outcome, exitp = 'win', o
        elif stop_hit and target_hit:
            ambiguous = True
            ds = abs(o - stop); dt = abs(o - target)
            if dt <= ds:
                outcome, exitp = 'win', target
            else:
                outcome, exitp = 'loss', stop
        elif stop_hit:
            outcome, exitp = 'loss', stop
        elif target_hit:
            outcome, exitp = 'win', target

        if outcome:
            gross = ((exitp - entry) if is_long else (entry - exitp)) / entry * 100
            tr['barsSeen'] = bars_seen
            tr['status'] = 'cerrada'
            tr['exitPrice'] = round(exitp, 4)
            tr['outcome'] = outcome
            tr['pnlPct'] = round(gross - COST_ROUNDTRIP_PCT, 2)
            tr['ambiguous'] = ambiguous
            tr['closedAt'] = bar['ts']
            tr['closeReason'] = 'stop' if outcome == 'loss' else 'objetivo'
            return tr

        # timeout
        if bars_seen >= MAX_BARS:
            gross = ((bar['close'] - entry) if is_long else (entry - bar['close'])) / entry * 100
            tr['barsSeen'] = bars_seen
            tr['status'] = 'cerrada'
            tr['exitPrice'] = round(bar['close'], 4)
            tr['outcome'] = 'timeout'
            tr['pnlPct'] = round(gross - COST_ROUNDTRIP_PCT, 2)
            tr['ambiguous'] = False
            tr['closedAt'] = bar['ts']
            tr['closeReason'] = 'timeout'
            return tr

    # sigue viva: actualizar barras vistas y precio actual
    tr['barsSeen'] = bars_seen
    if bars_after:
        last = bars_after[-1]
        cur = ((last['close'] - entry) if is_long else (entry - last['close'])) / entry * 100
        tr['curPrice'] = round(last['close'], 4)
        tr['curPnlPct'] = round(cur - COST_ROUNDTRIP_PCT, 2)
    return tr


def _stat_block(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    closed = [t for t in trades if t['status'] == 'cerrada']
    n = len(closed)
    if n == 0:
        return {'n': 0, 'wr': None, 'pnl': None, 'open': len([t for t in trades if t['status'] == 'abierta'])}
    wins = sum(1 for t in closed if t['pnlPct'] > 0)
    pnl = sum(t['pnlPct'] for t in closed)
    return {
        'n': n, 'wr': round(wins / n * 100), 'pnl': round(pnl / n, 2),
        'totalPnl': round(pnl, 2),
        'open': len([t for t in trades if t['status'] == 'abierta']),
        'ambiguous': sum(1 for t in closed if t.get('ambiguous')),
    }


def update(prev: Optional[Dict[str, Any]],
           today_signals: Dict[str, Dict[str, Any]],
           price_tails: Dict[str, List[Dict[str, Any]]],
           today_ts: int) -> Dict[str, Any]:
    """
    prev: contenido anterior de seguimiento_vivo.json (o None la primera vez).
    today_signals: {key: signal_dict} con las alertas élite vivas de hoy
                   (cada una con sym, signal, entry, stop, target, eliteClass, etc.).
    price_tails: {sym: [barras OHLC recientes]} para resolver las operaciones vivas.
                 Debe cubrir desde la entrada de cada operación abierta hasta hoy.
    today_ts: timestamp (ms) de la sesión de hoy.

    Devuelve el nuevo estado completo a guardar.
    """
    prev = prev or {'trades': [], 'generatedAt': None}
    trades: List[Dict[str, Any]] = prev.get('trades', [])

    # Índice de operaciones ya registradas por (sym, signal, entryTs) para no duplicar
    seen_keys = set()
    by_id = {}
    for t in trades:
        seen_keys.add((t['sym'], t['signal'], t['entryTs']))
        by_id[t['id']] = t

    # 1) Registrar alertas NUEVAS de hoy (congelar entrada de hoy)
    for key, s in today_signals.items():
        sym = s['sym']; sig = s['signal']
        k = (sym, sig, today_ts)
        # si ese símbolo ya tiene una operación ABIERTA del mismo tipo, no abrir otra
        already_open = any(
            t['sym'] == sym and t['signal'] == sig and t['status'] == 'abierta'
            for t in trades
        )
        if already_open or k in seen_keys:
            continue
        trades.append({
            'id': f'{sym}-{sig}-{today_ts}',
            'sym': sym, 'signal': sig, 'eliteClass': s.get('labelClass'),
            'entry': s['entry'], 'stop': s['stop'], 'target': s['target'],
            'target2': s.get('target2'),
            'poc': s.get('poc'), 'vah': s.get('vah'), 'val': s.get('val'),
            'entryTs': today_ts, 'openedAt': today_ts,
            'status': 'abierta', 'barsSeen': 0,
            'exitPrice': None, 'outcome': None, 'pnlPct': None,
            'curPrice': None, 'curPnlPct': None,
            'ambiguous': False, 'closedAt': None, 'closeReason': None,
        })

    # 2) Resolver las operaciones ABIERTAS con las barras posteriores a su entrada
    for t in trades:
        if t['status'] != 'abierta':
            continue
        tail = price_tails.get(t['sym'], [])
        # barras estrictamente posteriores a la entrada de esta operación
        after = [b for b in tail if b['ts'] > t['entryTs']]
        # saltar las que ya se contaron en ejecuciones previas
        already = t.get('barsSeen', 0)
        after = after[already:]
        if after:
            _resolve_open_trade(t, after)

    # 3) Estadísticas reales (solo cerradas cuentan para WR/PnL)
    def subset(pred):
        return [t for t in trades if pred(t)]
    by_type = {}
    for s in ('LONG', 'SHORT', 'RANGE_LONG', 'RANGE_SHORT'):
        arr = subset(lambda t, s=s: t['signal'] == s)
        if arr:
            by_type[s] = _stat_block(arr)
    by_class = {}
    for c in ('caida_giro', 'caida_recorrido', 'empuje_favor', 'rango_extendido', 'corto_favor', 'corto_estrecho'):
        arr = subset(lambda t, c=c: t.get('eliteClass') == c)
        if arr:
            by_class[c] = _stat_block(arr)

    return {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'stats': {
            'global': _stat_block(trades),
            'byType': by_type,
            'byClass': by_class,
        },
        'trades': trades,
    }
