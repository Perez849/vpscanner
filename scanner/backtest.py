"""
backtest.py — Motor de backtest de dos fases.

Portado LITERALMENTE desde stocks.html (v2026.05.31-r49), runBacktest (1694-end).
Replica: forward scan honesto, gaps de apertura, empate intradía por proximidad,
T2, timeout, coste round-trip, MAE p75 por grupo (edge/range), las DOS pasadas,
clasificación financiera (finWin) y métricas agregadas (wr, avgPnl, etc.).

NOTA sobre el score: calcConfluences es secundaria (solo afecta a `score`, no a
WR/PnL/etiquetas). Se inyecta como callable opcional `conf_fn(df, vp, close, sig)->score`.
Si no se pasa, score=0. Esto permite validar el núcleo del backtest sin depender
del port (largo) de calcConfluences, que se añadirá después con su propia validación.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Callable, Union
from signals import classify_signal
from jsmath import to_fixed

COST_FEE_PCT = 0.10
COST_SLIP_PCT = 0.05
COST_ROUNDTRIP_PCT = (COST_FEE_PCT + COST_SLIP_PCT) * 2  # 0.30


def _round3(x):
    return to_fixed(x, 3)


def run_backtest(df: List[Dict[str, Any]], segs: List[Dict[str, Any]],
                 piv_len: int, n_rows: int, va_pct: float,
                 calc_vp: Callable, conf_fn: Optional[Callable] = None,
                 feature_fn: Optional[Callable] = None) -> Dict[str, Any]:
    trades: List[Dict[str, Any]] = []
    test_segs = segs
    empty = {
        'total': 0, 'wins': 0, 'losses': 0, 'wr': None, 'avgRR': None, 'avgPnl': None,
        'totalPnl': None, 'expectancy': None, 'list': [], 'allScores': [],
        'appliedMaeStop': None,
    }
    if len(test_segs) < 3:
        return empty

    all_scores: List[float] = []

    def process_entry(entry_i: int, vp: Dict[str, Any],
                      mae_stop_pct: Optional[Union[float, Dict[str, float]]] = None) -> bool:
        if entry_i >= len(df):
            return False
        if entry_i + 3 >= len(df):
            return False
        entry_close = df[entry_i]['close']
        df_up = df[:entry_i + 1]
        sig = classify_signal(entry_close, vp, df_up, mae_stop_pct)
        if sig['signal'] == 'NEUTRAL' or not sig['stop'] or not sig['target']:
            return False
        rr = sig['rr']
        if not rr or rr > 15:
            return False
        if abs(sig['target'] - entry_close) / entry_close < 0.005:
            return False

        score = conf_fn(df_up, vp, entry_close, sig) if conf_fn else 0
        feats = feature_fn(df_up, vp, sig) if feature_fn else None
        all_scores.append(score)
        is_long = sig['signal'] in ('LONG', 'RANGE_LONG')

        outcome = 'timeout'
        exit_price: Optional[float] = None
        t2_hit = False
        bars_to_res = 0
        worst_in_win = 0.0
        last_seen_close = entry_close
        last_bar_idx = 0

        fi = 1
        while fi <= 60 and entry_i + fi < len(df):
            bar = df[entry_i + fi]
            bars_to_res = fi
            last_seen_close = bar['close']
            last_bar_idx = fi
            stop_hit = (bar['low'] <= sig['stop']) if is_long else (bar['high'] >= sig['stop'])
            target_hit = (bar['high'] >= sig['target']) if is_long else (bar['low'] <= sig['target'])
            drawdown = ((entry_close - bar['low']) / entry_close * 100) if is_long \
                else ((bar['high'] - entry_close) / entry_close * 100)
            if drawdown > worst_in_win:
                worst_in_win = drawdown

            o = bar['open']
            gap_beyond_stop = (o <= sig['stop']) if is_long else (o >= sig['stop'])
            gap_beyond_target = (o >= sig['target']) if is_long else (o <= sig['target'])

            if gap_beyond_stop:
                outcome = 'loss'; exit_price = o; break
            if gap_beyond_target:
                outcome = 'win'; exit_price = o
                if sig['target2']:
                    fi2 = fi + 1
                    while fi2 <= fi + 30 and entry_i + fi2 < len(df):
                        bar2 = df[entry_i + fi2]
                        s2 = (bar2['low'] <= sig['stop']) if is_long else (bar2['high'] >= sig['stop'])
                        g2 = (bar2['high'] >= sig['target2']) if is_long else (bar2['low'] <= sig['target2'])
                        if s2:
                            break
                        if g2:
                            t2_hit = True; break
                        fi2 += 1
                break

            if stop_hit and target_hit:
                dist_to_stop = abs(o - sig['stop'])
                dist_to_target = abs(o - sig['target'])
                if dist_to_target <= dist_to_stop:
                    outcome = 'win'; exit_price = sig['target']
                    if sig['target2']:
                        fi2 = fi + 1
                        while fi2 <= fi + 30 and entry_i + fi2 < len(df):
                            bar2 = df[entry_i + fi2]
                            s2 = (bar2['low'] <= sig['stop']) if is_long else (bar2['high'] >= sig['stop'])
                            g2 = (bar2['high'] >= sig['target2']) if is_long else (bar2['low'] <= sig['target2'])
                            if s2:
                                break
                            if g2:
                                t2_hit = True; break
                            fi2 += 1
                else:
                    outcome = 'loss'; exit_price = sig['stop']
                break

            if stop_hit:
                outcome = 'loss'; exit_price = sig['stop']; break
            if target_hit:
                outcome = 'win'; exit_price = sig['target']
                if sig['target2']:
                    fi2 = fi + 1
                    while fi2 <= fi + 30 and entry_i + fi2 < len(df):
                        bar2 = df[entry_i + fi2]
                        s2 = (bar2['low'] <= sig['stop']) if is_long else (bar2['high'] >= sig['stop'])
                        g2 = (bar2['high'] >= sig['target2']) if is_long else (bar2['low'] <= sig['target2'])
                        if s2:
                            break
                        if g2:
                            t2_hit = True; break
                        fi2 += 1
                break
            fi += 1

        if outcome == 'timeout':
            exit_price = last_seen_close
            bars_to_res = last_bar_idx or None

        gross_pnl = ((exit_price - entry_close) if is_long else (entry_close - exit_price)) / entry_close * 100 if exit_price else 0
        pnl_pct = _round3(gross_pnl - COST_ROUNDTRIP_PCT)

        pnl50_50 = pnl_pct
        if outcome == 'win' and sig['target2']:
            t1_gross = (sig['target'] - entry_close) if is_long else (entry_close - sig['target'])
            t1_pnl = t1_gross / entry_close * 100
            if t2_hit:
                t2_gross = (sig['target2'] - entry_close) if is_long else (entry_close - sig['target2'])
            else:
                t2_gross = (sig['stop'] - entry_close) if is_long else (entry_close - sig['stop'])
            t2_pnl = t2_gross / entry_close * 100
            pnl50_50 = _round3((t1_pnl * 0.5 + t2_pnl * 0.5) - COST_ROUNDTRIP_PCT)

        entry_ts = df[entry_i]['ts']
        exit_idx = entry_i + bars_to_res if (bars_to_res and entry_i + bars_to_res < len(df)) else len(df) - 1
        exit_ts = df[exit_idx]['ts']

        trades.append({
            'signal': sig['signal'], 'entry': entry_close, 'stop': sig['stop'],
            'target': sig['target'], 'target2': sig['target2'], 'exitPrice': exit_price,
            'rr': rr, 'score': score, 'outcome': outcome, 'pnlPct': pnl_pct, 'pnl50_50': pnl50_50, 't2Hit': t2_hit,
            'labeled': bool(sig.get('longClass') or sig.get('shortClass') or sig.get('rangeClass')),
            'labelClass': sig.get('longClass') or sig.get('shortClass') or sig.get('rangeClass') or None,
            'isElite': sig.get('isElite', False),
            'eliteClass': sig.get('eliteClass'),
            'barsToRes': bars_to_res, 'barsHeld': bars_to_res,
            'maxDrawdown': to_fixed(worst_in_win, 2) if outcome == 'win' else None,
            'ts': entry_ts, 'entryTs': entry_ts, 'exitTs': exit_ts,
            'feats': feats,
        })
        return True

    def run_pass(mae_stop_pct):
        trades.clear()
        all_scores.clear()
        last_exit_bar = -1
        for seg in test_segs:
            seg_start = seg['startI']
            seg_end = min(seg['endI'], len(df) - 2)
            if seg_end - seg_start < 3:
                continue
            sl = seg['slice'] if (seg.get('slice') and len(seg['slice']) >= 10) else df[seg_start:seg_end + 1]
            vp = calc_vp(sl, n_rows, va_pct)
            if not vp:
                continue
            for i in range(seg_start + 1, seg_end + 1):
                entry_i = i
                if entry_i <= last_exit_bar or entry_i + 3 >= len(df):
                    continue
                probe = classify_signal(df[i]['close'], vp, df[:i + 1], mae_stop_pct)
                if probe['signal'] == 'NEUTRAL' or not probe['stop'] or not probe['target']:
                    continue
                ok = process_entry(entry_i, vp, mae_stop_pct)
                if ok:
                    last = trades[-1]
                    last_exit_bar = entry_i + (last['barsToRes'] or 1)

    # FASE 1
    run_pass(None)

    def p75(arr):
        if len(arr) < 6:
            return None
        s = sorted(arr)
        return to_fixed(s[min(len(s) - 1, int(len(s) * 0.75))], 2)

    def is_edge(s):
        return s in ('LONG', 'SHORT')

    mae_edge = p75([t['maxDrawdown'] for t in trades if t['outcome'] == 'win' and t['maxDrawdown'] is not None and is_edge(t['signal'])])
    mae_range = p75([t['maxDrawdown'] for t in trades if t['outcome'] == 'win' and t['maxDrawdown'] is not None and not is_edge(t['signal'])])
    mae_p75_phase1 = p75([t['maxDrawdown'] for t in trades if t['outcome'] == 'win' and t['maxDrawdown'] is not None])

    def to_stop(mae):
        return min(max(mae * 1.3, 1.5), 6.0) if mae is not None else None

    stop_edge = to_stop(mae_edge if mae_edge is not None else mae_p75_phase1)
    stop_range = to_stop(mae_range if mae_range is not None else mae_p75_phase1)
    mae_stops = {'edge': stop_edge, 'range': stop_range} if (stop_edge is not None or stop_range is not None) else None

    applied_mae_stop = mae_stops
    if mae_stops:
        run_pass(mae_stops)

    total = len(trades)
    if total == 0:
        return empty

    for t in trades:
        t['finWin'] = t['pnlPct'] > 0

    # Sumas L-R explícitas (como Array.reduce de JS). sum() de Python puede
    # diferir 1 ULP por su orden de acumulación, y eso cambia el redondeo final.
    def _sum_lr(vals):
        acc = 0.0
        for v in vals:
            acc += v
        return acc

    wins = sum(1 for t in trades if t['finWin'])
    losses = sum(1 for t in trades if not t['finWin'])
    wr = int((wins / total * 100) + 0.5)
    avg_rr = to_fixed(_sum_lr([t['rr'] for t in trades]) / total, 2)
    avg_pnl = to_fixed(_sum_lr([t['pnlPct'] for t in trades]) / total, 2)
    total_pnl = to_fixed(_sum_lr([t['pnlPct'] for t in trades]), 2)

    return {
        'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
        'avgRR': avg_rr, 'avgPnl': avg_pnl, 'totalPnl': total_pnl,
        'expectancy': avg_pnl, 'appliedMaeStop': applied_mae_stop,
        'list': trades, 'allScores': all_scores,
    }