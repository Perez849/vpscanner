#!/usr/bin/env python3
"""
scanner.py — Robot diario. Descarga precios, corre el motor (idéntico al HTML)
y escribe los JSON que la web leerá.

Salidas (en ./data):
  signals.json  — señal viva de hoy por símbolo+timeframe (con etiqueta y niveles)
  history.json  — operaciones cerradas del backtest por símbolo (el histórico real)
  stats.json    — estadísticas agregadas (WR, PnL, etiquetadas vs no) globales
  meta.json     — fecha de ejecución, nº de símbolos OK / fallidos

Diseño defensivo: si un símbolo falla (red, datos), se salta y se registra,
nunca tumba la ejecución entera. Yahoo primero, Stooq de respaldo.
"""

from __future__ import annotations
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vp_core import get_pivot_segments, calc_vp
from signals import classify_signal
from backtest import run_backtest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
CFG = json.load(open(os.path.join(HERE, 'symbols.json'), encoding='utf-8'))

PARAMS = CFG['params']
INTERVALS = CFG['intervals']
SYMBOLS = CFG['symbols']

UA = 'Mozilla/5.0 (compatible; VPScannerBot/1.0)'


# ── DESCARGA DE DATOS ─────────────────────────────────────────────────────
def _http_get(url: str, timeout: int = 20) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def fetch_yahoo(yh_sym: str, interval_key: str) -> Optional[List[Dict[str, Any]]]:
    iv = INTERVALS[interval_key]
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{yh_sym}'
           f'?interval={iv["yahoo"]}&range={iv["yrange"]}')
    raw = _http_get(url)
    if not raw:
        return None
    try:
        j = json.loads(raw)
        res = j['chart']['result'][0]
        ts = res['timestamp']
        q = res['indicators']['quote'][0]
        o, h, l, c = q['open'], q['high'], q['low'], q['close']
        v = q.get('volume', [0] * len(ts))
        bars = []
        for i in range(len(ts)):
            if None in (o[i], h[i], l[i], c[i]):
                continue
            bars.append({
                'ts': ts[i] * 1000,
                'open': float(o[i]), 'high': float(h[i]),
                'low': float(l[i]), 'close': float(c[i]),
                'volume': float(v[i] or 0),
            })
        return bars if len(bars) >= 30 else None
    except Exception:
        return None


def fetch_stooq(stooq_sym: str, interval_key: str) -> Optional[List[Dict[str, Any]]]:
    iv = INTERVALS[interval_key]
    url = f'https://stooq.com/q/d/l/?s={stooq_sym}&i={iv["stooq"]}'
    raw = _http_get(url)
    if not raw:
        return None
    try:
        text = raw.decode('utf-8', errors='ignore').strip()
        rows = text.split('\n')
        if len(rows) < 31 or not rows[0].lower().startswith('date'):
            return None
        bars = []
        for row in rows[1:]:
            parts = row.split(',')
            if len(parts) < 6:
                continue
            d, op, hi, lo, cl, vol = parts[:6]
            try:
                ts = int(datetime.strptime(d, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
                bars.append({'ts': ts, 'open': float(op), 'high': float(hi),
                             'low': float(lo), 'close': float(cl), 'volume': float(vol or 0)})
            except Exception:
                continue
        return bars if len(bars) >= 30 else None
    except Exception:
        return None


def fetch_ohlcv(sym: str, meta: Dict[str, Any], interval_key: str,
                retries: int = 2) -> Optional[List[Dict[str, Any]]]:
    for attempt in range(retries):
        bars = fetch_yahoo(meta['yahoo'], interval_key)
        if bars:
            return bars
        bars = fetch_stooq(meta['stooq'], interval_key)
        if bars:
            return bars
        time.sleep(1.5 * (attempt + 1))
    return None


# ── PROCESADO DE UN SÍMBOLO ───────────────────────────────────────────────
def process_symbol(sym: str, meta: Dict[str, Any], interval_key: str) -> Optional[Dict[str, Any]]:
    df = fetch_ohlcv(sym, meta, interval_key)
    if not df or len(df) < 50:
        return None

    piv_len = PARAMS['pivLen']
    n_rows = PARAMS['nRows']
    va_pct = PARAMS['vaPct']

    segs = get_pivot_segments(df, piv_len)
    bt = run_backtest(df, segs, piv_len, n_rows, va_pct, calc_vp, conf_fn=None)

    # Señal viva de HOY: VP del último tramo (igual criterio que el HTML).
    last_seg = segs[-1] if segs else None
    if last_seg and last_seg.get('slice') and len(last_seg['slice']) >= 10:
        vp_now = calc_vp(last_seg['slice'], n_rows, va_pct)
    else:
        vp_now = calc_vp(df, n_rows, va_pct)

    sig_now = None
    if vp_now:
        applied = bt.get('appliedMaeStop')
        s = classify_signal(df[-1]['close'], vp_now, df, applied)
        # MODO FILTRADO TOTAL: solo se publica la señal si es ÉLITE.
        if s['signal'] != 'NEUTRAL' and s.get('isElite'):
            sig_now = {
                'signal': s['signal'], 'entry': df[-1]['close'],
                'stop': s['stop'], 'target': s['target'], 'target2': s['target2'],
                'rr': s['rr'], 'labelClass': s.get('eliteClass'),
                'isNewCross': s['isNewCross'], 'trendBias': s['trendBias'],
                'poc': vp_now['pocPrice'], 'vah': vp_now['vahPrice'], 'val': vp_now['valPrice'],
                'clsRsi': s['clsRsi'], 'clsAccel': s['clsAccel'], 'clsGreen': s['clsGreen'],
                'scenario': s['scenario'],
            }

    # Histórico: SOLO operaciones élite (las que el sistema de verdad operaría).
    elite_trades = [t for t in bt.get('list', []) if t.get('isElite')]
    trades = [{
        'signal': t['signal'], 'entry': t['entry'], 'stop': t['stop'],
        'target': t['target'], 'exitPrice': t['exitPrice'], 'outcome': t['outcome'],
        'pnlPct': t['pnlPct'], 'labeled': True, 'labelClass': t.get('eliteClass'),
        'barsToRes': t['barsToRes'], 'entryTs': t['entryTs'], 'exitTs': t['exitTs'],
    } for t in elite_trades]

    # Backtest del activo recalculado SOLO con élites (lo que verás es lo que opera).
    def _agg(ts):
        n = len(ts)
        if n == 0:
            return {'total': 0, 'wr': None, 'avgPnl': None, 'totalPnl': None, 'avgRR': None}
        wins = sum(1 for t in ts if t['pnlPct'] > 0)
        tot = 0.0; rr = 0.0
        for t in ts:
            tot += t['pnlPct']; rr += (t.get('rr') or 0)
        return {'total': n, 'wr': round(wins / n * 100), 'avgPnl': round(tot / n, 2),
                'totalPnl': round(tot, 2), 'avgRR': round(rr / n, 2)}
    bt_elite = _agg(elite_trades)

    return {
        'sym': sym, 'tf': INTERVALS[interval_key].get('label', interval_key),
        'sector': meta.get('sector', ''),
        'signal': sig_now,
        'bt': bt_elite,
        'trades': trades,
    }


def stat_block(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {'n': 0, 'wr': None, 'pnl': None}
    wins = sum(1 for t in trades if t['pnlPct'] > 0)
    acc = 0.0
    for t in trades:
        acc += t['pnlPct']
    return {'n': n, 'wr': round(wins / n * 100), 'pnl': round(acc / n, 2)}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    signals_out: Dict[str, Any] = {}
    history_out: Dict[str, Any] = {}
    all_trades: List[Dict[str, Any]] = []
    ok, failed = [], []

    interval_keys = ['1d']   # SOLO diario (decisión del usuario)
    total_jobs = len(SYMBOLS) * len(interval_keys)
    done = 0

    for sym, meta in SYMBOLS.items():
        for ik in interval_keys:
            done += 1
            try:
                r = process_symbol(sym, meta, ik)
            except Exception as e:
                r = None
                print(f'  ! {sym} {ik}: error {e}', flush=True)
            key = f'{sym}|{ik}'
            if r is None:
                failed.append(key)
                continue
            ok.append(key)
            if r['signal']:
                signals_out[key] = {**r['signal'], 'sym': sym, 'tf': r['tf'],
                                    'sector': r['sector'], 'bt': r['bt']}
            history_out[key] = {'sym': sym, 'tf': r['tf'], 'sector': r['sector'],
                                'bt': r['bt'], 'trades': r['trades']}
            all_trades.extend(r['trades'])
            if done % 20 == 0:
                print(f'  ... {done}/{total_jobs}', flush=True)

    # Todo all_trades ya es ÉLITE (el sistema filtrado). Desglose por tipo y clase.
    by_type = {}
    for s in ('LONG', 'SHORT', 'RANGE_LONG', 'RANGE_SHORT'):
        arr = [t for t in all_trades if t['signal'] == s]
        by_type[s] = {'all': stat_block(arr)}
    by_class = {}
    for c in ('pelotazo_max', 'pelotazo', 'premium', 'rebote', 'seguro'):
        arr = [t for t in all_trades if t['labelClass'] == c]
        if arr:
            by_class[c] = stat_block(arr)
    stats_out = {
        'global': stat_block(all_trades),
        'elite': True,
        'byType': by_type,
        'byClass': by_class,
    }

    now = datetime.now(timezone.utc).isoformat()
    meta_out = {'generatedAt': now, 'symbolsOK': len(ok), 'symbolsFailed': len(failed),
                'failed': failed, 'totalTrades': len(all_trades)}

    json.dump(signals_out, open(os.path.join(DATA_DIR, 'signals.json'), 'w'), ensure_ascii=False)
    json.dump(history_out, open(os.path.join(DATA_DIR, 'history.json'), 'w'), ensure_ascii=False)
    json.dump(stats_out, open(os.path.join(DATA_DIR, 'stats.json'), 'w'), ensure_ascii=False)
    json.dump(meta_out, open(os.path.join(DATA_DIR, 'meta.json'), 'w'), ensure_ascii=False, indent=2)

    print(f'\nOK: {len(ok)} · Fallidos: {len(failed)} · Trades: {len(all_trades)}', flush=True)
    print(f'Señales vivas: {len(signals_out)}', flush=True)
    if failed:
        print('Fallidos:', ', '.join(failed[:20]), flush=True)


if __name__ == '__main__':
    main()
