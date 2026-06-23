#!/usr/bin/env python3
"""
lab_long.py — Laboratorio de análisis de los LONG dentro del marco Volume Profile.

Objetivo: encontrar qué define a un LONG "muy caído que gira y sube al PoC" con
buen WR sin matar el PnL/op. Todo dentro del ecosistema PoC/VAH/VAL.

Reusa el motor validado (vp_core, signals, backtest). Para cada operación LONG
del histórico, captura un vector de características medidas EN EL MOMENTO DE
ENTRADA (sin look-ahead), y luego mide WR y PnL/op de cada partición y combo.

Uso:
  python lab_long.py                 # usa datos cacheados si existen
  python lab_long.py --fetch         # descarga datos frescos (en GitHub Actions)

Salida: lab_long.json (tablas de resultados) + impresión legible por consola.
"""

from __future__ import annotations
import json, os, sys, math
from typing import List, Dict, Any, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vp_core import get_pivot_segments, calc_vp
from signals import classify_signal, calc_ema
from backtest import COST_ROUNDTRIP_PCT
from jsmath import to_fixed

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'symbols.json'), encoding='utf-8'))
PARAMS = CFG['params']


# ──────────────────────────────────────────────────────────────────────────
# Captura de características de UN LONG en el momento de entrada (sin futuro)
# ──────────────────────────────────────────────────────────────────────────
def long_features(df_up: List[Dict], vp: Dict, sig: Dict) -> Dict[str, Any]:
    """Mide todas las dimensiones VP de un LONG en la vela de entrada."""
    close = df_up[-1]['close']
    poc, vah, val = vp['pocPrice'], vp['vahPrice'], vp['valPrice']
    L = len(df_up)
    feat = {}

    # ── A. Profundidad de la caída ──
    feat['distBelowVal'] = (val - close) / close * 100 if val else 0      # cuán caído
    feat['distToPoc'] = (poc - close) / close * 100 if close else 0        # recorrido potencial al objetivo
    feat['distToVah'] = (vah - close) / close * 100 if close else 0        # recorrido potencial al target2

    # ── B. Confirmación de giro ──
    feat['accel'] = sig.get('clsAccel')                                    # caída frenándose si >0
    feat['frenando'] = (sig.get('clsAccel') is not None and sig['clsAccel'] > 0)
    feat['green'] = sig.get('clsGreen', 0)                                 # velas verdes consecutivas
    feat['rsi'] = sig.get('clsRsi')
    # Mecha inferior (rechazo) de la última vela
    last = df_up[-1]
    rng = last['high'] - last['low']
    lower_wick = (min(last['open'], last['close']) - last['low']) / rng if rng > 0 else 0
    feat['lowerWick'] = lower_wick
    feat['rechazo'] = lower_wick > 0.4
    # ¿Recuperó el VAL? (estaba debajo y la vela cierra cerca/encima del VAL)
    prev_close = df_up[-2]['close'] if L >= 2 else close
    feat['recuperaVal'] = (prev_close < val) and (close >= val * 0.99)
    # RSI saliendo de sobreventa: estaba <30 hace poco y ahora sube
    rsi_now = sig.get('clsRsi')
    feat['rsiSaliendo'] = (rsi_now is not None and 30 <= rsi_now <= 45 and feat['frenando'])
    # Distancia al mínimo de las últimas 20 velas (¿ya rebotó algo?)
    lows20 = [b['low'] for b in df_up[-20:]] if L >= 20 else [b['low'] for b in df_up]
    min20 = min(lows20)
    feat['distFromLow20'] = (close - min20) / min20 * 100 if min20 else 0
    # Vela verde fuerte (cierra en el tercio alto de su rango)
    feat['verdeFuerte'] = (rng > 0 and (last['close'] - last['low']) / rng > 0.66 and last['close'] > last['open'])

    # ── C. Contexto del Value Area ──
    va_w = vah - val
    feat['vaWidthPct'] = va_w / close * 100 if close else 0               # ancho del VA en %
    feat['pocPosition'] = (poc - val) / va_w if va_w > 0 else 0.5         # 0=PoC en VAL, 1=PoC en VAH
    # Volumen en el nivel del VAL (suelo fuerte)
    vols = vp.get('volByLevel', [])
    total_vol = vp.get('totalVol', 0)
    val_lvl = vp.get('valLevel', 0)
    n_rows = vp.get('nRows', 25)
    if vols and total_vol > 0:
        lo = max(0, val_lvl - 1); hi = min(n_rows - 1, val_lvl + 1)
        val_vol = sum(vols[lo:hi+1])
        feat['valVolPct'] = val_vol / total_vol * 100
    else:
        feat['valVolPct'] = 0
    feat['volReliable'] = vp.get('volReliable', False)

    # ── D. Tendencia y régimen ──
    feat['trendBias'] = sig.get('trendBias', 0)
    feat['aFavor'] = sig.get('trendBias', 0) >= 0
    feat['roOsc'] = sig.get('diagRoOsc')
    feat['roBlockedDn'] = sig.get('diagRoBreakDn', False)

    # ── E. Geometría riesgo/recorrido ──
    feat['rr'] = sig.get('rr')

    return feat


# ──────────────────────────────────────────────────────────────────────────
# Recolección: corre el motor y captura (features, resultado) de cada LONG
# ──────────────────────────────────────────────────────────────────────────
def collect_long_trades(df: List[Dict], n_rows: int, va_pct: float, piv_len: int) -> List[Dict]:
    """Captura los LONG usando EXACTAMENTE el motor real (run_backtest), para que
    WR/PnL coincidan 1:1 con lo que opera el sistema. feature_fn adjunta las
    dimensiones VP de cada entrada."""
    from backtest import run_backtest
    segs = get_pivot_segments(df, piv_len)
    bt = run_backtest(df, segs, piv_len, n_rows, va_pct, calc_vp,
                      conf_fn=None, feature_fn=long_features)
    out = []
    for tr in bt.get('list', []):
        if tr['signal'] != 'LONG':
            continue
        feat = dict(tr['feats']) if tr.get('feats') else {}
        feat['win'] = tr['pnlPct'] > 0
        feat['pnl'] = tr['pnlPct']
        feat['bars'] = tr['barsToRes']
        out.append(feat)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Análisis: estadística de una partición
# ──────────────────────────────────────────────────────────────────────────
def stat(trades: List[Dict]) -> Optional[Dict]:
    n = len(trades)
    if n == 0:
        return None
    wins = sum(1 for t in trades if t['win'])
    acc = 0.0
    for t in trades:
        acc += t['pnl']
    avg = acc / n
    win_pnls = [t['pnl'] for t in trades if t['win']]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    return {'n': n, 'wr': round(wins / n * 100), 'pnl': round(avg, 2),
            'avgWin': round(avg_win, 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='lab_cache.json', help='archivo con OHLCV cacheado')
    ap.add_argument('--fetch', action='store_true', help='descargar OHLCV en vivo (GitHub Actions)')
    args = ap.parse_args()

    n_rows = PARAMS['nRows']; va_pct = PARAMS['vaPct']; piv_len = PARAMS['pivLen']

    # Cargar OHLCV: --fetch descarga en vivo (GitHub); si no, usa cache local.
    if args.fetch:
        import scanner as sc
        data = {}
        syms = CFG['symbols']
        print(f'Descargando {len(syms)} símbolos (1d)…')
        done = 0
        for sym, meta in syms.items():
            df = sc.fetch_ohlcv(sym, meta, '1d')
            done += 1
            if df and len(df) >= 50:
                data.setdefault(sym, {})['1d'] = df
            if done % 20 == 0:
                print(f'  ... {done}/{len(syms)}')
        print(f'Descargados {len(data)} símbolos con datos.')
    elif not os.path.exists(args.cache):
        print(f'Falta {args.cache}. Usa --fetch (GitHub) o genera el cache.')
        return
    else:
        data = json.load(open(args.cache))

    all_longs = []
    for sym, tfdata in data.items():
        for tf, df in tfdata.items():
            if len(df) < 50:
                continue
            longs = collect_long_trades(df, n_rows, va_pct, piv_len)
            for t in longs:
                t['sym'] = sym; t['tf'] = tf
            all_longs.extend(longs)

    print(f'\nLONG capturados: {len(all_longs)}')
    base = stat(all_longs)
    print(f'BASE: {base["wr"]}% WR · {base["pnl"]}%/op · gan media {base["avgWin"]}%\n')

    results = {'base': base, 'dimensions': {}, 'combos': {}}

    # ── Particiones por cada dimensión ──
    def measure(name, buckets):
        print(f'═══ {name} ═══')
        res = {}
        for label, pred in buckets:
            s = stat([t for t in all_longs if pred(t)])
            res[label] = s
            if s and s['n'] >= 20:
                flag = '✓' if s['wr'] > base['wr'] else ' '
                print(f'  {flag} {label:38s} {s["wr"]:3d}% WR · {s["pnl"]:+6.2f}%/op · gan {s["avgWin"]:+6.2f}% · n={s["n"]}')
        results['dimensions'][name] = res
        print()

    measure('A. Profundidad bajo VAL', [
        ('<3% (cerca)', lambda t: t['distBelowVal'] < 3),
        ('3-5%', lambda t: 3 <= t['distBelowVal'] < 5),
        ('5-8%', lambda t: 5 <= t['distBelowVal'] < 8),
        ('8-12% (muy caído)', lambda t: 8 <= t['distBelowVal'] < 12),
        ('12-20% (desplome)', lambda t: 12 <= t['distBelowVal'] < 20),
        ('>20% (capitulación)', lambda t: t['distBelowVal'] >= 20),
    ])
    measure('A2. Recorrido potencial al PoC', [
        ('PoC <3%', lambda t: t['distToPoc'] < 3),
        ('PoC 3-8%', lambda t: 3 <= t['distToPoc'] < 8),
        ('PoC 8-15%', lambda t: 8 <= t['distToPoc'] < 15),
        ('PoC >15% (gran recorrido)', lambda t: t['distToPoc'] >= 15),
    ])
    measure('B. Confirmación de giro', [
        ('frenando (accel>0)', lambda t: t['frenando']),
        ('1+ vela verde', lambda t: t['green'] >= 1),
        ('2+ verdes', lambda t: t['green'] >= 2),
        ('3+ verdes', lambda t: t['green'] >= 3),
        ('mecha rechazo >40%', lambda t: t['rechazo']),
        ('verde fuerte (cierra arriba)', lambda t: t['verdeFuerte']),
        ('recupera VAL', lambda t: t['recuperaVal']),
        ('RSI saliendo sobreventa', lambda t: t['rsiSaliendo']),
        ('rebotó del mín20 (>3%)', lambda t: t['distFromLow20'] > 3),
        ('aún en el suelo (<1% del mín20)', lambda t: t['distFromLow20'] < 1),
    ])
    measure('C. Estructura del Value Area', [
        ('VAL con mucho volumen (>10%)', lambda t: t['valVolPct'] > 10),
        ('VAL volumen medio (5-10%)', lambda t: 5 <= t['valVolPct'] <= 10),
        ('VA estrecho (<10%)', lambda t: t['vaWidthPct'] < 10),
        ('VA ancho (>25%)', lambda t: t['vaWidthPct'] > 25),
        ('PoC alto en el VA (>0.6)', lambda t: t['pocPosition'] > 0.6),
        ('volumen fiable', lambda t: t['volReliable']),
    ])
    measure('D. Tendencia y régimen', [
        ('a favor tendencia', lambda t: t['aFavor']),
        ('contra tendencia', lambda t: not t['aFavor']),
        ('RO no bloquea', lambda t: not t['roBlockedDn']),
    ])

    # ── Buscador de combinaciones (cruza giro × profundidad × estructura) ──
    print('═══ COMBINACIONES (caído + giro + estructura) ═══')
    deep = lambda t: t['distBelowVal'] > 8
    combos = {
        'caído + frenando': lambda t: deep(t) and t['frenando'],
        'caído + 2 verdes': lambda t: deep(t) and t['green'] >= 2,
        'caído + recupera VAL': lambda t: deep(t) and t['recuperaVal'],
        'caído + mecha rechazo': lambda t: deep(t) and t['rechazo'],
        'caído + RSI saliendo': lambda t: deep(t) and t['rsiSaliendo'],
        'caído + rebotó mín20': lambda t: deep(t) and t['distFromLow20'] > 3,
        'caído + verde fuerte': lambda t: deep(t) and t['verdeFuerte'],
        'caído + VAL alto volumen': lambda t: deep(t) and t['valVolPct'] > 10,
        'caído + frenando + VAL volumen': lambda t: deep(t) and t['frenando'] and t['valVolPct'] > 8,
        'caído + 2verdes + a favor': lambda t: deep(t) and t['green'] >= 2 and t['aFavor'],
        'caído + recupera VAL + frenando': lambda t: deep(t) and t['recuperaVal'] and t['frenando'],
        'caído + PoC lejos + giro': lambda t: deep(t) and t['distToPoc'] > 10 and (t['frenando'] or t['green'] >= 1),
        'caído + recupera VAL + PoC lejos': lambda t: deep(t) and t['recuperaVal'] and t['distToPoc'] > 10,
        'desplome>12 + giro': lambda t: t['distBelowVal'] > 12 and (t['frenando'] or t['green'] >= 1 or t['recuperaVal']),
    }
    combo_res = {}
    for label, pred in combos.items():
        s = stat([t for t in all_longs if pred(t)])
        combo_res[label] = s
        if s and s['n'] >= 15:
            flag = '✓' if s['wr'] > base['wr'] else ' '
            print(f'  {flag} {label:42s} {s["wr"]:3d}% WR · {s["pnl"]:+6.2f}%/op · gan {s["avgWin"]:+6.2f}% · n={s["n"]}')
    results['combos'] = combo_res
    print()

    json.dump(results, open(os.path.join(HERE, 'lab_long.json'), 'w'), ensure_ascii=False, indent=2)
    print('Guardado lab_long.json')


if __name__ == '__main__':
    main()