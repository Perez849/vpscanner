#!/usr/bin/env python3
"""
lab_all.py — SUPER laboratorio para los 4 tipos de señal (LONG, SHORT,
RANGE_LONG, RANGE_SHORT). Mismo rigor que lab_long: todas las dimensiones del
marco Volume Profile (PoC/VAH/VAL), confirmación de giro, estructura del VA,
tendencia, y buscador de combinaciones. Objetivo: maximizar WR y PnL/op por tipo.

Para cada tipo se mide el "espejo" correcto:
  · LONG        → caído bajo VAL + giro AL ALZA (verdes, rechazo por abajo)
  · SHORT       → sobreextendido sobre VAH + giro A LA BAJA (rojas, rechazo arriba)
  · RANGE_LONG  → dentro VA bajo PoC, empuje al alza hacia PoC
  · RANGE_SHORT → dentro VA sobre PoC, empuje a la baja hacia PoC

Uso:
  python lab_all.py --fetch          # descarga en vivo (GitHub Actions)
  python lab_all.py --cache F.json   # usa OHLCV cacheado (local)
"""
from __future__ import annotations
import json, os, sys
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vp_core import get_pivot_segments, calc_vp
from signals import classify_signal
from backtest import COST_ROUNDTRIP_PCT
from jsmath import to_fixed

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'symbols.json'), encoding='utf-8'))
PARAMS = CFG['params']

TYPES = ['LONG', 'SHORT', 'RANGE_LONG', 'RANGE_SHORT']
IS_LONG_DIR = {'LONG': True, 'RANGE_LONG': True, 'SHORT': False, 'RANGE_SHORT': False}


def features(df_up: List[Dict], vp: Dict, sig: Dict) -> Dict[str, Any]:
    """Captura todas las dimensiones VP en la entrada (sin look-ahead). Las
    medidas de 'giro' se orientan según la dirección de la señal."""
    close = df_up[-1]['close']
    poc, vah, val = vp['pocPrice'], vp['vahPrice'], vp['valPrice']
    sigt = sig['signal']
    is_long = IS_LONG_DIR[sigt]
    L = len(df_up)
    f = {}

    # ── Distancia al borde relevante y recorrido al objetivo (PoC) ──
    if sigt == 'LONG':
        f['distEdge'] = (val - close) / close * 100      # cuánto bajo el VAL
    elif sigt == 'SHORT':
        f['distEdge'] = (close - vah) / close * 100      # cuánto sobre el VAH
    elif sigt == 'RANGE_LONG':
        f['distEdge'] = (poc - close) / close * 100      # cuánto bajo el PoC (dentro VA)
    else:  # RANGE_SHORT
        f['distEdge'] = (close - poc) / close * 100      # cuánto sobre el PoC (dentro VA)
    f['distToPoc'] = abs(poc - close) / close * 100      # recorrido potencial al objetivo

    # ── Confirmación de giro (orientada a la dirección) ──
    accel = sig.get('clsAccel')
    f['accel'] = accel
    # "frenando" = el movimiento adverso pierde fuerza (a favor de la reversión)
    f['frenando'] = (accel is not None and accel > 0) if is_long else (accel is not None and accel < 0)
    green = sig.get('clsGreen', 0)            # velas verdes consecutivas (al alza)
    # Para señales bajistas, contamos velas ROJAS consecutivas
    red = 0
    k = L - 1
    while k >= max(1, L - 6):
        if df_up[k]['close'] < df_up[k-1]['close']:
            red += 1
        else:
            break
        k -= 1
    f['confirmCandles'] = green if is_long else red   # velas a favor del giro
    f['c2'] = f['confirmCandles'] >= 2
    f['c3'] = f['confirmCandles'] >= 3

    rsi = sig.get('clsRsi')
    f['rsi'] = rsi
    # RSI saliendo de extremo: sobreventa→arriba (long) / sobrecompra→abajo (short)
    if is_long:
        f['rsiSaliendo'] = (rsi is not None and 30 <= rsi <= 45 and f['frenando'])
    else:
        f['rsiSaliendo'] = (rsi is not None and 55 <= rsi <= 70 and f['frenando'])

    # Mecha de rechazo en la dirección del giro
    last = df_up[-1]
    rng = last['high'] - last['low']
    if rng > 0:
        lower_wick = (min(last['open'], last['close']) - last['low']) / rng
        upper_wick = (last['high'] - max(last['open'], last['close'])) / rng
    else:
        lower_wick = upper_wick = 0
    f['rechazo'] = (lower_wick > 0.4) if is_long else (upper_wick > 0.4)

    # Vela fuerte a favor (cierra en el tercio favorable)
    if rng > 0:
        if is_long:
            f['velaFuerte'] = (last['close'] - last['low']) / rng > 0.66 and last['close'] > last['open']
        else:
            f['velaFuerte'] = (last['high'] - last['close']) / rng > 0.66 and last['close'] < last['open']
    else:
        f['velaFuerte'] = False

    # Rebote/caída desde el extremo de 20 velas
    if is_long:
        ext = min(b['low'] for b in df_up[-20:]) if L >= 20 else min(b['low'] for b in df_up)
        f['distFromExtreme'] = (close - ext) / ext * 100 if ext else 0
    else:
        ext = max(b['high'] for b in df_up[-20:]) if L >= 20 else max(b['high'] for b in df_up)
        f['distFromExtreme'] = (ext - close) / ext * 100 if ext else 0

    # Recuperar el borde (volver a entrar al VA): long recupera VAL, short pierde VAH
    prev = df_up[-2]['close'] if L >= 2 else close
    if sigt == 'LONG':
        f['recoverEdge'] = (prev < val) and (close >= val * 0.99)
    elif sigt == 'SHORT':
        f['recoverEdge'] = (prev > vah) and (close <= vah * 1.01)
    else:
        f['recoverEdge'] = False

    # ── Estructura del Value Area ──
    va_w = vah - val
    f['vaWidthPct'] = va_w / close * 100 if close else 0
    f['pocPosition'] = (poc - val) / va_w if va_w > 0 else 0.5
    vols = vp.get('volByLevel', []); total_vol = vp.get('totalVol', 0)
    n_rows = vp.get('nRows', 25)
    edge_lvl = vp.get('valLevel', 0) if is_long else vp.get('vahLevel', 0)
    if vols and total_vol > 0:
        lo = max(0, edge_lvl - 1); hi = min(n_rows - 1, edge_lvl + 1)
        f['edgeVolPct'] = sum(vols[lo:hi+1]) / total_vol * 100
    else:
        f['edgeVolPct'] = 0

    # ── Tendencia y régimen ──
    tb = sig.get('trendBias', 0)
    f['trendBias'] = tb
    # "a favor": para long, tendencia alcista; para short, tendencia bajista
    f['aFavor'] = (tb >= 0) if is_long else (tb <= 0)
    f['rr'] = sig.get('rr')
    return f


def resolve(df, entry_i, sig, is_long):
    entry = df[entry_i]['close']; last_close = entry; last_bar = 0; fi = 1
    while fi <= 60 and entry_i + fi < len(df):
        bar = df[entry_i + fi]; last_close = bar['close']; last_bar = fi
        stop_hit = bar['low'] <= sig['stop'] if is_long else bar['high'] >= sig['stop']
        tgt_hit = bar['high'] >= sig['target'] if is_long else bar['low'] <= sig['target']
        o = bar['open']
        gap_stop = o <= sig['stop'] if is_long else o >= sig['stop']
        gap_tgt = o >= sig['target'] if is_long else o <= sig['target']
        if gap_stop: return 'loss', o, fi
        if gap_tgt: return 'win', o, fi
        if stop_hit and tgt_hit:
            ds = abs(o - sig['stop']); dt = abs(o - sig['target'])
            return ('win', sig['target'], fi) if dt <= ds else ('loss', sig['stop'], fi)
        if stop_hit: return 'loss', sig['stop'], fi
        if tgt_hit: return 'win', sig['target'], fi
        fi += 1
    return 'timeout', last_close, last_bar


def collect(df, n_rows, va_pct, piv_len):
    """Captura (features, resultado) de TODAS las señales, agrupadas por tipo."""
    segs = get_pivot_segments(df, piv_len)
    by_type = {t: [] for t in TYPES}
    last_exit = -1
    for seg in segs:
        ss = seg['startI']; se = min(seg['endI'], len(df) - 2)
        if se - ss < 3: continue
        sl = seg['slice'] if (seg.get('slice') and len(seg['slice']) >= 10) else df[ss:se+1]
        vp = calc_vp(sl, n_rows, va_pct)
        if not vp: continue
        for i in range(ss + 1, se + 1):
            if i <= last_exit or i + 3 >= len(df): continue
            df_up = df[:i+1]
            sig = classify_signal(df[i]['close'], vp, df_up)
            st = sig['signal']
            if st not in TYPES or not sig['stop'] or not sig['target']: continue
            rr = sig['rr']
            if not rr or rr > 15: continue
            entry = df[i]['close']
            if abs(sig['target'] - entry) / entry < 0.005: continue
            is_long = IS_LONG_DIR[st]
            outcome, exitp, bars = resolve(df, i, sig, is_long)
            gross = ((exitp - entry) if is_long else (entry - exitp)) / entry * 100 if exitp else 0
            pnl = to_fixed(gross - COST_ROUNDTRIP_PCT, 3)
            f = features(df_up, vp, sig)
            f['win'] = pnl > 0; f['pnl'] = pnl; f['bars'] = bars
            by_type[st].append(f)
            last_exit = i + (bars or 1)
    return by_type


def stat(trades):
    n = len(trades)
    if n == 0: return None
    wins = sum(1 for t in trades if t['win'])
    acc = 0.0
    for t in trades: acc += t['pnl']
    wp = [t['pnl'] for t in trades if t['win']]
    return {'n': n, 'wr': round(wins/n*100), 'pnl': round(acc/n, 2),
            'avgWin': round(sum(wp)/len(wp), 2) if wp else 0}


def analyze_type(name, trades, results):
    base = stat(trades)
    if not base:
        print(f'\n########## {name}: sin operaciones ##########'); return
    print(f'\n{"#"*60}\n## {name}  ·  BASE: {base["wr"]}% WR · {base["pnl"]:+.2f}%/op · gan {base["avgWin"]:+.2f}% · n={base["n"]}\n{"#"*60}')
    res = {'base': base, 'dimensions': {}, 'combos': {}}

    def measure(title, buckets):
        print(f'  ─ {title}')
        d = {}
        for label, pred in buckets:
            s = stat([t for t in trades if pred(t)])
            d[label] = s
            if s and s['n'] >= 20:
                flag = '✓' if (s['wr'] > base['wr'] and s['pnl'] >= base['pnl']) else '+' if s['wr'] > base['wr'] else ' '
                print(f'    {flag} {label:34s} {s["wr"]:3d}% · {s["pnl"]:+6.2f}%/op · gan {s["avgWin"]:+6.2f}% · n={s["n"]}')
        res['dimensions'][title] = d

    measure('Distancia al borde', [
        ('<3%', lambda t: t['distEdge'] < 3),
        ('3-5%', lambda t: 3 <= t['distEdge'] < 5),
        ('5-8%', lambda t: 5 <= t['distEdge'] < 8),
        ('8-12%', lambda t: 8 <= t['distEdge'] < 12),
        ('12-20%', lambda t: 12 <= t['distEdge'] < 20),
        ('>20%', lambda t: t['distEdge'] >= 20),
    ])
    measure('Recorrido al PoC', [
        ('<3%', lambda t: t['distToPoc'] < 3),
        ('3-8%', lambda t: 3 <= t['distToPoc'] < 8),
        ('8-15%', lambda t: 8 <= t['distToPoc'] < 15),
        ('>15%', lambda t: t['distToPoc'] >= 15),
    ])
    measure('Confirmación de giro', [
        ('frenando', lambda t: t['frenando']),
        ('1+ vela a favor', lambda t: t['confirmCandles'] >= 1),
        ('2+ a favor', lambda t: t['c2']),
        ('3+ a favor', lambda t: t['c3']),
        ('mecha rechazo', lambda t: t['rechazo']),
        ('vela fuerte', lambda t: t['velaFuerte']),
        ('RSI saliendo extremo', lambda t: t['rsiSaliendo']),
        ('rebotó del extremo20 >3%', lambda t: t['distFromExtreme'] > 3),
        ('recupera el borde', lambda t: t['recoverEdge']),
    ])
    measure('Estructura del VA', [
        ('borde alto volumen >10%', lambda t: t['edgeVolPct'] > 10),
        ('VA estrecho <10%', lambda t: t['vaWidthPct'] < 10),
        ('VA ancho >25%', lambda t: t['vaWidthPct'] > 25),
    ])
    measure('Tendencia', [
        ('a favor', lambda t: t['aFavor']),
        ('contra', lambda t: not t['aFavor']),
    ])

    # Combos: borde extendido + giro + estructura
    print('  ─ COMBINACIONES')
    far = lambda t: t['distEdge'] > 8
    combos = {
        'extendido + 2 a favor': lambda t: far(t) and t['c2'],
        'extendido + 2 a favor + a favor tend': lambda t: far(t) and t['c2'] and t['aFavor'],
        'extendido + frenando': lambda t: far(t) and t['frenando'],
        'extendido + rebotó extremo20': lambda t: far(t) and t['distFromExtreme'] > 3,
        'extendido + PoC lejos + giro': lambda t: far(t) and t['distToPoc'] > 10 and (t['frenando'] or t['confirmCandles'] >= 1),
        'extendido + vela fuerte': lambda t: far(t) and t['velaFuerte'],
        'extendido + borde volumen': lambda t: far(t) and t['edgeVolPct'] > 10,
        'extendido + RSI saliendo': lambda t: far(t) and t['rsiSaliendo'],
        'muy extendido>12 + giro': lambda t: t['distEdge'] > 12 and (t['frenando'] or t['confirmCandles'] >= 1),
        # Para RANGE (que no se "extienden" fuera del VA), combos sin 'far':
        'PoC lejos + 2 a favor': lambda t: t['distToPoc'] > 8 and t['c2'],
        'PoC lejos + RSI extremo': lambda t: t['distToPoc'] > 8 and (t['rsi'] is not None and (t['rsi'] > 70 or t['rsi'] < 30)),
        'a favor + 2 a favor velas': lambda t: t['aFavor'] and t['c2'],
    }
    cres = {}
    for label, pred in combos.items():
        s = stat([t for t in trades if pred(t)])
        cres[label] = s
        if s and s['n'] >= 15:
            flag = '✓' if (s['wr'] > base['wr'] and s['pnl'] >= base['pnl']) else '+' if s['wr'] > base['wr'] else ' '
            print(f'    {flag} {label:38s} {s["wr"]:3d}% · {s["pnl"]:+6.2f}%/op · gan {s["avgWin"]:+6.2f}% · n={s["n"]}')
    res['combos'] = cres
    results[name] = res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='lab_cache.json')
    ap.add_argument('--fetch', action='store_true')
    args = ap.parse_args()
    n_rows, va_pct, piv_len = PARAMS['nRows'], PARAMS['vaPct'], PARAMS['pivLen']

    if args.fetch:
        import scanner as sc
        data = {}; syms = CFG['symbols']; done = 0
        print(f'Descargando {len(syms)} símbolos (SOLO 1d)…')
        for sym, meta in syms.items():
            df = sc.fetch_ohlcv(sym, meta, '1d')
            if df and len(df) >= 50:
                data.setdefault(sym, {})['1d'] = df
            done += 1
            if done % 20 == 0: print(f'  ... {done}/{len(syms)}')
        print(f'Descargados {len(data)} símbolos.')
    elif not os.path.exists(args.cache):
        print(f'Falta {args.cache}. Usa --fetch (GitHub).'); return
    else:
        data = json.load(open(args.cache))

    agg = {t: [] for t in TYPES}
    for sym, tfd in data.items():
        for tf, df in tfd.items():
            if tf != '1d':         # SOLO diario
                continue
            if len(df) < 50: continue
            bt = collect(df, n_rows, va_pct, piv_len)
            for t in TYPES:
                agg[t].extend(bt[t])

    results = {}
    for t in TYPES:
        analyze_type(t, agg[t], results)

    json.dump(results, open(os.path.join(HERE, 'lab_all.json'), 'w'), ensure_ascii=False, indent=2)
    print('\nGuardado lab_all.json')


if __name__ == '__main__':
    main()
