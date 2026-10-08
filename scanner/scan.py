#!/usr/bin/env python3
"""
scan.py — Robot diario. Descarga el universo, busca candidatos (sobreventa dentro de tendencia),
estima la probabilidad de acierto de cada uno con el modelo validado (model/validated.json) y
publica solo las alertas que superan el umbral validado.

Salidas en ./data:
  alerts.json    alertas de hoy (probabilidad, niveles, contexto) + lista de vigilancia
  tracking.json  seguimiento en vivo (paper trading) de todas las alertas + estadísticas real vs esperado
  candles.json   velas recientes de los activos con alerta u operación abierta (gráficos)
  registry.json  resumen público de lo validado (estrategias, umbrales, prueba ciega)
  meta.json      fecha, nº de activos, régimen de mercado, fallos de descarga

La señal usa el CIERRE de hoy; se opera en la APERTURA de la próxima sesión.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D
import feats
import model as MD
import notify
import setups as SU
import simulate as SM
import track as TR
import universe as UV

DATA_DIR = os.path.join(HERE, 'data')
REG_PATH = os.path.join(HERE, 'model', 'validated.json')


def jdump(obj, name, out=None):
    with open(os.path.join(out or DATA_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'), default=float)


def jload(name, out=None):
    p = os.path.join(out or DATA_DIR, name)
    try:
        return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None
    except Exception:
        return None


def _nz(a, nd=4):
    return [None if not np.isfinite(x) else round(float(x), nd) for x in a]


def candles_of(F: Dict[str, np.ndarray], n: int = 90) -> Dict[str, Any]:
    sl = slice(-n, None)
    r = lambda a: [round(float(x), 4) for x in a[sl]]
    return {'t': [int(x * 1000) for x in F['t'][sl]], 'o': r(F['o']), 'h': r(F['h']), 'l': r(F['l']), 'c': r(F['c']),
            'v': [round(float(x)) for x in F['v'][sl]], 'sma50': _nz(F['sma50'][sl]), 'sma200': _nz(F['sma200'][sl])}


def exit_trigger(F: Dict[str, np.ndarray], i: int) -> float | None:
    """Cierre que dejaría el RSI(2) por encima de 70 en la próxima barra (activaría la salida)."""
    ru, rd = F['rsi2_ru'][i], F['rsi2_rd'][i]
    if not (np.isfinite(ru) and np.isfinite(rd)):
        return None
    return float(F['c'][i] + max(0.0, (7.0 / 3.0) * rd - ru))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache', 'prices_2y.pkl.gz'))
    ap.add_argument('--registry', default=REG_PATH)
    ap.add_argument('--out', default=DATA_DIR)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--synthetic', type=int, default=0, help='datos sintéticos (pruebas sin red)')
    ap.add_argument('--no-notify', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    reg = json.load(open(args.registry, encoding='utf-8')) if os.path.exists(args.registry) else {'strategies': [], 'patterns': []}
    strategies = reg.get('strategies', [])
    rules = {'maxAlerts': 15, 'watchMargin': 0.05, **reg.get('rules', {})}
    pattern_ids = [p['id'] for p in reg.get('patterns', [])]
    labels = {p['id']: p['label'] for p in reg.get('patterns', [])}
    models = [MD.LogitModel.from_json(s['model']) for s in strategies]

    if args.synthetic:
        import synth
        now_ts = float(int(time.time() // 86400) * 86400)
        uni, bars = synth.universe(args.synthetic, ar=-0.13, n_bars=520, t_end=now_ts - 86400)
        failed: List[str] = []
    else:
        uni = UV.load()
        syms = list(uni.keys())
        if args.limit:
            syms = [s for s in syms if uni[s].get('aux')] + [s for s in syms if not uni[s].get('aux')][:args.limit]
        bars, failed = D.fetch_many(syms, rng='2y', workers=args.workers, cache_path=args.cache, max_age_h=6)
    print(f'  descarga lista · {time.time() - t0:.0f}s', flush=True)

    spy, vix = bars.get('SPY'), bars.get('^VIX')
    regime = feats.regime_series(spy, vix) if (spy is not None and vix is not None) else None

    # ── pasada 1: indicadores y amplitud de mercado ───────────────────────
    series_F: Dict[str, Dict[str, np.ndarray]] = {}
    bre = feats.Breadth()
    for sym, m in uni.items():
        if m.get('aux') or sym not in bars:
            continue
        F = feats.build(bars[sym])
        feats.ensure_regime(F, regime)
        bre.add(F)
        series_F[sym] = F
    bd = bre.finalize()
    for F in series_F.values():
        feats.align_breadth(bd, F['t'], F)
    print(f'  indicadores listos · {len(series_F)} activos · {time.time() - t0:.0f}s', flush=True)

    # ── pasada 2: candidatos de hoy y probabilidad ────────────────────────
    cand: List[Dict[str, Any]] = []
    cost_of: Dict[str, float] = {}
    last_ts_all: List[float] = []
    n_os = 0
    for sym, F in series_F.items():
        g = uni[sym]['group']
        cost_of[sym] = SM.COST_RT.get(g, 0.30)
        last = len(F['c']) - 1
        last_ts_all.append(float(F['t'][last]))
        n_os += int(np.isfinite(F['rsi2'][last]) and F['rsi2'][last] < 10 and np.isfinite(F['sma200'][last]) and F['c'][last] > F['sma200'][last])
        if not strategies:
            continue
        fl = SU.candidate_at_last(F, g, pattern_ids)
        if fl is not None:
            cand.append({'sym': sym, 'g': g, 'F': F, 'last': last, 'fl': fl})

    alerts: List[Dict[str, Any]] = []
    for a in cand:
        F, i, sym, g = a['F'], a['last'], a['sym'], a['g']
        X = feats.event_matrix(F, np.array([i]))
        FLm = a['fl'][None, :]
        gid = np.array([feats.GROUPS.index(g)])
        best = None
        for s, mdl in zip(strategies, models):
            if g not in s.get('groups', feats.GROUPS):
                continue
            p = float(mdl.predict(X, FLm, gid)[0])
            if best is None or p - s['thr'] > best[0] - best[1]['thr']:
                best = (p, s)
        if best is None:
            continue
        p, s = best
        ex = s['exit']
        ev_pct = float(p * s['ev']['aw'] + (1 - p) * s['ev']['al'])
        atr, close = float(F['atr'][i]), float(F['c'][i])
        trig = exit_trigger(F, i) if ex['kind'] == 'rsi' else None
        tgt = close + ex.get('T', 0.0) * atr if ex['kind'] == 'atr' else None
        flagged = [pid for pid, v in zip(pattern_ids, a['fl']) if v > 0]
        alerts.append({
            'id': f"{sym}|1|{int(F['t'][i] * 1000)}", 'sym': sym, 'group': g, 'sector': uni[sym].get('sector', ''),
            'setup': flagged[0] if flagged else '', 'cell': s['id'], 'strategy': s['id'],
            'label': ' · '.join(labels.get(f, f) for f in flagged[:3]), 'patterns': flagged,
            'dir': 1, 'sigTs': int(F['t'][i] * 1000), 'sigClose': round(close, 4), 'atr': round(atr, 4),
            'atrPct': round(atr / close * 100, 2), 'p': round(p, 4), 'pMin': s['thr'], 'ev': round(ev_pct, 2), 'exit': ex,
            'levels': {'stopPct': round(-ex['S'] * atr / close * 100, 2), 'stop': round(close - ex['S'] * atr, 4),
                       'targetPct': round((tgt / close - 1) * 100, 2) if tgt else None, 'target': round(tgt, 4) if tgt else None,
                       'exitTrig': round(trig, 4) if trig else None,
                       'exitTrigPct': round((trig / close - 1) * 100, 2) if trig else None},
            'hist': {'wr': s['stats']['oos']['wr'], 'n': s['stats']['oos']['n'], 'mean': s['stats']['oos']['mean'],
                     'pf': s['stats']['oos']['pf']},
            'ctx': {k: (None if not np.isfinite(X[0][feats.FEATURES.index(k)]) else round(float(X[0][feats.FEATURES.index(k)]), 2))
                    for k in ('rsi2', 'ibs', 'ret5', 'dist200', 'dd20', 'vol_ratio', 'spy_up', 'vix', 'vix_z', 'b_up200', 'b_os', 'vp_val_atr')},
        })
    alerts.sort(key=lambda x: (-(x['p'] - x['pMin']), -x['ev']))
    main_alerts = [a for a in alerts if a['p'] >= a['pMin'] and a['ev'] > 0][:rules['maxAlerts']]
    n_over_cap = len([a for a in alerts if a['p'] >= a['pMin'] and a['ev'] > 0]) - len(main_alerts)
    watch = [a for a in alerts if a not in main_alerts and a['p'] >= a['pMin'] - rules['watchMargin']][:60]

    # ── seguimiento en vivo ───────────────────────────────────────────────
    prev = jload('tracking.json', args.out)
    tracking = TR.update(prev, main_alerts, series_F, cost_of)
    for t in tracking['trades']:
        if t['status'] in ('abierta', 'pendiente') and t['exit']['kind'] == 'rsi' and t['sym'] in series_F:
            F = series_F[t['sym']]
            tr_ = exit_trigger(F, len(F['c']) - 1)
            t['exitTrig'] = round(tr_, 4) if tr_ else None
    tracking['generatedAt'] = datetime.now(timezone.utc).isoformat()
    g = tracking['stats']['global']
    print(f"  seguimiento: {g['open']} abiertas · {g['pending']} pendientes · {g['n']} cerradas"
          + (f" · WR {g['wr']}% · μ {g['mean']}%" if g['n'] else ''), flush=True)

    chart_syms = {a['sym'] for a in main_alerts + watch[:20]} | {t['sym'] for t in tracking['trades'] if t['status'] != 'cerrada'}
    candles = {s: candles_of(series_F[s]) for s in chart_syms if s in series_F}

    mk = {}
    if regime is not None:
        j = len(regime['t']) - 1
        mk = {'spyUp': bool(regime['spy_up'][j] == 1), 'spyDist200': round(float(regime['spy_dist200'][j]), 1),
              'vix': round(float(regime['vix'][j]), 1), 'vixZ': round(float(regime['vix_z'][j]), 2),
              'spyRsi2': round(float(regime['spy_rsi2'][j]), 1)}
        j2 = len(bd['day']) - 1
        while j2 > 0 and not np.isfinite(bd['b_up200'][j2]):
            j2 -= 1
        mk['breadthUp200'] = round(float(bd['b_up200'][j2]), 1)
        mk['breadthOversold'] = round(float(bd['b_os'][j2]), 1)
    last_day = datetime.fromtimestamp(max(last_ts_all), timezone.utc).strftime('%Y-%m-%d') if last_ts_all else None
    now = datetime.now(timezone.utc).isoformat()
    meta = {'generatedAt': now, 'dataThrough': last_day, 'symbols': len(series_F), 'failed': len(failed),
            'failedList': failed[:80], 'strategies': len(strategies), 'candidates': len(cand), 'alerts': len(main_alerts),
            'overCap': max(0, n_over_cap), 'watch': len(watch), 'market': mk, 'oversoldUptrend': n_os, 'rules': rules,
            'validatedAt': reg.get('generatedAt'), 'validatedThrough': reg.get('dataThrough'), 'secs': round(time.time() - t0)}

    jdump({'generatedAt': now, 'dataThrough': last_day, 'alerts': main_alerts, 'watch': watch}, 'alerts.json', args.out)
    jdump(tracking, 'tracking.json', args.out)
    jdump(candles, 'candles.json', args.out)
    jdump(meta, 'meta.json', args.out)
    jdump({'generatedAt': reg.get('generatedAt'), 'dataThrough': reg.get('dataThrough'), 'universe': reg.get('universe'),
           'periods': reg.get('periods'), 'criteria': reg.get('criteria'), 'rules': rules, 'summary': reg.get('summary'),
           'patterns': reg.get('patterns'),
           'strategies': [{k: v for k, v in s.items() if k != 'model'} for s in strategies]}, 'registry.json', args.out)

    print(f"\nActivos: {len(series_F)} · fallidos {len(failed)} · candidatos {len(cand)} · alertas {len(main_alerts)} (+{max(0, n_over_cap)} sobre el tope) · vigilancia {len(watch)} · {time.time() - t0:.0f}s", flush=True)
    for a in main_alerts[:15]:
        print(f"  {a['sym']:10s} P={a['p'] * 100:4.0f}% (mín {a['pMin'] * 100:.0f}%) EV={a['ev']:+.2f}%  {a['label'][:70]}", flush=True)
    if not strategies:
        print('  ⚠ Registro sin estrategias validadas: el sistema no emite alertas (la ventaja no superó la validación).', flush=True)
    if not args.no_notify:
        notify.send(main_alerts, meta, tracking)
    notify.step_summary(main_alerts, watch, meta, tracking)


if __name__ == '__main__':
    main()
