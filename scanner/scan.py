#!/usr/bin/env python3
"""
scan.py — Robot diario. Descarga el universo, busca los setups VALIDADOS (model/validated.json),
estima la probabilidad de acierto de cada señal y publica las alertas.

Salidas en ./data:
  alerts.json    alertas de hoy (con probabilidad, niveles y contexto) + lista de vigilancia
  tracking.json  seguimiento en vivo (paper trading) de todas las alertas emitidas + estadísticas
  candles.json   velas recientes de los activos con alerta / operación abierta (para el gráfico)
  registry.json  resumen público de los setups validados (qué se opera y con qué estadísticas)
  meta.json      fecha, nº de activos, régimen de mercado, fallos de descarga

Entrada:  la señal se calcula con el CIERRE de hoy; se opera en la APERTURA de la próxima sesión.
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


def jdump(obj, name):
    with open(os.path.join(DATA_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'), default=float)


def jload(name):
    p = os.path.join(DATA_DIR, name)
    try:
        return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None
    except Exception:
        return None


def candles_of(F: Dict[str, np.ndarray], n: int = 90) -> Dict[str, Any]:
    sl = slice(-n, None)
    r = lambda a: [round(float(x), 4) for x in a[sl]]
    return {'t': [int(x * 1000) for x in F['t'][sl]], 'o': r(F['o']), 'h': r(F['h']), 'l': r(F['l']), 'c': r(F['c']),
            'v': [round(float(x)) for x in F['v'][sl]], 'sma50': [None if not np.isfinite(x) else round(float(x), 4) for x in F['sma50'][sl]],
            'sma200': [None if not np.isfinite(x) else round(float(x), 4) for x in F['sma200'][sl]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache', 'prices_2y.pkl.gz'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-notify', action='store_true')
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()

    reg = json.load(open(REG_PATH, encoding='utf-8')) if os.path.exists(REG_PATH) else {'cells': [], 'rules': {}}
    cells = reg.get('cells', [])
    rules = {'minP': 0.70, 'minEv': 0.15, 'watchP': 0.62, 'maxAlerts': 30, **reg.get('rules', {})}
    mdl = MD.LogitModel.from_json(reg['model']) if reg.get('model') else None
    cell_idx = {c['id']: k for k, c in enumerate(cells)}

    uni = UV.load()
    syms = list(uni.keys())
    if args.limit:
        syms = [s for s in syms if uni[s].get('aux')] + [s for s in syms if not uni[s].get('aux')][:args.limit]
    bars, failed = D.fetch_many(syms, rng='2y', workers=args.workers, cache_path=args.cache, max_age_h=6)
    print(f'  descarga lista · {time.time() - t0:.0f}s', flush=True)

    spy, vix = bars.get('SPY'), bars.get('^VIX')
    regime = feats.regime_series(spy, vix) if (spy is not None and vix is not None) else None
    need_setups = [SU.BY_ID[c['setup']] for c in cells]
    sid_of_setup: Dict[str, List[dict]] = {}
    for c in cells:
        sid_of_setup.setdefault(c['setup'], []).append(c)

    cand: List[Dict[str, Any]] = []         # señales de hoy (todas, antes de filtrar por probabilidad)
    series_F: Dict[str, Dict[str, np.ndarray]] = {}
    cost_of: Dict[str, float] = {}
    last_ts_all: List[float] = []
    stress = {'rsi2_lt10': 0, 'n': 0}
    for sym in syms:
        m = uni[sym]
        if m.get('aux') or sym not in bars:
            continue
        g = m['group']
        F = feats.build(bars[sym])
        feats.ensure_regime(F, regime)
        series_F[sym] = F
        cost_of[sym] = SM.COST_RT.get(g, 0.30)
        last = len(F['c']) - 1
        last_ts_all.append(float(F['t'][last]))
        stress['n'] += 1
        stress['rsi2_lt10'] += int(np.isfinite(F['rsi2'][last]) and F['rsi2'][last] < 10 and F['c'][last] > (F['sma200'][last] if np.isfinite(F['sma200'][last]) else 1e18))
        if not need_setups:
            continue
        ev = SU.detect(F, g, need_setups)
        for sname, idx in ev.items():
            if len(idx) == 0 or idx[-1] != last:
                continue
            for c in sid_of_setup[sname]:
                if g not in c['groups']:
                    continue
                cand.append({'sym': sym, 'g': g, 'cell': c, 'F': F, 'last': last})

    # ── probabilidad de cada señal ────────────────────────────────────────
    alerts: List[Dict[str, Any]] = []
    if cand:
        Xs, sids, gids = [], [], []
        for a in cand:
            Xs.append(feats.event_matrix(a['F'], np.array([a['last']]))[0])
            sids.append(cell_idx[a['cell']['id']])
            gids.append(feats.GROUPS.index(a['g']))
        X = np.stack(Xs)
        if mdl is not None:
            p = mdl.predict(X, np.array(sids), np.array(gids))
        else:
            p = np.array([a['cell']['stats']['dev']['wr'] / 100 for a in cand])
        for a, pi, xrow in zip(cand, p, X):
            c, F, i, sym = a['cell'], a['F'], a['last'], a['sym']
            st = c['stats']['dev']
            ev_pct = float(pi * st['avg_win'] + (1 - pi) * st['avg_loss'])
            atr, close = float(F['atr'][i]), float(F['c'][i])
            sgn = int(c['dir'])
            ex = c['exit']
            stop_d = ex['S'] * atr
            tgt_d = ex.get('T', 0.0) * atr
            alerts.append({
                'id': f"{sym}|{sgn}|{int(F['t'][i] * 1000)}", 'sym': sym, 'group': a['g'],
                'sector': uni[sym].get('sector', ''), 'setup': c['setup'], 'cell': c['id'], 'label': c['label'],
                'dir': sgn, 'sigTs': int(F['t'][i] * 1000), 'sigClose': round(close, 4), 'atr': round(atr, 4),
                'atrPct': round(atr / close * 100, 2), 'p': round(float(pi), 4), 'ev': round(ev_pct, 2),
                'exit': ex,
                'levels': {'stopPct': round(-sgn * stop_d / close * 100, 2),
                           'targetPct': round(sgn * tgt_d / close * 100, 2) if ex['kind'] == 'atr' else None,
                           'stop': round(close - sgn * stop_d, 4),
                           'target': round(close + sgn * tgt_d, 4) if ex['kind'] == 'atr' else None},
                'hist': {'wr': round(st['wr'], 1), 'n': st['n'], 'mean': round(st['mean'], 2),
                         'test': c['stats'].get('test')},
                'ctx': {k: (None if not np.isfinite(xrow[feats.FEATURES.index(k)]) else round(float(xrow[feats.FEATURES.index(k)]), 2))
                        for k in ('rsi2', 'ibs', 'ret5', 'dist200', 'dd20', 'vol_ratio', 'spy_up', 'vix', 'vix_z', 'vp_val_atr')},
            })
        # una alerta por (símbolo, dirección): la de mayor probabilidad; el resto queda como 'también'
        best: Dict[str, Dict[str, Any]] = {}
        for al in sorted(alerts, key=lambda x: -x['p']):
            k = f"{al['sym']}|{al['dir']}"
            if k not in best:
                best[k] = al
                al['also'] = []
            else:
                best[k]['also'].append({'setup': al['setup'], 'p': al['p']})
        alerts = sorted(best.values(), key=lambda x: (-x['p'], -x['ev']))

    main_alerts = [a for a in alerts if a['p'] >= rules['minP'] and a['ev'] >= rules['minEv']][:rules['maxAlerts']]
    watch = [a for a in alerts if a not in main_alerts and a['p'] >= rules['watchP'] and a['ev'] > 0][:60]

    # ── seguimiento en vivo ───────────────────────────────────────────────
    prev = jload('tracking.json')
    tracking = TR.update(prev, main_alerts, series_F, cost_of)
    tracking['generatedAt'] = datetime.now(timezone.utc).isoformat()
    g = tracking['stats']['global']
    print(f"  seguimiento: {g['open']} abiertas · {g['pending']} pendientes · {g['n']} cerradas"
          + (f" · WR {g['wr']}% · μ {g['mean']}%" if g['n'] else ''), flush=True)

    # ── velas para los gráficos ───────────────────────────────────────────
    chart_syms = {a['sym'] for a in main_alerts + watch[:20]} | {t['sym'] for t in tracking['trades'] if t['status'] != 'cerrada'}
    candles = {s: candles_of(series_F[s]) for s in chart_syms if s in series_F}

    # ── régimen de mercado (para el encabezado) ───────────────────────────
    mk = {}
    if regime is not None:
        j = len(regime['t']) - 1
        mk = {'spyUp': bool(regime['spy_up'][j] == 1), 'spyDist200': round(float(regime['spy_dist200'][j]), 1),
              'vix': round(float(regime['vix'][j]), 1), 'vixZ': round(float(regime['vix_z'][j]), 2),
              'spyRsi2': round(float(regime['spy_rsi2'][j]), 1)}
    last_day = datetime.fromtimestamp(max(last_ts_all), timezone.utc).strftime('%Y-%m-%d') if last_ts_all else None
    now = datetime.now(timezone.utc).isoformat()
    meta = {'generatedAt': now, 'dataThrough': last_day, 'symbols': stress['n'], 'failed': len(failed),
            'failedList': failed[:80], 'cells': len(cells), 'alerts': len(main_alerts), 'watch': len(watch),
            'market': mk, 'oversoldUptrend': stress['rsi2_lt10'], 'rules': rules,
            'validatedAt': reg.get('generatedAt'), 'validatedThrough': reg.get('dataThrough'),
            'secs': round(time.time() - t0)}

    jdump({'generatedAt': now, 'dataThrough': last_day, 'alerts': main_alerts, 'watch': watch}, 'alerts.json')
    jdump(tracking, 'tracking.json')
    jdump(candles, 'candles.json')
    jdump(meta, 'meta.json')
    jdump({'generatedAt': reg.get('generatedAt'), 'dataThrough': reg.get('dataThrough'), 'periods': reg.get('periods'),
           'rules': rules, 'summary': reg.get('summary'),
           'cells': [{k: v for k, v in c.items()} for c in cells]}, 'registry.json')

    print(f"\nActivos: {stress['n']} · fallidos {len(failed)} · señales {len(cand)} · alertas {len(main_alerts)} · vigilancia {len(watch)} · {time.time() - t0:.0f}s", flush=True)
    for a in main_alerts[:15]:
        print(f"  {a['sym']:10s} {'LARGO' if a['dir'] > 0 else 'CORTO':5s} P={a['p']*100:4.0f}% EV={a['ev']:+.2f}% {a['setup']}", flush=True)
    if not args.no_notify:
        notify.send(main_alerts, meta, tracking)
    notify.step_summary(main_alerts, watch, meta, tracking)


if __name__ == '__main__':
    main()
