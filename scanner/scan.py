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

Modo --mode preclose (≈15:20 ET): la misma lógica sobre la barra PROVISIONAL de hoy, solo EE.UU. Sus avisos
se compran AL CIERRE con una orden MOC (límite NYSE 15:50 ET) y se guardan en alerts_pre.json. El seguimiento
las registra con mode='moc' (entrada = cierre final) y el escaneo posterior al cierre no las duplica.
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

PRE_GROUPS = {'us_large', 'us_mid', 'us_small', 'etf'}     # grupos con orden MOC y validados con barras de 60 min
PRE_WINDOW = (15 * 60 + 10, 15 * 60 + 40)                  # minutos ET: antes no hay barra útil; después no da tiempo (límite MOC NYSE 15:50)


def in_preclose_window(now: datetime | None = None) -> bool:
    from zoneinfo import ZoneInfo
    et = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo('America/New_York'))
    return et.weekday() < 5 and PRE_WINDOW[0] <= et.hour * 60 + et.minute <= PRE_WINDOW[1]


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
    ap.add_argument('--mode', choices=('close', 'preclose'), default='close')
    ap.add_argument('--force', action='store_true', help='preclose: ignora la ventana horaria (pruebas)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    pre = args.mode == 'preclose'
    if pre and not args.force and not in_preclose_window():
        print('Fuera de la ventana previa al cierre (15:10–15:40 ET, L–V): nada que hacer.', flush=True)
        return

    reg = json.load(open(args.registry, encoding='utf-8')) if os.path.exists(args.registry) else {'strategies': [], 'patterns': []}
    strategies = reg.get('strategies', [])
    rules = {'nPerDay': 5, 'watchMargin': 0.03, **reg.get('rules', {})}
    pattern_ids = [p['id'] for p in reg.get('patterns', [])]
    labels = {p['id']: p['label'] for p in reg.get('patterns', [])}
    models = [MD.LogitModel.from_json(s['model']) for s in strategies]
    pel = None if pre else reg.get('pelotazo')           # los pelotazos solo se validaron entrando a la apertura
    pel_model = MD.LogitModel.from_json(pel['model']) if pel else None
    pel_pattern_ids = [p['id'] for p in pel['patterns']] if pel else []
    pel_labels = {p['id']: p['label'] for p in pel['patterns']} if pel else {}

    if args.synthetic:
        import synth
        now_ts = float(int(time.time() // 86400) * 86400)
        uni, bars = synth.universe(args.synthetic, ar=-0.13, n_bars=520, t_end=now_ts - 86400)
        failed: List[str] = []
        if pre:
            for b in bars.values():
                b['partial'] = np.array([1.0])
    else:
        uni = UV.load()
        if pre:
            uni = {k: m for k, m in uni.items() if m.get('aux') or m['group'] in PRE_GROUPS}
        syms = list(uni.keys())
        if args.limit:
            syms = [s for s in syms if uni[s].get('aux')] + [s for s in syms if not uni[s].get('aux')][:args.limit]
        bars, failed = D.fetch_many(syms, rng='2y', workers=args.workers, cache_path=None if pre else args.cache,
                                    max_age_h=6, keep_partial=pre)
    print(f'  descarga lista · {time.time() - t0:.0f}s', flush=True)

    expected = len([x for x in uni if not uni[x].get('aux')])
    if not args.limit and len([x for x in bars if not uni.get(x, {}).get('aux')]) < 0.6 * expected:
        raise SystemExit(f'Descarga incompleta ({len(bars)}/{expected}): no se sobrescriben los datos publicados.')
    partial_syms = {x for x, b in bars.items() if 'partial' in b}      # símbolos cuya última barra es la sesión en curso
    bars = {x: {k: v for k, v in b.items() if k != 'partial'} for x, b in bars.items()}
    if pre:
        n_part = len([x for x in partial_syms if not uni.get(x, {}).get('aux')])
        print(f'  barras provisionales de hoy: {n_part}/{expected}', flush=True)
        if 'SPY' not in partial_syms or n_part < 0.5 * expected:
            print('Sin sesión en curso (festivo, cierre anticipado o datos sin barra de hoy): no se emiten avisos previos.', flush=True)
            return
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
    pel_cand: List[Dict[str, Any]] = []
    cost_of: Dict[str, float] = {}
    last_ts_all: List[float] = []
    n_os = 0
    for sym, F in series_F.items():
        g = uni[sym]['group']
        cost_of[sym] = SM.COST_RT.get(g, 0.30)
        if pre and sym not in partial_syms:
            continue
        last = len(F['c']) - 1
        last_ts_all.append(float(F['t'][last]))
        n_os += int(np.isfinite(F['rsi2'][last]) and F['rsi2'][last] < 10 and np.isfinite(F['sma200'][last]) and F['c'][last] > F['sma200'][last])
        if not strategies:
            continue
        fl = SU.candidate_at_last(F, g, pattern_ids)
        if fl is not None:
            cand.append({'sym': sym, 'g': g, 'F': F, 'last': last, 'fl': fl})
        if pel and not pel.get('paused') and g in pel['groups']:
            flp = SU.candidate_at_last(F, g, pel_pattern_ids)
            if flp is not None:
                pel_cand.append({'sym': sym, 'g': g, 'F': F, 'last': last, 'fl': flp})

    # probabilidad de cada candidato en cada plan; cada plan elige sus N mejores del día
    per_plan: Dict[str, List[Dict[str, Any]]] = {s['id']: [] for s in strategies}
    for a in cand:
        F, i, g = a['F'], a['last'], a['g']
        X = feats.event_matrix(F, np.array([i]))
        FLm = a['fl'][None, :]
        gid = np.array([feats.GROUPS.index(g)])
        a['X'] = X
        for s, mdl in zip(strategies, models):
            if s.get('paused') or g not in s.get('groups', []):
                continue
            if float(X[0][feats.FEATURES.index('atrp')]) < s.get('minAtrPct', 0):        # suelo de volatilidad del plan
                continue
            p_raw = float(mdl.predict_raw(X, FLm, gid)[0])
            if p_raw >= s['floorRaw']:
                per_plan[s['id']].append({'a': a, 'p_raw': p_raw, 'p': float(mdl.predict(X, FLm, gid)[0])})
    for lst in per_plan.values():
        lst.sort(key=lambda r: -r['p_raw'])

    by_sym: Dict[str, Dict[str, Any]] = {}
    watch_syms: Dict[str, Dict[str, Any]] = {}
    for s in strategies:
        n_day = int(s.get('nPerDay', rules['nPerDay']))
        for rank, r in enumerate(per_plan[s['id']]):
            a = r['a']
            F, i, sym, g = a['F'], a['last'], a['sym'], a['g']
            ex = s['exit']
            atr, close = float(F['atr'][i]), float(F['c'][i])
            # Probabilidad HONESTA: el modelo apenas separa ganadoras de perdedoras (AUC ≈ 0,5), así que se muestra el acierto HISTÓRICO
            # de su puesto en el ranking del día (con el rango por año) y no la probabilidad individual del modelo.
            st_ = s['stats']; hit = (st_.get('calibration') or {}).get('hit'); rh = (st_.get('byRank') or {}).get(str(rank + 1))
            p_disp = (rh['wr'] if rh else (hit['wr'] if hit else st_['oos']['wr'])) / 100.0
            ev_pct = float(rh['mean'] if rh else st_['oos']['mean'])
            trig = exit_trigger(F, i) if ex['kind'] == 'rsi' else None
            tgt = close + ex.get('T', 0.0) * atr if ex['kind'] == 'atr' else None
            plan = {'strategy': s['id'], 'role': s.get('role'), 'label': s['label'], 'blurb': s.get('blurb'), 'p': round(p_disp, 4), 'pModel': round(r['p'], 4),
                    'pRange': [hit['yearMin'], hit['yearMax']] if hit and hit.get('yearMin') is not None else None,
                    'ev': round(ev_pct, 2), 'exit': ex, 'rank': rank + 1,
                    'levels': {'stopPct': round(-ex['S'] * atr / close * 100, 2), 'stop': round(close - ex['S'] * atr, 4),
                               'targetPct': round((tgt / close - 1) * 100, 2) if tgt else None, 'target': round(tgt, 4) if tgt else None,
                               'exitTrig': round(trig, 4) if trig else None,
                               'exitTrigPct': round((trig / close - 1) * 100, 2) if trig else None},
                    'hist': {'wr': s['stats']['oos']['wr'], 'n': s['stats']['oos']['n'], 'mean': s['stats']['oos']['mean'], 'pf': s['stats']['oos']['pf']},
                    'rankHist': s['stats'].get('byRank', {}).get(str(rank + 1))}
            target = by_sym if rank < n_day else watch_syms
            if rank >= n_day and rank >= n_day + 20:
                continue
            if sym not in target:
                flagged = [pid for pid, v in zip(pattern_ids, a['fl']) if v > 0]
                X = a['X']
                target[sym] = {
                    'id': f"{sym}|1|{int(F['t'][i] * 1000)}" + ('|moc' if pre else ''), 'sym': sym, 'group': g, 'sector': uni[sym].get('sector', ''),
                    'setup': flagged[0] if flagged else '', 'label': ' · '.join(labels.get(f, f) for f in flagged[:3]), 'patterns': flagged,
                    'dir': 1, 'sigTs': int(F['t'][i] * 1000), 'sigClose': round(close, 4), 'atr': round(atr, 4), 'atrPct': round(atr / close * 100, 2),
                    **({'mode': 'moc'} if pre else {}),
                    'ctx': {k: (None if not np.isfinite(X[0][feats.FEATURES.index(k)]) else round(float(X[0][feats.FEATURES.index(k)]), 2))
                            for k in ('rsi2', 'ibs', 'ret3', 'ret5', 'ret60', 'dist200', 'atrp', 'dd20', 'dd52', 'vol_ratio', 'spy_up', 'spy_dd60', 'vix', 'vix_z', 'b_up200', 'b_os', 'vp_val_atr')},
                    'plans': []}
            target[sym]['plans'].append(plan)

    def finish(d):
        out = []
        for al in d.values():
            pr = next((pl for pl in al['plans'] if pl.get('role') == 'principal'), al['plans'][0])
            al.update({'p': pr['p'], 'ev': pr['ev'], 'exit': pr['exit'], 'levels': pr['levels'], 'hist': pr['hist'], 'strategy': pr['strategy'],
                       'cell': pr['strategy'], 'pMin': 0.0})
            out.append(al)
        out.sort(key=lambda x: (min(pl['rank'] for pl in x['plans']), -x['ev']))
        return out
    main_alerts = finish(by_sym)
    watch = [w for w in finish(watch_syms) if w['sym'] not in by_sym][:40]
    n_over_cap = 0
    alerts = main_alerts

    # ── pelotazos (experimental): las N mejores del día por probabilidad de ganar ≥ +12 % ──
    pelotazos: List[Dict[str, Any]] = []
    if pel and pel_cand:
        rows_ = []
        for a in pel_cand:
            X = feats.event_matrix(a['F'], np.array([a['last']]))
            gid = np.array([feats.GROUPS.index(a['g'])])
            p_raw = float(pel_model.predict_raw(X, a['fl'][None, :], gid)[0])
            if p_raw >= pel['floorRaw']:
                rows_.append((p_raw, float(pel_model.predict(X, a['fl'][None, :], gid)[0]), a, X))
        rows_.sort(key=lambda r: -r[0])
        for rank, (p_raw, p_cal, a, X) in enumerate(rows_[:int(pel.get('nPerDay', 3))]):
            F, i, sym, g = a['F'], a['last'], a['sym'], a['g']
            atr, close = float(F['atr'][i]), float(F['c'][i])
            ex = pel['exit']
            flagged = [pid for pid, v in zip(pel_pattern_ids, a['fl']) if v > 0]
            pelotazos.append({
                'id': f"{sym}|pel|{int(F['t'][i] * 1000)}", 'sym': sym, 'group': g, 'sector': uni[sym].get('sector', ''),
                'label': ' · '.join(pel_labels.get(f, f) for f in flagged[:3]), 'patterns': flagged, 'dir': 1,
                'sigTs': int(F['t'][i] * 1000), 'sigClose': round(close, 4), 'atr': round(atr, 4), 'atrPct': round(atr / close * 100, 2),
                'strategy': pel['id'], 'rank': rank + 1, 'pBig': round(p_cal, 4), 'exit': ex,
                'levels': {'stopPct': round(-ex['S'] * atr / close * 100, 2), 'stop': round(close - ex['S'] * atr, 4)},
                'ctx': {k: (None if not np.isfinite(X[0][feats.FEATURES.index(k)]) else round(float(X[0][feats.FEATURES.index(k)]), 2))
                        for k in ('rsi2', 'ret5', 'ret60', 'ret120', 'rs120', 'dist200', 'dd52', 'vol_ratio', 'gap', 'vix', 'vix_z', 'b_up200', 'b_os')},
                'hist': {k: pel['stats']['oos'].get(k) for k in ('n', 'wr', 'mean', 'median', 'pf', 'p20', 'p30')}})

    # ── seguimiento en vivo ───────────────────────────────────────────────
    prev = jload('tracking.json', args.out)
    moc_keys = {(t['sym'], t['setup'], t['sigTs']) for t in (prev or {}).get('trades', []) if t.get('mode') == 'moc'}
    plan_trades = []
    for al in main_alerts:
        for pl in al['plans']:
            if not pre and (al['sym'], pl['strategy'], al['sigTs']) in moc_keys:
                pl['preAlerted'] = True            # ya avisada antes del cierre: su operación MOC está en el seguimiento
                continue
            plan_trades.append({'id': f"{al['sym']}|{pl['strategy']}|{al['sigTs']}" + ('|moc' if pre else ''), 'sym': al['sym'], 'group': al['group'],
                                'setup': pl['strategy'], 'cell': pl['strategy'], 'dir': 1, 'exit': pl['exit'], 'sigTs': al['sigTs'], 'p': pl['p'],
                                'sigClose': al['sigClose'], **({'mode': 'moc'} if pre else {})})
        al['pre'] = not pre and all(pl.get('preAlerted') for pl in al['plans'])
    for pa in pelotazos:
        plan_trades.append({'id': pa['id'], 'sym': pa['sym'], 'group': pa['group'], 'setup': pa['strategy'], 'cell': pa['strategy'], 'dir': 1,
                            'exit': pa['exit'], 'sigTs': pa['sigTs'], 'p': None, 'sigClose': pa['sigClose']})
    # con barras provisionales NO se resuelven las operaciones existentes (una barra a medias no puede cerrar nada de forma definitiva)
    tracking = TR.update(prev, plan_trades, series_F, cost_of, add_only=pre)
    if not pre:
        for t in tracking['trades']:
            if t['status'] in ('abierta', 'pendiente') and t['exit']['kind'] == 'rsi' and t['sym'] in series_F:
                F = series_F[t['sym']]
                tr_ = exit_trigger(F, len(F['c']) - 1)
                t['exitTrig'] = round(tr_, 4) if tr_ else None
    tracking['generatedAt'] = datetime.now(timezone.utc).isoformat()
    g = tracking['stats']['global']
    print(f"  seguimiento: {g['open']} abiertas · {g['pending']} pendientes · {g['n']} cerradas"
          + (f" · WR {g['wr']}% · μ {g['mean']}%" if g['n'] else ''), flush=True)

    if pre:
        # ── escaneo previo al cierre: solo alerts_pre.json y las operaciones MOC nuevas en tracking.json ──
        last_day = datetime.fromtimestamp(max(last_ts_all), timezone.utc).strftime('%Y-%m-%d') if last_ts_all else None
        now = datetime.now(timezone.utc).isoformat()
        jdump({'generatedAt': now, 'mode': 'moc', 'sigDate': last_day, 'symbols': len(partial_syms), 'candidates': len(cand),
               'alerts': main_alerts, 'candles': {a['sym']: candles_of(series_F[a['sym']]) for a in main_alerts}},
              'alerts_pre.json', args.out)
        jdump(tracking, 'tracking.json', args.out)
        print(f"\nPREVIO AL CIERRE · {last_day} · barras provisionales {len(partial_syms)} · candidatos {len(cand)} · avisos {len(main_alerts)} · {time.time() - t0:.0f}s", flush=True)
        for a in main_alerts[:15]:
            print(f"  {a['sym']:10s} " + ' | '.join(f"{pl['label']}: P={pl['p'] * 100:.0f}% EV={pl['ev']:+.2f}%" for pl in a['plans']) + f"  {a['label'][:60]}", flush=True)
        if not args.no_notify:
            notify.send_pre(main_alerts, {'sigDate': last_day, 'market': {}})
        notify.step_summary_pre(main_alerts, last_day)
        return

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
    if last_ts_all:      # fecha más habitual de la última barra completa (algún activo suelto puede ir un día por delante)
        from collections import Counter
        last_day = Counter(datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in last_ts_all).most_common(1)[0][0]
    else:
        last_day = None
    now = datetime.now(timezone.utc).isoformat()
    meta = {'generatedAt': now, 'dataThrough': last_day, 'symbols': len(series_F), 'failed': len(failed),
            'failedList': failed[:80], 'strategies': len(strategies), 'candidates': len(cand), 'alerts': len(main_alerts),
            'paused': [s['id'] for s in strategies if s.get('paused')] + ([pel['id']] if pel and pel.get('paused') else []), 'watch': len(watch), 'pelotazos': len(pelotazos), 'market': mk, 'oversoldUptrend': n_os, 'rules': rules,
            'validatedAt': reg.get('generatedAt'), 'validatedThrough': reg.get('dataThrough'), 'secs': round(time.time() - t0)}

    jdump({'generatedAt': now, 'dataThrough': last_day, 'alerts': main_alerts, 'watch': watch, 'pelotazos': pelotazos}, 'alerts.json', args.out)
    jdump(tracking, 'tracking.json', args.out)
    jdump(candles, 'candles.json', args.out)
    jdump(meta, 'meta.json', args.out)
    jdump({'generatedAt': reg.get('generatedAt'), 'dataThrough': reg.get('dataThrough'), 'universe': reg.get('universe'),
           'design': reg.get('design'), 'rules': rules,
           'patterns': reg.get('patterns'),
           'strategies': [{k: v for k, v in s.items() if k != 'model'} for s in strategies],
           'pelotazo': ({k: v for k, v in pel.items() if k != 'model'} if pel else None)}, 'registry.json', args.out)

    print(f"\nActivos: {len(series_F)} · fallidos {len(failed)} · candidatos {len(cand)} · alertas {len(main_alerts)} · vigilancia {len(watch)} · {time.time() - t0:.0f}s", flush=True)
    for pa in pelotazos:
        print(f"  🚀 {pa['sym']:10s} P(≥+12%)≈{pa['pBig'] * 100:.0f}%  stop inicial {pa['levels']['stopPct']:+.1f}%  {pa['label'][:60]}", flush=True)
    for a in main_alerts[:15]:
        print(f"  {a['sym']:10s} " + ' | '.join(f"{pl['label']}: P={pl['p'] * 100:.0f}% EV={pl['ev']:+.2f}%" for pl in a['plans']) + f"  {a['label'][:60]}", flush=True)
    if not strategies:
        print('  ⚠ Registro sin planes validados: el sistema no emite alertas.', flush=True)
    for s_ in strategies:
        if s_.get('paused'):
            print(f"  ⏸ Plan «{s_['label']}» en PAUSA: el chequeo de salud (últimos 12 meses) no es rentable.", flush=True)
    if not args.no_notify:
        notify.send(main_alerts, meta, tracking, pelotazos)
    notify.step_summary(main_alerts, watch, meta, tracking, pelotazos)


if __name__ == '__main__':
    main()
