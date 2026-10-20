#!/usr/bin/env python3
"""
stops.py — «acierta mucho pero un stop de −10 % o más se come muchas ganancias» (comando `stops` de research.py; NO interviene en producción)

El plan Equilibrado usa un stop de 4×ATR: con ATR del 3–6 % del precio eso es un −12 % a −24 % de pérdida máxima por operación (más si hay hueco de apertura),
frente a ganancias medias de +3–4 %. Aquí, con la validación SIN sesgo de supervivencia (pertenencia histórica a los índices), se mide:
  A. La distribución real de resultados del plan (cuántas operaciones pierden ≥5/10/15 %, cuánto de la pérdida bruta sale de las peores, tipo de salida).
  B. Las MISMAS alertas con otros stops (1–6×ATR y sin stop): ¿cambia la media, el acierto, la cola de pérdidas?
  C. El sistema reentrenado entero con otros stops (selección, grupos, cartera) para ver si el modelo elige distinto cuando el stop es más corto.
  D. Tamaño por riesgo: en vez de 10 % de la cartera por operación, tamaño = riesgo fijo / distancia al stop (una pérdida por stop cuesta siempre lo mismo).
  E. Tope de volatilidad (ATR %): sin operar valores con stops de 25 %.
  F. Las 15 peores operaciones.
"""
from __future__ import annotations
import math
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import research as RS
import setups as SU
import simulate as SM
import pit
from feats import FEATURES

S_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 99.0)
H = 10


def vname(S: float) -> str:
    return SM.Variant('rsi', 0.0, S, H).name


def plan_of(S: float, n: int = 3) -> Dict:
    return {'id': vname(S), 'role': 'principal', 'label': f'Equilibrado stop {S:g}×ATR', 'n': n, 'blurb': ''}


def metrics(x: np.ndarray, kind: np.ndarray | None = None) -> Dict:
    w, l = x[x > 0], x[x <= 0]
    gl = -l.sum()
    srt = np.sort(x)
    k5, k10 = max(int(len(x) * 0.05), 1), max(int(len(x) * 0.10), 1)
    out = {'n': len(x), 'wr': float((x > 0).mean() * 100), 'mean': float(x.mean()), 'median': float(np.median(x)),
           'pf': float(w.sum() / gl) if gl > 0 else float('inf'), 'aw': float(w.mean()) if len(w) else 0.0, 'al': float(l.mean()) if len(l) else 0.0,
           'p5': float(np.percentile(x, 5)), 'worst': float(x.min()),
           'l5': float((x <= -5).mean() * 100), 'l10': float((x <= -10).mean() * 100), 'l15': float((x <= -15).mean() * 100),
           'sh5': float(-srt[:k5][srt[:k5] < 0].sum() / gl * 100) if gl > 0 else 0.0, 'sh10': float(-srt[:k10][srt[:k10] < 0].sum() / gl * 100) if gl > 0 else 0.0}
    if kind is not None:
        out['stop'] = float((kind == -1).mean() * 100)
    return out


def portfolio_var(cal: np.ndarray, ti: np.ndarray, bars: np.ndarray, res: np.ndarray, rank: np.ndarray, M: int, frac: np.ndarray, max_expo: float = 99.0) -> Dict:
    """Como RS.portfolio_sim pero con una fracción de capital distinta por operación (tamaño por riesgo)."""
    ok = np.isfinite(res) & np.isfinite(frac)
    idx = np.flatnonzero(ok)
    idx = idx[np.lexsort((rank[idx], ti[idx]))]
    by_t: Dict[int, List[int]] = {}
    for k in idx:
        by_t.setdefault(int(ti[k]), []).append(int(k))
    nd = len(cal)
    t0 = int(ti[idx].min())
    eq = 1.0
    open_: List[tuple] = []
    hist = np.ones(nd)
    expo = []
    for t in range(t0, nd):
        still = []
        for ex_t, amt, r in open_:
            if ex_t <= t:
                eq += amt * r / 100.0
            else:
                still.append((ex_t, amt, r))
        open_ = still
        for k in by_t.get(t - 1, []):
            if len(open_) >= M:
                break
            if sum(a for _, a, _ in open_) + eq * float(frac[k]) > eq * max_expo:       # no se apalanca: la exposición total no pasa del límite
                continue
            open_.append((t - 1 + int(max(bars[k], 1)), eq * float(frac[k]), float(res[k])))
        hist[t] = eq
        expo.append(sum(a for _, a, _ in open_) / eq)
    h = hist[t0:]
    peak = np.maximum.accumulate(h)
    rets = np.diff(h) / h[:-1]
    yrs_n = len(h) / 252.0
    mo = np.array([h[min(i + 21, len(h) - 1)] / h[i] - 1 for i in range(0, len(h) - 1, 21)])
    return {'cagr': float(h[-1] ** (1 / yrs_n) - 1) * 100, 'dd': float(((peak - h) / peak).max()) * 100, 'sharpe': float(rets.mean() / (rets.std() + 1e-12) * math.sqrt(252)),
            'worstMonth': float(mo.min() * 100), 'expo': float(np.mean(expo) * 100)}


def run(uni: Dict, data: Dict, args, out_dir: str):
    uni2, data2, row_ok, info = pit.prepare(uni, data, args)
    extra = [SM.Variant('rsi', 0.0, S, H) for S in (1.0, 2.0, 3.0, 5.0, 6.0, 99.0)]
    variants = SM.default_variants() + extra
    EV = RS.build_events(data2, uni2, SU.SETUPS, variants)
    rows, FL, names = RS.union_events(EV, +1)
    k = row_ok(EV, rows)
    rows, FL = rows[k], FL[k]
    last_day = int(EV['day'].max())
    day, sym = EV['day'][rows], EV['sym'][rows]
    X = EV['X'][rows]
    atrp = X[:, FEATURES.index('atrp')]
    gid = EV['grp'][rows]
    print(f'\n######## STOPS · PERTENENCIA HISTÓRICA · {len(rows):,} candidatos · datos hasta {np.datetime64("1970-01-01") + np.timedelta64(last_day, "D")}', flush=True)

    exp: Dict = {}
    base = RS.build_plan(EV, rows, FL, names, plan_of(4.0), last_day, spy=data2['SPY'], export=exp)
    sel = exp['sel']
    n_sel = int(sel.sum())
    print(f'\n  (selección del plan desplegado: {n_sel:,} operaciones, grupos {base["groups"]})', flush=True)

    def col(S, key):
        return EV[key][rows, EV['variants'].index(vname(S))]

    # ── A. distribución del plan desplegado (stop 4×ATR) ─────────────────
    x, kd, bs = col(4.0, 'pnl')[sel], col(4.0, 'kind')[sel], col(4.0, 'bars')[sel]
    m = metrics(x, kd)
    print('\n  A. DISTRIBUCIÓN DE RESULTADOS DEL PLAN (stop 4×ATR, salida RSI(2)>70, 10 sesiones, top-3/día, pertenencia histórica)')
    print(f"    n={m['n']} · acierto {m['wr']:.1f}% · media {m['mean']:+.2f}% · mediana {m['median']:+.2f}% · PF {m['pf']:.2f} · ganancia media {m['aw']:+.2f}% · pérdida media {m['al']:+.2f}%")
    print(f"    percentiles: p1 {np.percentile(x, 1):+.1f}% · p5 {np.percentile(x, 5):+.1f}% · p10 {np.percentile(x, 10):+.1f}% · p25 {np.percentile(x, 25):+.1f}% · p50 {np.percentile(x, 50):+.1f}% · p75 {np.percentile(x, 75):+.1f}% · p90 {np.percentile(x, 90):+.1f}%")
    print(f"    pierden ≥5 %: {m['l5']:.1f}% de las operaciones · ≥10 %: {m['l10']:.1f}% · ≥15 %: {m['l15']:.1f}% · peor: {m['worst']:+.1f}%")
    print(f"    el 5 % peor de las operaciones aporta el {m['sh5']:.0f}% de toda la pérdida bruta; el 10 % peor, el {m['sh10']:.0f}%")
    print(f"    tipo de salida: stop {np.mean(kd == -1) * 100:.1f}% (media {x[kd == -1].mean():+.1f}%) · salida por RSI/tiempo {np.mean(kd != -1) * 100:.1f}% (media {x[kd != -1].mean():+.2f}%)")
    st = x[kd == -1]
    theo = -(4.0 * atrp[sel][kd == -1])
    print(f"    stops: pérdida realizada media {st.mean():+.1f}% frente a la teórica −4×ATR {theo.mean():+.1f}% (el hueco de apertura y el coste la empeoran en {st.mean() - theo.mean():+.1f} pts)")
    print(f"    distancia al stop (4×ATR en %): mediana {np.median(4 * atrp[sel]):.1f}% · p25 {np.percentile(4 * atrp[sel], 25):.1f}% · p75 {np.percentile(4 * atrp[sel], 75):.1f}% · máximo {np.max(4 * atrp[sel]):.0f}%")
    ys = RS.year_of(day[sel])
    print('    por año (n · ≥10 % de pérdida · media): ' + ' · '.join(f"{Y}: {int((ys == Y).sum())}/{(x[ys == Y] <= -10).mean() * 100:.0f}%/{x[ys == Y].mean():+.2f}" for Y in sorted(set(ys))))

    # ── B. mismas alertas, otros stops ───────────────────────────────────
    print('\n  B. LAS MISMAS ALERTAS CON OTRO STOP (salida RSI(2)>70 o 10 sesiones; solo cambia el stop)')
    print('    ' + 'stop'.ljust(10) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'mediana'.rjust(9) + 'PF'.rjust(6) + 'gan.'.rjust(7) + 'pérd.'.rjust(7) + '≥10%'.rjust(7) + '≥15%'.rjust(7) + 'peor'.rjust(8) + '%stop'.rjust(7) + 'dist.stop'.rjust(11))
    for S in S_GRID:
        xs, ks = col(S, 'pnl')[sel], col(S, 'kind')[sel]
        okk = np.isfinite(xs)
        mm = metrics(xs[okk], ks[okk])
        lab = 'sin stop' if S >= 99 else f'{S:g}×ATR'
        dist = 'sin' if S >= 99 else f'{np.median(S * atrp[sel]):.1f}%'
        print('    ' + lab.ljust(10) + f"{mm['n']:6d}{mm['wr']:8.1f}%{mm['mean']:+7.2f}%{mm['median']:+8.2f}%{mm['pf']:6.2f}{mm['aw']:+6.2f}%{mm['al']:+6.2f}%{mm['l10']:6.1f}%{mm['l15']:6.1f}%{mm['worst']:+7.1f}%{mm['stop']:6.1f}%{dist:>11}")

    # ── C. sistema reentrenado con otros stops ───────────────────────────
    print('\n  C. EL SISTEMA ENTERO REENTRENADO CON OTRO STOP (modelo, grupos, política y cartera walk-forward)')
    plans = {}
    for S in (2.0, 2.5, 3.0, 4.0, 5.0):
        if S == 4.0:
            pl = base
        else:
            pl = RS.build_plan(EV, rows, FL, names, plan_of(S), last_day, spy=data2['SPY'])
        plans[S] = pl
    print('    ' + 'stop'.ljust(8) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'PF'.rjust(6) + 'alfa vs S&P'.rjust(13) + '   cartera 10×10 %: anual · caída · Sharpe · peor mes' + '   cartera 20×5 %: anual · caída')
    for S, pl in plans.items():
        o, vs = pl['stats']['oos'], pl['stats'].get('vsSpy', {})
        p10, p20 = pl['stats']['portfolio']['10x10'], pl['stats']['portfolio']['20x5']
        print('    ' + f'{S:g}×ATR'.ljust(8) + f"{o['n']:6.0f}{o['wr']:8.1f}%{o['mean']:+7.2f}%{o['pf']:6.2f}{vs.get('alpha', float('nan')):+8.2f}% (t={vs.get('alphaT', float('nan')):.1f})   "
              f"{p10['cagr']:+6.1f}% · {p10['dd']:4.1f}% · {p10['sharpe']:.2f} · {p10['worstMonth']:+5.1f}%      {p20['cagr']:+6.1f}% · {p20['dd']:4.1f}%   grupos {pl['groups']}")

    # ── D. tamaño por riesgo (stop 4×ATR) ────────────────────────────────
    cal_d = (data2['SPY']['t'] // 86400).astype(np.int64)
    ti = np.searchsorted(cal_d, day, side='left')
    vi4 = EV['variants'].index(vname(4.0))
    bars4 = EV['bars'][rows, vi4]
    pnl4 = EV['pnl'][rows, vi4]
    res = np.where(sel & (day >= RS.day_of(RS.PORT_START)), pnl4, np.nan)
    rk5 = exp['rk5']
    stopfrac = np.clip(4.0 * atrp / 100.0, 0.02, 0.60)
    print('\n  D. TAMAÑO POR RIESGO (stop 4×ATR; 10 posiciones como máximo; cada operación arriesga siempre lo mismo hasta su stop, con tope del 10 % de la cartera)')
    print('    ' + 'reglas'.ljust(46) + 'anual'.rjust(8) + 'caída'.rjust(8) + 'Sharpe'.rjust(8) + 'peor mes'.rjust(10) + 'expuesto'.rjust(10))
    o = portfolio_var(cal_d, ti, bars4, res, rk5, 10, np.full(len(rows), 0.10))
    print('    ' + '10 % fijo por operación (como ahora)'.ljust(46) + f"{o['cagr']:+7.1f}%{o['dd']:7.1f}%{o['sharpe']:8.2f}{o['worstMonth']:+9.1f}%{o['expo']:9.0f}%")
    for r in (0.0050, 0.0075, 0.0100, 0.0150, 0.0200):
        fr = np.minimum(0.10, r / stopfrac)
        o = portfolio_var(cal_d, ti, bars4, res, rk5, 10, fr)
        print('    ' + f'riesgo {r * 100:.2f} % de la cartera por stop (tope 10 %)'.ljust(46) + f"{o['cagr']:+7.1f}%{o['dd']:7.1f}%{o['sharpe']:8.2f}{o['worstMonth']:+9.1f}%{o['expo']:9.0f}%")
    fr = np.minimum(0.10, 0.0100 / stopfrac)
    o20 = portfolio_var(cal_d, ti, bars4, res, rk5, 20, fr)
    print('    ' + 'riesgo 1,00 % por stop, hasta 20 posiciones'.ljust(46) + f"{o20['cagr']:+7.1f}%{o20['dd']:7.1f}%{o20['sharpe']:8.2f}{o20['worstMonth']:+9.1f}%{o20['expo']:9.0f}%")
    fr1 = np.minimum(0.10, 0.0100 / stopfrac)
    wk = sel & np.isfinite(pnl4)
    print(f"    (peor pérdida de UNA operación medida sobre la cartera: con 10 % fijo {np.max(-np.minimum(pnl4[wk], 0) * 0.10):.2f}% · con riesgo 1 % por stop {np.max(-np.minimum(pnl4[wk], 0) * fr1[wk]):.2f}%)")

    # ── E. tope de volatilidad ───────────────────────────────────────────
    print('\n  E. TOPE DE VOLATILIDAD (ATR % del precio) sobre las mismas alertas, stop 4×ATR')
    print('    ' + 'ATR %'.ljust(14) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'PF'.rjust(6) + 'pérd.media'.rjust(12) + '≥10%'.rjust(7) + '≥15%'.rjust(7) + 'peor'.rjust(8))
    for lo, hi in ((2.5, 3.5), (3.5, 5.0), (5.0, 7.0), (7.0, 10.0), (10.0, 99.0)):
        mk = sel & (atrp >= lo) & (atrp < hi) & np.isfinite(pnl4)
        if mk.sum() < 30:
            continue
        mm = metrics(pnl4[mk])
        print('    ' + f'{lo:g}–{hi:g}'.ljust(14) + f"{mm['n']:6d}{mm['wr']:8.1f}%{mm['mean']:+7.2f}%{mm['pf']:6.2f}{mm['al']:+11.2f}%{mm['l10']:6.1f}%{mm['l15']:6.1f}%{mm['worst']:+7.1f}%")
    print('    con tope (todas las alertas con ATR % por debajo del tope):')
    for cap in (5.0, 6.0, 7.0, 8.0, 99.0):
        mk = sel & (atrp < cap) & np.isfinite(pnl4)
        mm = metrics(pnl4[mk])
        print(f"      ATR < {cap:g}%: n={mm['n']} · acierto {mm['wr']:.1f}% · media {mm['mean']:+.2f}% · PF {mm['pf']:.2f} · ≥10 % de pérdida: {mm['l10']:.1f}% · peor {mm['worst']:+.1f}%")

    # ── F. peores operaciones ────────────────────────────────────────────
    print('\n  F. LAS 15 PEORES OPERACIONES (stop 4×ATR)')
    gi = {i: g for i, g in enumerate(RS.GROUPS)}
    idx = np.flatnonzero(sel & np.isfinite(pnl4))
    worst = idx[np.argsort(pnl4[idx])[:15]]
    for j in worst:
        d = np.datetime64('1970-01-01') + np.timedelta64(int(day[j]), 'D')
        print(f"    {EV['sym_names'][sym[j]]:9s} {d} {gi[int(gid[j])]:9s} ATR {atrp[j]:4.1f}% · {int(bars4[j])} sesiones · {'STOP' if EV['kind'][rows[j], vi4] == -1 else 'tiempo/RSI'} · {pnl4[j]:+6.1f}%")
    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════
#  Stops FIJOS en % («como máximo un 5 %») y stop al cierre
# ═════════════════════════════════════════════════════════════════════════
CAPS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)


def run2(uni: Dict, data: Dict, args, out_dir: str):
    uni2, data2, row_ok, info = pit.prepare(uni, data, args)
    extra = [SM.Variant('rsi', 0.0, 99.0, H, c, cs) for c in CAPS for cs in (False, True)]
    variants = SM.default_variants() + extra
    EV = RS.build_events(data2, uni2, SU.SETUPS, variants)
    rows, FL, names = RS.union_events(EV, +1)
    k = row_ok(EV, rows)
    rows, FL = rows[k], FL[k]
    last_day = int(EV['day'].max())
    day, sym = EV['day'][rows], EV['sym'][rows]
    atrp = EV['X'][rows][:, FEATURES.index('atrp')]
    print(f'\n######## STOPS FIJOS EN % · PERTENENCIA HISTÓRICA · {len(rows):,} candidatos · datos hasta {np.datetime64("1970-01-01") + np.timedelta64(last_day, "D")}', flush=True)
    exp: Dict = {}
    base = RS.build_plan(EV, rows, FL, names, plan_of(4.0), last_day, spy=data2['SPY'], export=exp)
    sel = exp['sel']
    rk5 = exp['rk5']
    print(f'\n  (selección del plan desplegado con techo de ATR: {int(sel.sum()):,} operaciones; los candidatos tienen ATR ≥ 2,5 %, así que 4×ATR ≥ 10 % y un tope de 5 % siempre es un stop fijo del 5 %)', flush=True)

    def vn(c, cs):
        return SM.Variant('rsi', 0.0, 99.0, H, c, cs).name

    def col(name, key):
        return EV[key][rows, EV['variants'].index(name)]

    cands = [('4×ATR (actual, ≈ 14 %)', vname(4.0), None, 4.0 * atrp / 100.0)]
    for c in CAPS:
        cands.append((f'fijo {c:g} %', vn(c, False), c, np.full(len(rows), c / 100.0)))
    for c in (3.0, 5.0, 7.0):
        cands.append((f'fijo {c:g} % AL CIERRE', vn(c, True), c, np.full(len(rows), c / 100.0)))

    print('\n  A. LAS MISMAS ALERTAS CON UN STOP FIJO EN % (salida RSI(2)>70 o 10 sesiones). «R» = ganancia media medida en unidades de lo arriesgado (resultado / distancia al stop)')
    print('    ' + 'stop'.ljust(24) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'mediana'.rjust(9) + 'PF'.rjust(6) + 'gan.'.rjust(7) + 'pérd.'.rjust(8) + '%stop'.rjust(7) + 'peor'.rjust(8) + 'R medio'.rjust(9) + 'R pérd.'.rjust(9))
    for lab, name, c, sf in cands:
        x, kd = col(name, 'pnl')[sel], col(name, 'kind')[sel]
        ok = np.isfinite(x)
        mm = metrics(x[ok], kd[ok])
        r = x[ok] / (sf[sel][ok] * 100.0)
        print('    ' + lab.ljust(24) + f"{mm['n']:6d}{mm['wr']:8.1f}%{mm['mean']:+7.2f}%{mm['median']:+8.2f}%{mm['pf']:6.2f}{mm['aw']:+6.2f}%{mm['al']:+7.2f}%{mm['stop']:6.1f}%{mm['worst']:+7.1f}%{np.mean(r):+8.3f}{np.mean(r[x[ok] <= 0]):+9.2f}")

    # ── B. cartera con tamaño por riesgo ─────────────────────────────────
    cal_d = (data2['SPY']['t'] // 86400).astype(np.int64)
    ti = np.searchsorted(cal_d, day, side='left')
    print('\n  B. CARTERA 2019–2026 CON TAMAÑO POR RIESGO (cada operación arriesga lo mismo hasta su stop; sin apalancar: exposición total ≤ 100 %)')
    print('    ' + 'stop'.ljust(24) + 'tamaño'.ljust(20) + 'anual'.rjust(8) + 'caída'.rjust(8) + 'Sharpe'.rjust(8) + 'peor mes'.rjust(10) + 'expuesto'.rjust(10))
    for lab, name, c, sf in cands:
        if lab.startswith('fijo') and c not in (3.0, 5.0, 7.0, 10.0):
            continue
        pnl = col(name, 'pnl')
        bars = col(name, 'bars')
        res = np.where(sel & (day >= RS.day_of(RS.PORT_START)), pnl, np.nan)
        sfc = np.clip(sf, 0.02, 0.60)
        for risk, fmax, M in ((None, 0.10, 10), (0.01, 0.10, 10), (0.01, 0.20, 5)):
            fr = np.full(len(rows), fmax) if risk is None else np.minimum(fmax, risk / sfc)
            o = portfolio_var(cal_d, ti, bars, res, rk5, M, fr, max_expo=1.0)
            tam = '10 % fijo' if risk is None else f'riesgo 1 %, máx {fmax * 100:.0f} %'
            print('    ' + lab.ljust(24) + tam.ljust(20) + f"{o['cagr']:+7.1f}%{o['dd']:7.1f}%{o['sharpe']:8.2f}{o['worstMonth']:+9.1f}%{o['expo']:9.0f}%")

    # ── C. sistema entero reentrenado ────────────────────────────────────
    print('\n  C. EL SISTEMA ENTERO REENTRENADO CON ESE STOP (modelo, grupos, política y cartera 10×10 % walk-forward)')
    print('    ' + 'stop'.ljust(24) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'PF'.rjust(6) + 'alfa vs S&P'.rjust(14) + '   cartera 10×10 %: anual · caída · Sharpe · peor mes')
    for lab, name, c, sf in [cands[0], cands[4], cands[2], cands[6], cands[-2]]:
        pl = base if name == vname(4.0) else RS.build_plan(EV, rows, FL, names, {'id': name, 'role': 'principal', 'label': lab, 'n': 3, 'blurb': ''}, last_day, spy=data2['SPY'])
        o, vs, p10 = pl['stats']['oos'], pl['stats'].get('vsSpy', {}), pl['stats']['portfolio']['10x10']
        print('    ' + lab.ljust(24) + f"{o['n']:6.0f}{o['wr']:8.1f}%{o['mean']:+7.2f}%{o['pf']:6.2f}{vs.get('alpha', float('nan')):+8.2f}% (t={vs.get('alphaT', float('nan')):.1f})   "
              f"{p10['cagr']:+6.1f}% · {p10['dd']:4.1f}% · {p10['sharpe']:.2f} · {p10['worstMonth']:+5.1f}%   grupos {pl['groups']}")
    sys.stdout.flush()
