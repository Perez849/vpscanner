#!/usr/bin/env python3
"""
lab.py — Laboratorio exploratorio (cómo se llegó al diseño final). NO interviene en producción.

Comandos (python research.py <cmd>): explore · meta · meta2 · meta3 · meta4 · meta5 · regime
Cada uno imprime tablas sobre la tabla de eventos del universo completo; las conclusiones que
justifican el diseño están resumidas en README.md.
"""
from __future__ import annotations
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

from research import (FEATURES, GROUPS, SU, SM, TRAIN_END, VAL_END, day_of, year_of, stats, fmt, union_events,
                      topn_mask as _topn_mask, wf_predict_linear, select_policy, PLANS)

# ═════════════════════════════════════════════════════════════════════════
#  EXPLORE
# ═════════════════════════════════════════════════════════════════════════
def explore(EV: Dict, out_dir: str):
    tr, va, te = split_masks(EV['day'])
    dev = tr | va
    setups_ = EV['setups']; variants = EV['variants']
    sid, grp, day, sym = EV['sid'], EV['grp'], EV['day'], EV['sym']
    print(f"\n=== EVENTOS: total {len(sid):,} · train {tr.sum():,} · val {va.sum():,} · test(bloqueado) {te.sum():,}\n", flush=True)
    base_i = {+1: setups_.index('B_up'), -1: setups_.index('B_dn')}
    years = (np.datetime64('1970-01-01') + day.astype('timedelta64[D]')).astype('datetime64[Y]').astype(int) + 1970

    # benchmark de entrada aleatoria por variante y periodo (misma dirección)
    base = {}
    for d_, bi in base_i.items():
        for vi in range(len(variants)):
            p = EV['pnl'][:, vi]
            for pn, pm in (('dev', dev), ('tr', tr), ('va', va)):
                m = (sid == bi) & pm
                base[(d_, vi, pn)] = stats(p[m], day[m], sym[m])
    print('Benchmark (entrada aleatoria sobre/bajo SMA200, DEV) para algunas salidas:')
    for d_ in (+1, -1):
        for vn in ('atr_T0.5_S4_H10', 'atr_T1_S4_H10', 'atr_T1.5_S4_H10', 'atr_T2.5_S4_H10', 'sig_S4_H10'):
            vi = variants.index(vn)
            print(f"  {'LARGO' if d_ > 0 else 'CORTO'} {vn:18s} {fmt(base[(d_, vi, 'dev')])}")

    rows = []
    for si, sname in enumerate(setups_):
        st_ = SU.BY_ID[sname]
        if st_.family == 'baseline':
            continue
        ms = (sid == si) & dev
        if ms.sum() < 100:
            continue
        for vi, vname in enumerate(variants):
            p = EV['pnl'][:, vi]
            st = stats(p[ms], day[ms], sym[ms])
            if st['n'] < 100:
                continue
            bs = base[(st_.dir, vi, 'dev')]
            st_tr = stats(p[ms & tr]); st_va = stats(p[ms & va])
            btr, bva = base[(st_.dir, vi, 'tr')], base[(st_.dir, vi, 'va')]
            rows.append({'setup': sname, 'variant': vname, **st,
                         'edge': st['mean'] - bs['mean'], 'wr_edge': st['wr'] - bs['wr'],
                         't_edge': (st['mean'] - bs['mean']) / math.sqrt(st['se'] ** 2 + bs['se'] ** 2 + 1e-12),
                         'tr_wr': st_tr.get('wr', np.nan), 'tr_mean': st_tr.get('mean', np.nan), 'tr_n': st_tr.get('n', 0),
                         'tr_edge': st_tr.get('mean', np.nan) - btr.get('mean', np.nan),
                         'va_wr': st_va.get('wr', np.nan), 'va_mean': st_va.get('mean', np.nan), 'va_n': st_va.get('n', 0),
                         'va_edge': st_va.get('mean', np.nan) - bva.get('mean', np.nan)})
    json.dump(rows, open(os.path.join(out_dir, 'explore_rows.json'), 'w'), default=float)

    def show(title, sel, key, n=30, rev=True):
        print(f'\n--- {title}')
        print(f"{'setup':14s} {'salida':20s} {'n':>7s} {'WR%':>5s} {'loWR':>5s} {'μ%':>6s} {'edge':>6s} {'t_ed':>5s} {'PF':>5s} | {'trWR':>5s} {'trμ':>6s} {'trEd':>6s} | {'vaWR':>5s} {'vaμ':>6s} {'vaEd':>6s}")
        for r in sorted(sel, key=lambda r: r[key], reverse=rev)[:n]:
            print(f"{r['setup']:14s} {r['variant']:20s} {r['n']:7d} {r['wr']:5.1f} {r['wr_lo']:5.1f} {r['mean']:+6.2f} {r['edge']:+6.2f} {r['t_edge']:5.1f} {r['pf']:5.2f} | {r['tr_wr']:5.1f} {r['tr_mean']:+6.2f} {r['tr_edge']:+6.2f} | {r['va_wr']:5.1f} {r['va_mean']:+6.2f} {r['va_edge']:+6.2f}")

    show('TOP por VENTAJA sobre entrada aleatoria (edge) con n>=300 y t_edge>2.5', [r for r in rows if r['n'] >= 300 and r['t_edge'] > 2.5], 'edge', 30)
    show('TOP por % acierto con μ>0.1% y edge>0.1 y t_edge>2 (n>=300)', [r for r in rows if r['n'] >= 300 and r['mean'] > 0.1 and r['edge'] > 0.1 and r['t_edge'] > 2], 'wr', 30)
    best = []
    for s_ in setups_:
        cand = [r for r in rows if r['setup'] == s_ and r['n'] >= 200]
        if cand:
            best.append(max(cand, key=lambda r: r['edge']))
    show('Mejor salida por setup (por edge, n>=200)', best, 'edge', 45)

    strict = [r for r in rows if r['n'] >= 300 and r['tr_n'] >= 150 and r['va_n'] >= 100 and
              r['tr_wr'] >= 60 and r['va_wr'] >= 60 and r['tr_edge'] > 0.05 and r['va_edge'] > 0.05 and r['tr_mean'] > 0 and r['va_mean'] > 0]
    print(f'\n>>> Celdas con WR>=60%, μ>0 y edge>0.05 en TRAIN **y** en VAL: {len(strict)} de {len(rows)}')
    show('Pasan el criterio dual (orden por edge val)', strict, 'va_edge', 40)

    tops = sorted([r for r in rows if r['n'] >= 300 and r['t_edge'] > 2], key=lambda r: -r['edge'])
    print('\n=== Por grupo de activo (mejor salida por setup con edge significativo, DEV) ===')
    seen = set(); k = 0
    for r in tops:
        if r['setup'] in seen:
            continue
        seen.add(r['setup']); k += 1
        if k > 10:
            break
        si = setups_.index(r['setup']); vi = variants.index(r['variant']); d_ = SU.BY_ID[r['setup']].dir
        print(f"{r['setup']} / {r['variant']}")
        for gi, gname in enumerate(GROUPS):
            m = (sid == si) & dev & (grp == gi)
            st = stats(EV['pnl'][m, vi], day[m], sym[m])
            mb = (sid == base_i[d_]) & dev & (grp == gi)
            sb = stats(EV['pnl'][mb, vi], day[mb], sym[mb])
            if st['n'] >= 30 and sb.get('n', 0) >= 30:
                print(f"    {gname:9s} {fmt(st)}   base μ={sb['mean']:+.2f}% WR={sb['wr']:.0f}%")
    print('\n=== Por año ===')
    seen = set(); k = 0
    for r in tops:
        if r['setup'] in seen:
            continue
        seen.add(r['setup']); k += 1
        if k > 8:
            break
        si = setups_.index(r['setup']); vi = variants.index(r['variant'])
        line = f"{r['setup']:13s}"
        for y in range(2016, 2024):
            m = (sid == si) & (years == y)
            st = stats(EV['pnl'][m, vi])
            line += f" {y}:{st['wr']:4.0f}%/{st['mean']:+5.2f}" if st.get('n', 0) >= 20 else f' {y}:   —     '
        print(line)

    print('\n=== Régimen de mercado ===')
    X = EV['X']; fi = {f: j for j, f in enumerate(FEATURES)}
    for sname in ('L_rsi2_10', 'L_streak3', 'L_low10', 'L_vpval', 'L_pbtrend'):
        if sname not in setups_:
            continue
        cand = [r for r in rows if r['setup'] == sname and r['n'] >= 200]
        if not cand:
            continue
        r = max(cand, key=lambda r: r['edge']); vi = variants.index(r['variant']); si = setups_.index(sname)
        b0 = (sid == si) & dev
        print(f"{sname} / {r['variant']}")
        for lbl, m in (('SPY>SMA200', X[:, fi['spy_up']] == 1), ('SPY<SMA200', X[:, fi['spy_up']] == 0),
                       ('VIX z<0', X[:, fi['vix_z']] < 0), ('VIX z 0-1', (X[:, fi['vix_z']] >= 0) & (X[:, fi['vix_z']] < 1)),
                       ('VIX z>1', X[:, fi['vix_z']] >= 1), ('VIX z>2', X[:, fi['vix_z']] >= 2)):
            st = stats(EV['pnl'][b0 & m, vi], day[b0 & m], sym[b0 & m])
            mb = (sid == base_i[+1]) & dev & m
            sb = stats(EV['pnl'][mb, vi], day[mb], sym[mb])
            print(f"    {lbl:11s} {fmt(st)}   base μ={sb.get('mean', float('nan')):+.2f}% WR={sb.get('wr', float('nan')):.0f}%")

    print('\n=== ¿Aporta el Volume Profile? RSI(2)<15 con/sin precio bajo el VAL del perfil móvil ===')
    if 'L_rsi2_15' in setups_:
        si = setups_.index('L_rsi2_15')
        b0 = (sid == si) & dev
        for vname in ('atr_T1_S4_H10', 'atr_T1.5_S4_H10', 'atr_T2.5_S4_H10', 'sig_S4_H10'):
            vi = variants.index(vname)
            below = b0 & (X[:, fi['vp_val_atr']] < 0)
            above = b0 & (X[:, fi['vp_val_atr']] >= 0)
            far = b0 & (X[:, fi['vp_poc_atr']] > 3)
            print(f"  {vname}\n    bajo VAL     {fmt(stats(EV['pnl'][below, vi], day[below], sym[below]))}"
                  f"\n    dentro/sobre {fmt(stats(EV['pnl'][above, vi], day[above], sym[above]))}"
                  f"\n    POC>3ATR     {fmt(stats(EV['pnl'][far, vi], day[far], sym[far]))}")
    return rows




def wf_predict(EV, rows, FL, y, upto_day: int, first_year: int = 2019, label_ok=None):
    """Predicciones fuera de muestra año a año (entrena con eventos anteriores al año)."""
    import model as MD
    day = EV['day'][rows]; years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    p_oos = np.full(len(rows), np.nan)
    last_year = int(year_of(np.array([upto_day]))[0])
    for Y in range(first_year, last_year + 1):
        trm = (day < day_of(f'{Y}-01-01')) & label_ok
        tem = (years == Y) & (day <= upto_day)
        if trm.sum() < 3000 or tem.sum() == 0:
            continue
        m = MD.LogitModel(fidx, ['f%d' % i for i in range(FL.shape[1])], GROUPS)
        m.fit(X[trm], FL[trm], gid[trm], y[trm])
        p_oos[tem] = m.predict_raw(X[tem], FL[tem], gid[tem])
    return p_oos


def meta_explore(EV: Dict, out_dir: str):
    import model as MD
    variants = EV['variants']
    cut = day_of(VAL_END)
    for sgn in (+1, -1):
        rows, FL, names = union_events(EV, sgn)
        day, sym = EV['day'][rows], EV['sym'][rows]
        dev = day <= cut
        print(f"\n######## {'LARGOS' if sgn > 0 else 'CORTOS'}: {len(rows):,} eventos únicos (DEV {dev.sum():,}) ########", flush=True)
        years = year_of(day)
        for vname in ('atr_T1_S4_H10', 'atr_T1.5_S4_H10', 'sig_S4_H10', 'atr_T0.5_S4_H5'):
            vi = variants.index(vname)
            pnl = EV['pnl'][rows, vi]
            ok = np.isfinite(pnl)
            y = (pnl > 0).astype(np.float32)
            p = wf_predict(EV, rows, FL, y, cut, label_ok=ok)
            m = dev & ok & np.isfinite(p)
            print(f"\n=== {vname} · walk-forward 2019–2023 · {m.sum():,} eventos OOS · AUC={MD.auc(p[m], y[m]):.3f}")
            st0 = stats(pnl[m], day[m], sym[m])
            print(f"   sin modelo (todos los candidatos): {fmt(st0)}")
            print(f"   {'corte':>12s} {'n':>7s} {'WR%':>6s} {'loWR':>6s} {'μ%':>7s} {'PF':>5s} {'t':>5s} {'avgW':>6s} {'avgL':>6s}")
            for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
                thr = np.quantile(p[m], q)
                mm = m & (p >= thr)
                st = stats(pnl[mm], day[mm], sym[mm])
                print(f"   top {100 * (1 - q):4.0f}% p≥{thr:.2f} {st['n']:7d} {st['wr']:6.1f} {st['wr_lo']:6.1f} {st['mean']:+7.2f} {st['pf']:5.2f} {st['t_day']:5.1f} {st['avg_win']:+6.2f} {st['avg_loss']:+6.2f}")
            for thr in (0.70, 0.75, 0.80, 0.85):
                mm = m & (p >= thr)
                if mm.sum() >= 50:
                    st = stats(pnl[mm], day[mm], sym[mm])
                    print(f"   p≥{thr:.2f}       {st['n']:7d} {st['wr']:6.1f} {st['wr_lo']:6.1f} {st['mean']:+7.2f} {st['pf']:5.2f} {st['t_day']:5.1f} {st['avg_win']:+6.2f} {st['avg_loss']:+6.2f}")
            # por año para el top 10%
            thr = np.quantile(p[m], 0.9)
            line = '   top10% por año: '
            for Y in range(2019, 2024):
                mm = m & (p >= thr) & (years == Y)
                st = stats(pnl[mm])
                line += f" {Y}:{st['wr']:.0f}%/{st['mean']:+.2f}%(n={st['n']})" if st.get('n', 0) >= 10 else f' {Y}:—'
            print(line)
            # calibración
            cal = '   calibración (p→real): '
            for a, b in ((0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)):
                mm = m & (p >= a) & (p < b)
                if mm.sum() >= 50:
                    cal += f" [{a:.1f},{b:.1f}) n={mm.sum()} real={y[mm].mean() * 100:.0f}%;"
            print(cal, flush=True)
        # importancia de variables (último modelo con todos los datos DEV) para el 1er variant
        vi = variants.index('atr_T1.5_S4_H10')
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl) & dev
        y = (pnl > 0).astype(np.float32)
        fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
        mm = MD.LogitModel(fidx, names, GROUPS).fit(EV['X'][rows][ok], FL[ok], EV['grp'][rows][ok], y[ok])
        off = 0; imp = []
        for k, f in enumerate(MD.MODEL_FEATURES):
            nb = len(mm.edges[k]) + 2
            wb = mm.w[off:off + nb]; off += nb
            imp.append((f, float(wb.max() - wb.min()), [round(float(x), 2) for x in wb]))
        print('\n   Variables más influyentes (rango de coeficientes por tramo, 5 tramos por cuantiles + NaN):')
        for f, r, wb in sorted(imp, key=lambda t: -t[1])[:14]:
            print(f'     {f:12s} rango={r:.2f}  {wb}')
        print('   banderas de patrón:', {n: round(float(w), 2) for n, w in zip(names, mm.w[off:off + len(names)])})
        print('   grupos:', {g: round(float(w), 2) for g, w in zip(GROUPS, mm.w[off + len(names):off + len(names) + len(GROUPS)])}, flush=True)




def _top_table(tag, score, pnl, day, sym, years, m, qs=(0.9, 0.95, 0.98)):
    out = []
    for q in qs:
        thr = np.quantile(score[m], q)
        mm = m & (score >= thr)
        st = stats(pnl[mm], day[mm], sym[mm])
        line = ''
        for Y in range(2019, 2024):
            my = mm & (years == Y)
            sy = stats(pnl[my])
            line += f" {Y}:{sy['wr']:.0f}%/{sy['mean']:+.2f}" if sy.get('n', 0) >= 20 else f' {Y}:—'
        print(f"   {tag:7s} top{100 * (1 - q):3.0f}% n={st['n']:6d} WR={st['wr']:5.1f}% (lo {st['wr_lo']:4.1f}) μ={st['mean']:+5.2f}% PF={st['pf']:4.2f} t={st['t_day']:4.1f} |{line}")


def meta2_explore(EV: Dict, out_dir: str):
    """Compara modelos (logístico / gradient boosting / ridge de expectativa) en walk-forward, solo largos y solo DEV."""
    import model as MD
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception:
        HistGradientBoostingClassifier = None
    variants = EV['variants']
    cut = day_of(VAL_END)
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    Xm = X[:, fidx]
    print(f'\n######## META2 largos: {len(rows):,} eventos (DEV {(day <= cut).sum():,}) · sklearn={"sí" if HistGradientBoostingClassifier else "no"}', flush=True)
    for vname in ('sig_S4_H10', 'rsi_S4_H10', 'ph_S4_H10', 'ph_S4_H5', 'atr_T1.5_S4_H10'):
        vi = variants.index(vname)
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        preds = {k: np.full(len(rows), np.nan) for k in ('logit', 'ridge', 'gbm')}
        for Y in range(2019, 2024):
            trm = (day < day_of(f'{Y}-01-01')) & ok
            tem = (years == Y) & ok
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LogitModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
            preds['logit'][tem] = m1.predict_raw(X[tem], FL[tem], gid[tem])
            m2 = MD.RidgeModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], pnl[trm])
            preds['ridge'][tem] = m2.predict_raw(X[tem], FL[tem], gid[tem])
            if HistGradientBoostingClassifier is not None:
                Z = np.concatenate([Xm, FL, gid[:, None].astype(np.float32)], 1)
                good = [j for j in range(Z.shape[1]) if np.isfinite(Z[trm][:, j]).sum() > 100]
                Z = Z[:, good]
                cat = np.zeros(Z.shape[1], bool); cat[-1] = True
                g = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=300,
                                                   l2_regularization=5.0, categorical_features=cat, random_state=0)
                g.fit(Z[trm], y[trm])
                preds['gbm'][tem] = g.predict_proba(Z[tem])[:, 1]
        m = ok & (day <= cut) & np.isfinite(preds['logit'])
        print(f"\n=== {vname} · {m.sum():,} eventos OOS (2019–2023) · sin modelo: {fmt(stats(pnl[m], day[m], sym[m]))}")
        for k in ('logit', 'ridge', 'gbm'):
            if not np.isfinite(preds[k][m]).all():
                continue
            auc_ = MD.auc(preds[k][m], y[m]) if k != 'ridge' else MD.auc(preds[k][m], y[m])
            print(f"  [{k}] AUC(win)={auc_:.3f}")
            _top_table(k, preds[k], pnl, day, sym, years, m)
        # mezcla de rangos logit+ridge(+gbm)
        def rk(a):
            r = np.full(len(a), np.nan); idx = np.flatnonzero(m); r[idx] = np.argsort(np.argsort(a[idx])) / len(idx); return r
        ens = rk(preds['logit']) + rk(preds['ridge']) + (rk(preds['gbm']) if np.isfinite(preds['gbm'][m]).all() else 0)
        print('  [mezcla de rangos]')
        _top_table('ens', ens, pnl, day, sym, years, m)
        sys.stdout.flush()



def meta3_explore(EV: Dict, out_dir: str):
    """Ablaciones y reparto por grupo con el modelo logístico (solo largos, solo DEV, walk-forward)."""
    import model as MD
    variants = EV['variants']
    cut = day_of(VAL_END)
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    allf = MD.MODEL_FEATURES
    def fi(fs): return [FEATURES.index(f) for f in fs]
    configs = {
        'base (l2=30, 5 tramos)': dict(fs=allf, l2=30.0, nb=5),
        'l2=10': dict(fs=allf, l2=10.0, nb=5),
        'l2=100': dict(fs=allf, l2=100.0, nb=5),
        '8 tramos': dict(fs=allf, l2=30.0, nb=8),
        'SIN Volume Profile': dict(fs=[f for f in allf if not f.startswith('vp_')], l2=30.0, nb=5),
        'SIN amplitud mercado': dict(fs=[f for f in allf if not f.startswith('b_')], l2=30.0, nb=5),
        'SIN VIX/SPY': dict(fs=[f for f in allf if not (f.startswith('vix') or f.startswith('spy'))], l2=30.0, nb=5),
        'SIN choque noticias': dict(fs=[f for f in allf if f not in ('max_gap5', 'shock5', 'max_vr5', 'rel5')], l2=30.0, nb=5),
    }
    print(f'\n######## META3 largos: {len(rows):,} eventos', flush=True)
    for vname in ('rsi_S4_H10', 'rsi_S2.5_H10', 'rsi_S1.5_H10', 'rsi_S4_H5'):
        vi = variants.index(vname)
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        print(f"\n=== {vname} · sin modelo: {fmt(stats(pnl[ok & (day <= cut)], day[ok & (day <= cut)], sym[ok & (day <= cut)]))}")
        keep_p = None
        for cname, cfg in configs.items():
            if vname != 'rsi_S4_H10' and cname != 'base (l2=30, 5 tramos)':
                continue
            idxf = fi(cfg['fs'])
            p = np.full(len(rows), np.nan)
            for Y in range(2019, 2024):
                trm = (day < day_of(f'{Y}-01-01')) & ok
                tem = (years == Y) & ok
                if trm.sum() < 3000 or not tem.any():
                    continue
                m1 = MD.LogitModel(idxf, names, GROUPS, nb=cfg['nb'], l2=cfg['l2']).fit(X[trm], FL[trm], gid[trm], y[trm])
                p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem])
            m = ok & (day <= cut) & np.isfinite(p)
            print(f"  [{cname}] AUC={MD.auc(p[m], y[m]):.3f}")
            _top_table('  ', p, pnl, day, sym, years, m)
            if cname.startswith('base'):
                keep_p = p
        if keep_p is not None and vname in ('rsi_S4_H10', 'rsi_S2.5_H10'):
            m = ok & (day <= cut) & np.isfinite(keep_p)
            thr = np.quantile(keep_p[m], 0.95)
            top = m & (keep_p >= thr)
            print(f'  -- reparto del top 5% (p≥{thr:.3f}) por grupo de activo:')
            for gi, gname in enumerate(GROUPS):
                mm = top & (EV['grp'][rows] == gi)
                if mm.sum() >= 30:
                    print(f"     {gname:9s} {fmt(stats(pnl[mm], day[mm], sym[mm]))}")
            nd = len(np.unique(day[top]))
            per_day = np.bincount(np.unique(day[top], return_inverse=True)[1])
            print(f'  -- alertas/día del top 5%: media {top.sum() / max(1, len(np.unique(day[m]))):.2f} (sobre todos los días) · días con alerta {nd}/{len(np.unique(day[m]))} · máx {per_day.max()} · p90 {np.percentile(per_day, 90):.0f}')
            print(f"  -- duración media: {np.nanmean(EV['bars'][rows, vi][top]):.1f} sesiones")
            # coste de Volume Profile: top5% con VP bajo VAL vs no
            fvp = FEATURES.index('vp_val_atr')
            for lbl, mm in (('precio bajo VAL', top & (X[:, fvp] < 0)), ('dentro/sobre VAL', top & (X[:, fvp] >= 0))):
                if mm.sum() >= 30:
                    print(f"     top5% {lbl:17s} {fmt(stats(pnl[mm], day[mm], sym[mm]))}")
        sys.stdout.flush()



def regime_explore(EV: Dict, out_dir: str):
    """Descriptivo sobre TODOS los años: ¿qué relaciones con el contexto son estables entre años?"""
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fi = {f: j for j, f in enumerate(FEATURES)}
    ys = list(range(2017, 2027))
    print(f'\n######## REGIMEN (todos los años, {len(rows):,} candidatos largos)', flush=True)
    for vname in ('rsi_S2.5_H10', 'sig_S4_H10'):
        vi = variants.index(vname)
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        print(f'\n=== {vname}: todos los candidatos por año (WR% / μ% / n / días distintos)')
        line = ''
        for Y in ys:
            m = ok & (years == Y)
            if m.sum() >= 30:
                st = stats(pnl[m]); line += f" {Y}:{st['wr']:.0f}/{st['mean']:+.2f}/{st['n']}/{len(np.unique(day[m]))}"
        print(line)
        specs = {
            'vix': [0, 13, 16, 20, 25, 35, 999], 'vix_z': [-9, -1, 0, 1, 2, 99], 'b_up200': [0, 30, 50, 65, 80, 101],
            'b_os': [0, 2, 5, 10, 20, 101], 'spy_dist200': [-99, 0, 3, 6, 10, 99], 'spy_dd60': [-99, -10, -5, -2, -0.01, 1],
            'b_ret5': [-99, -4, -2, 0, 2, 99], 'dd52': [-99, -30, -20, -10, -5, 1], 'rsi2': [0, 2, 5, 10, 15, 30],
            'atrp': [0, 1.5, 2.5, 4, 6, 99], 'dist200': [-99, 0, 5, 10, 20, 999], 'ret5': [-99, -10, -6, -3, 0, 99],
            'rs60': [-999, -10, 0, 10, 25, 999], 'ret120': [-999, 0, 15, 30, 60, 9999], 'vol_ratio': [0, 0.7, 1, 1.5, 2.5, 999],
        }
        for f, edges in specs.items():
            col = X[:, fi[f]]
            print(f'  -- {f} (WR/μ/n por año)')
            for a, b in zip(edges[:-1], edges[1:]):
                mb = ok & np.isfinite(col) & (col >= a) & (col < b)
                if mb.sum() < 200:
                    continue
                st = stats(pnl[mb], day[mb], sym[mb])
                line = f"     [{a:>6g},{b:>6g}) n={st['n']:6d} WR={st['wr']:4.1f} μ={st['mean']:+5.2f} |"
                npos = 0; ntot = 0
                for Y in ys:
                    my = mb & (years == Y)
                    if my.sum() >= 30:
                        sy = stats(pnl[my]); line += f" {str(Y)[2:]}:{sy['wr']:.0f}/{sy['mean']:+.1f}/{sy['n']}"
                        ntot += 1; npos += sy['mean'] > 0
                    else:
                        line += f' {str(Y)[2:]}:—'
                print(line + f"  años+ {npos}/{ntot}")
        # capitulaciones de mercado: días con mucha sobreventa generalizada
        print(f'  -- días de capitulación (b_os>=10%) por año')
        bos = X[:, fi['b_os']]
        for Y in ys:
            m = ok & (years == Y) & np.isfinite(bos) & (bos >= 10)
            if m.sum() >= 20:
                st = stats(pnl[m]); print(f"     {Y}: n={st['n']} días={len(np.unique(day[m]))} WR={st['wr']:.0f}% μ={st['mean']:+.2f}%")
        sys.stdout.flush()




def meta4_explore(EV: Dict, out_dir: str):
    """Modelos de menor capacidad, ventana móvil y política 'N mejores por día'. Todos los años 2019–2026 (walk-forward)."""
    import model as MD
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    last_day = int(day.max())
    print(f'\n######## META4 largos: {len(rows):,} candidatos · todos los años (walk-forward, refit anual)', flush=True)
    yrs = list(range(2019, int(year_of(np.array([last_day]))[0]) + 1))
    for vname in ('rsi_S2.5_H10', 'rsi_S4_H10'):
        vi = variants.index(vname)
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        print(f"\n=== {vname} · todos los candidatos: " + ' '.join(f"{Y}:{stats(pnl[ok & (years == Y)])['wr']:.0f}/{stats(pnl[ok & (years == Y)])['mean']:+.2f}" for Y in yrs if (ok & (years == Y)).sum() > 30), flush=True)
        for mname, cls, win in (('binned·expansiva', MD.LogitModel, None), ('binned·3 años', MD.LogitModel, 3),
                                ('lineal·expansiva', MD.LinearLogitModel, None), ('lineal·3 años', MD.LinearLogitModel, 3),
                                ('lineal·2 años', MD.LinearLogitModel, 2)):
            p = np.full(len(rows), np.nan)
            for Y in yrs:
                lo = day_of(f'{Y - win}-01-01') if win else -10 ** 9
                trm = (day < day_of(f'{Y}-01-01')) & (day >= lo) & ok
                tem = (years == Y) & ok
                if trm.sum() < 3000 or not tem.any():
                    continue
                m1 = cls(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
                p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem])
            m = ok & np.isfinite(p)
            if m.sum() < 500:
                print(f'  [{mname}] (pocos datos)'); continue
            print(f"  [{mname}] AUC={MD.auc(p[m], y[m]):.3f}", flush=True)
            for pol, sel in (('top 5% global', m & (p >= np.quantile(p[m], 0.95))), ('top 10% global', m & (p >= np.quantile(p[m], 0.90))),
                             ('top 3/día', _topn_mask(p, day, m & (p >= np.quantile(p[m], 0.5)), 3)),
                             ('top 5/día', _topn_mask(p, day, m & (p >= np.quantile(p[m], 0.5)), 5)),
                             ('top 10/día', _topn_mask(p, day, m & (p >= np.quantile(p[m], 0.5)), 10))):
                st = stats(pnl[sel], day[sel], sym[sel])
                line = f"     {pol:15s} n={st['n']:6d} WR={st['wr']:4.1f} μ={st['mean']:+5.2f} PF={st['pf']:4.2f} t={st['t_day']:4.1f} |"
                for Y in yrs:
                    my = sel & (years == Y)
                    if my.sum() >= 20:
                        sy = stats(pnl[my]); line += f" {str(Y)[2:]}:{sy['wr']:.0f}/{sy['mean']:+.1f}/{sy['n']}"
                    else:
                        line += f' {str(Y)[2:]}:—'
                rec = sel & (years >= 2024)
                if rec.sum() >= 20:
                    sr = stats(pnl[rec]); line += f" || 24+: {sr['wr']:.0f}%/{sr['mean']:+.2f}% n={sr['n']}"
                print(line, flush=True)



def meta5_explore(EV: Dict, out_dir: str):
    """Perfiles de salida con el modelo lineal de ventana móvil: ¿hay un perfil de alta probabilidad estable? ¿Están calibradas las P?"""
    import model as MD
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    vz = X[:, FEATURES.index('vix_z')]
    yrs = list(range(2019, int(year_of(np.array([int(day.max())]))[0]) + 1))
    print(f'\n######## META5 largos · modelo lineal con ventana de 3 años · refit anual', flush=True)
    for vname in ('rsi_S4_H10', 'rsi_S2.5_H10', 'atr_T0.5_S4_H5', 'atr_T0.5_S4_H10', 'atr_T1_S4_H10', 'ph_S4_H10', 'sig_S4_H10'):
        vi = variants.index(vname)
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        p = np.full(len(rows), np.nan)
        for Y in yrs:
            trm = (day < day_of(f'{Y}-01-01')) & (day >= day_of(f'{Y - 3}-01-01')) & ok
            tem = (years == Y) & ok
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LinearLogitModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
            p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem])
        m = ok & np.isfinite(p)
        print(f"\n=== {vname} · AUC={MD.auc(p[m], y[m]):.3f} · todos los candidatos: " + ' '.join(f"{str(Y)[2:]}:{stats(pnl[m & (years == Y)])['wr']:.0f}/{stats(pnl[m & (years == Y)])['mean']:+.2f}" for Y in yrs if (m & (years == Y)).sum() > 30), flush=True)
        # calibración isotónica con 2019-2023 aplicada a 2024+
        cal = MD.LogitModel(fidx, names, GROUPS)
        mc = m & (years <= 2023)
        cal.set_calibration(p[mc], y[mc])
        xs = np.array([c[0] for c in cal.calib]); ys_ = np.array([c[1] for c in cal.calib])
        pc = np.interp(p, xs, ys_) if len(xs) >= 2 else p
        floor = p >= np.quantile(p[m], 0.5)
        pols = {'top 3/día': _topn_mask(p, day, m & floor, 3), 'top 5/día': _topn_mask(p, day, m & floor, 5),
                'top 5/día & VIXz<1': _topn_mask(p, day, m & floor & (vz < 1), 5),
                'top 5/día & P≥0.70': _topn_mask(p, day, m & floor & (pc >= 0.70), 5)}
        for pol, sel in pols.items():
            st = stats(pnl[sel], day[sel], sym[sel])
            if st.get('n', 0) < 50:
                print(f"     {pol:20s} (sin datos)"); continue
            line = f"     {pol:20s} n={st['n']:6d} WR={st['wr']:4.1f} μ={st['mean']:+5.2f} PF={st['pf']:4.2f} t={st['t_day']:4.1f} aW={st['avg_win']:+.2f} aL={st['avg_loss']:+.2f} |"
            npos = 0; nt = 0
            for Y in yrs:
                my = sel & (years == Y)
                if my.sum() >= 20:
                    sy = stats(pnl[my]); line += f" {str(Y)[2:]}:{sy['wr']:.0f}/{sy['mean']:+.1f}/{sy['n']}"; nt += 1; npos += sy['mean'] > 0
                else:
                    line += f' {str(Y)[2:]}:—'
            print(line + f"  años+ {npos}/{nt}", flush=True)
        # calibración en 2024+ de las seleccionadas top 5/día
        sel = pols['top 5/día'] & (years >= 2024)
        if sel.sum() >= 100:
            print(f"     calibración 2024+ (top 5/día, n={sel.sum()}): P media calibrada={pc[sel].mean() * 100:.1f}% · acierto real={y[sel].mean() * 100:.1f}%")
            for a_, b_ in ((0, .6), (.6, .65), (.65, .7), (.7, 1.01)):
                mm = sel & (pc >= a_) & (pc < b_)
                if mm.sum() >= 40:
                    print(f"        P∈[{a_:.2f},{b_:.2f}): n={mm.sum()} real={y[mm].mean() * 100:.1f}% μ={np.nanmean(pnl[mm]):+.2f}%")




def gap_explore(EV: Dict, out_dir: str):
    """¿Importa el hueco de apertura? ¿Mejora entrar AL CIERRE (escaneo antes del cierre) frente a la apertura siguiente?"""
    reg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'validated.json')))
    groups_of = {s['id']: s['groups'] for s in reg['strategies']}
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    gid = EV['grp'][rows]
    gap = EV['gapin'][rows]
    moc = EV['moc'][rows]
    last_day = int(day.max())
    yrs = sorted(set(years[years >= 2019]))
    print(f'\n######## GAP · {len(rows):,} candidatos largos · política top {PLANS and 5}/día de cada plan', flush=True)
    print('  hueco = (apertura de mañana − cierre de hoy) / ATR  (positivo = abre por encima del cierre)')
    allg = gap[np.isfinite(gap)]
    print(f"  todos los candidatos: mediana {np.median(allg):+.2f} ATR · p10 {np.percentile(allg, 10):+.2f} · p90 {np.percentile(allg, 90):+.2f}")
    for pi, plan in enumerate(PLANS):
        vi = variants.index(plan['id'])
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        p, base = wf_predict_linear(EV, rows, FL, y, ok, last_day)
        allowed = [GROUPS.index(g) for g in groups_of.get(plan['id'], GROUPS)]
        sel = select_policy(p, base, day, gid, ok, allowed) & np.isfinite(gap)
        pm = moc[:, pi]
        print(f"\n=== {plan['id']} · {plan['label']} · seleccionadas {sel.sum():,}: {fmt(stats(pnl[sel], day[sel], sym[sel]))}")
        gs = gap[sel]
        print(f"  hueco de las seleccionadas: mediana {np.median(gs):+.2f} ATR · >+0.25: {(gs > .25).mean() * 100:.0f}% · >+0.5: {(gs > .5).mean() * 100:.0f}% · >+1: {(gs > 1).mean() * 100:.0f}% · <−0.5: {(gs < -.5).mean() * 100:.0f}%")
        print('  A) resultado por tramo de hueco (entrando en la APERTURA):')
        edges = [-99, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 99]
        for a_, b_ in zip(edges[:-1], edges[1:]):
            m = sel & (gap >= a_) & (gap < b_)
            if m.sum() < 60:
                continue
            st = stats(pnl[m], day[m], sym[m])
            npos = nt = 0
            for Y in yrs:
                my = m & (years == Y)
                if my.sum() >= 20:
                    nt += 1; npos += np.nanmean(pnl[my]) > 0
            print(f"     hueco [{a_:>5g},{b_:>5g}) n={st['n']:5d} WR={st['wr']:4.1f}% μ={st['mean']:+5.2f}% PF={st['pf']:4.2f} aW={st['avg_win']:+.2f} aL={st['avg_loss']:+.2f}  años+ {npos}/{nt}")
        print('  B) regla "no entrar si abre más de X ATR por encima del cierre" (se salta esa señal):')
        base_st = stats(pnl[sel], day[sel], sym[sel])
        print(f"     sin regla           n={base_st['n']:5d} WR={base_st['wr']:4.1f}% μ={base_st['mean']:+5.2f}% PF={base_st['pf']:4.2f}")
        for g_ in (0.15, 0.25, 0.5, 0.75, 1.0):
            keep = sel & (gap <= g_)
            drop = sel & (gap > g_)
            sk = stats(pnl[keep], day[keep], sym[keep]); sd = stats(pnl[drop], day[drop], sym[drop])
            npos = nt = 0
            for Y in yrs:
                my = keep & (years == Y)
                if my.sum() >= 20:
                    nt += 1; npos += np.nanmean(pnl[my]) > 0
            print(f"     salta si > +{g_:4.2f} ATR n={sk['n']:5d} WR={sk['wr']:4.1f}% μ={sk['mean']:+5.2f}% PF={sk['pf']:4.2f} años+ {npos}/{nt}  | saltadas n={sd.get('n', 0)} μ={sd.get('mean', float('nan')):+.2f}%")
        okm = sel & np.isfinite(pm)
        so, sm = stats(pnl[okm], day[okm], sym[okm]), stats(pm[okm], day[okm], sym[okm])
        print(f"  C) MISMAS señales entrando en la apertura siguiente vs AL CIERRE (orden MOC) · n={okm.sum()}")
        print(f"     apertura: WR={so['wr']:4.1f}% μ={so['mean']:+5.2f}% PF={so['pf']:4.2f}")
        print(f"     cierre  : WR={sm['wr']:4.1f}% μ={sm['mean']:+5.2f}% PF={sm['pf']:4.2f} t={sm['t_day']:.1f}")
        line_o = line_m = '     por año  apertura:'
        line_o = '     por año · apertura: ' + ' '.join(f"{str(Y)[2:]}:{np.nanmean(pnl[okm & (years == Y)]):+.2f}" for Y in yrs if (okm & (years == Y)).sum() >= 20)
        line_m = '     por año · cierre  : ' + ' '.join(f"{str(Y)[2:]}:{np.nanmean(pm[okm & (years == Y)]):+.2f}" for Y in yrs if (okm & (years == Y)).sum() >= 20)
        print(line_o); print(line_m, flush=True)
        # sensibilidad: el cierre "provisional" (20 min antes) difiere del final; si el precio de la señal se mueve ε, cuántas señales cambian
        okc = okm
        d = pm[okc] - pnl[okc]
        print(f"     ganancia media de comprar al cierre en vez de a la apertura: {np.nanmean(d):+.2f}% por operación")


# ═════════════════════════════════════════════════════════════════════════
#  PELOTAZOS
# ═════════════════════════════════════════════════════════════════════════
PEL_VARIANTS = [SM.Variant('trl', 2.5, 2.5, 20), SM.Variant('trl', 3.5, 3.5, 40), SM.Variant('trl', 5.0, 5.0, 60),
                SM.Variant('atr', 4.0, 2.0, 20), SM.Variant('atr', 6.0, 2.5, 40)]


def _tail_stats(x: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return '—'
    gp, gl = x[x > 0].sum(), -x[x < 0].sum()
    return (f"n={len(x):6d} WR={np.mean(x > 0) * 100:4.1f} μ={x.mean():+6.2f} med={np.median(x):+5.2f} PF={(gp / gl if gl > 0 else 99):4.2f} "
            f"P≥10%={np.mean(x >= 10) * 100:4.1f} P≥20%={np.mean(x >= 20) * 100:4.1f} P≥30%={np.mean(x >= 30) * 100:4.1f} p95={np.percentile(x, 95):+6.1f}")


def pelotazo_explore(EV: Dict, out_dir: str):
    import model as MD
    setups_, variants = EV['setups'], EV['variants']
    sid, day, sym, grp = EV['sid'], EV['day'], EV['sym'], EV['grp']
    years = year_of(day)
    yrs = list(range(2017, 2027))
    base_i = setups_.index('B_up')
    print(f'\n######## PELOTAZOS · {len(sid):,} eventos · variantes {variants}', flush=True)
    for vi, vname in enumerate(variants):
        pnl = EV['pnl'][:, vi]
        print(f'\n=== salida {vname} ===')
        bm = (sid == base_i)
        st_b = stats(pnl[bm], day[bm], sym[bm])
        print(f"  {'BENCHMARK entrada aleatoria':32s} {_tail_stats(pnl[bm])}")
        for si, sname in enumerate(setups_):
            if SU.BY_ID[sname].family != 'pelotazo':
                continue
            m = (sid == si) & np.isfinite(pnl)
            if m.sum() < 150:
                continue
            st = stats(pnl[m], day[m], sym[m])
            t_edge = (st['mean'] - st_b['mean']) / math.sqrt(st['se'] ** 2 + st_b['se'] ** 2 + 1e-12)
            npos = nt = 0
            for Y in yrs:
                my = m & (years == Y)
                if my.sum() >= 20:
                    nt += 1; npos += np.nanmean(pnl[my]) > 0
            print(f"  {sname:12s} {_tail_stats(pnl[m])} | edge={st['mean'] - st_b['mean']:+.2f} t={t_edge:4.1f} años+ {npos}/{nt}")
    # ¿puede un modelo separar los pelotazos? walk-forward con ventana de 3 años; etiqueta: ganar ≥ +12 %
    rows, FL, names = union_events(EV, +1)
    dayr, symr = day[rows], sym[rows]
    yearsr = year_of(dayr)
    X = EV['X'][rows]; gid = grp[rows]
    feats_ = list(MD.MODEL_FEATURES) + ['ret60', 'ret120', 'rs60', 'rs120', 'max_gap5', 'shock5', 'max_vr5']
    fidx = [FEATURES.index(f) for f in feats_]
    last_day = int(dayr.max())
    print(f'\n######## MODELO sobre la unión de setups ({len(rows):,} eventos): ¿se pueden ordenar los candidatos? (walk-forward, ventana 3 años)', flush=True)
    for vi, vname in enumerate(variants):
        if vname not in ('trl_T3.5_S3.5_H40', 'trl_T5_S5_H60', 'atr_T6_S2.5_H40', 'trl_T2.5_S2.5_H20'):
            continue
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        y = (pnl >= 12).astype(np.float32)
        p = np.full(len(rows), np.nan)
        for Y in range(2019, int(year_of(np.array([last_day]))[0]) + 1):
            trm = (dayr < day_of(f'{Y}-01-01')) & (dayr >= day_of(f'{Y - 3}-01-01')) & ok
            tem = (yearsr == Y) & ok
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LinearLogitModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
            p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem])
        m = ok & np.isfinite(p)
        print(f"\n=== {vname} · etiqueta P(≥+12%) · AUC={MD.auc(p[m], y[m]):.3f} · base rate={y[m].mean() * 100:.1f}%")
        print(f"  todos los candidatos   {_tail_stats(pnl[m])}")
        for lbl, sel in (('top 1/día', _topn_mask(p, dayr, m, 1)), ('top 3/día', _topn_mask(p, dayr, m, 3)), ('top 5%', m & (p >= np.quantile(p[m], 0.95))), ('top 1%', m & (p >= np.quantile(p[m], 0.99)))):
            line = f"  {lbl:20s} {_tail_stats(pnl[sel])}"
            print(line)
            yl = '      por año: '
            for Y in range(2019, 2027):
                my = sel & (yearsr == Y)
                if my.sum() >= 20:
                    yl += f" {str(Y)[2:]}:{np.nanmean(pnl[my]):+.1f}/{np.mean(pnl[my] >= 12) * 100:.0f}%/{my.sum()}"
            print(yl, flush=True)
        # dependencia de unos pocos aciertos gordos
        sel = _topn_mask(p, dayr, m, 3)
        x = np.sort(pnl[sel][np.isfinite(pnl[sel])])[::-1]
        tot = x.sum()
        print(f"  top 3/día: el 1% de las operaciones mejores aporta el {x[:max(1, len(x) // 100)].sum() / tot * 100 if tot > 0 else float('nan'):.0f}% del beneficio total ({len(x)} ops); el 5%: {x[:max(1, len(x) // 20)].sum() / tot * 100 if tot > 0 else float('nan'):.0f}%")


def pelotazo2_explore(EV: Dict, out_dir: str):
    """Robustez del pelotazo: por grupo, sin días de pánico, descuento por quiebras, calibración y composición."""
    import model as MD
    setups_, variants = EV['setups'], EV['variants']
    sid, day, sym, grp = EV['sid'], EV['day'], EV['sym'], EV['grp']
    rows, FL, names = union_events(EV, +1)
    dayr, symr = day[rows], sym[rows]
    yearsr = year_of(dayr)
    X = EV['X'][rows]; gid = grp[rows]
    feats_ = list(MD.MODEL_FEATURES) + ['ret60', 'ret120', 'rs60', 'rs120', 'max_gap5', 'shock5', 'max_vr5']
    fidx = [FEATURES.index(f) for f in feats_]
    fi = {f: j for j, f in enumerate(FEATURES)}
    last_day = int(dayr.max())
    print(f'\n######## PELOTAZO · robustez · {len(rows):,} eventos', flush=True)
    for vname in ('trl_T5_S5_H60', 'trl_T3.5_S3.5_H40'):
        vi = variants.index(vname)
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        bars = EV['bars'][rows, vi]
        y = (pnl >= 12).astype(np.float32)
        p = np.full(len(rows), np.nan)
        for Y in range(2019, int(year_of(np.array([last_day]))[0]) + 1):
            trm = (dayr < day_of(f'{Y}-01-01')) & (dayr >= day_of(f'{Y - 3}-01-01')) & ok
            tem = (yearsr == Y) & ok
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LinearLogitModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
            p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem])
        m = ok & np.isfinite(p)
        sel = _topn_mask(p, dayr, m, 3)
        print(f"\n=== {vname} · política top 3/día · {sel.sum():,} operaciones · duración media {np.nanmean(bars[sel]):.1f} sesiones")
        print(f"  TODAS            {_tail_stats(pnl[sel])}")
        print('  por grupo de activo:')
        for gi, gname in enumerate(GROUPS):
            mg = sel & (gid == gi)
            if mg.sum() >= 40:
                print(f"    {gname:9s} {_tail_stats(pnl[mg])}")
        print('  por año y grupo (μ/n):')
        for gname in ('us_large', 'us_mid', 'us_small', 'etf', 'thematic', 'eu'):
            gi = GROUPS.index(gname); line = f"    {gname:9s}"
            for Y in range(2019, 2027):
                my = sel & (gid == gi) & (yearsr == Y)
                line += f" {str(Y)[2:]}:{np.nanmean(pnl[my]):+.1f}/{my.sum()}" if my.sum() >= 15 else f' {str(Y)[2:]}:—'
            print(line)
        # sin días de pánico de mercado
        bos = X[:, fi['b_os']]; vz = X[:, fi['vix_z']]; spyd = X[:, fi['spy_dd60']]
        calm = sel & (bos < 20) & (vz < 1.5)
        panic = sel & ~((bos < 20) & (vz < 1.5))
        print(f"  días SIN pánico (b_os<20% y VIXz<1.5): {_tail_stats(pnl[calm])}")
        print(f"  días CON pánico                          {_tail_stats(pnl[panic])}")
        for Y in range(2019, 2027):
            mc = calm & (yearsr == Y)
            if mc.sum() >= 20:
                print(f"     sin pánico {Y}: n={mc.sum()} μ={np.nanmean(pnl[mc]):+.2f}% P≥20%={np.mean(pnl[mc] >= 20) * 100:.0f}% PF={stats(pnl[mc])['pf']:.2f}")
        # composición por patrón
        print('  patrones que saltaron en las seleccionadas (% de las operaciones / μ):')
        for c, n_ in enumerate(names):
            mm = sel & (FL[:, c] > 0)
            if mm.sum() >= 40:
                print(f"    {n_:11s} {mm.sum() / sel.sum() * 100:4.0f}% μ={np.nanmean(pnl[mm]):+.2f}% P≥20%={np.mean(pnl[mm] >= 20) * 100:.0f}%")
        # sensibilidad a sesgo de supervivencia: una fracción f de operaciones acaba en catástrofe (-60 %)
        mu = np.nanmean(pnl[sel])
        print('  descuento por quiebras/supervivencia (fracción f de operaciones que acabarían en −60 %):', ' '.join(f"f={f * 100:.0f}%→μ={(1 - f) * mu + f * -60:+.2f}%" for f in (0, 0.02, 0.05, 0.08, 0.10)))
        # costes dobles
        cost_arr = np.array([SM.COST_RT.get(GROUPS[g], 0.30) for g in range(len(GROUPS))])[gid]
        print(f"  costes dobles: μ={np.nanmean(pnl[sel] - cost_arr[sel]):+.2f}%")
        # calibración 2024+: P(≥12%)
        pc_sel = sel & (yearsr >= 2024)
        if pc_sel.sum() > 200:
            qs = np.quantile(p[pc_sel], [0, .25, .5, .75, 1])
            for a_, b_ in zip(qs[:-1], qs[1:]):
                mm = pc_sel & (p >= a_) & (p <= b_)
                print(f"  calibración 2024+: p∈[{a_:.2f},{b_:.2f}] n={mm.sum()} predicho≈{np.mean(p[mm]) * 100:.0f}% real P(≥12%)={y[mm].mean() * 100:.0f}% μ={np.nanmean(pnl[mm]):+.1f}%")
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════
#  MOC: ¿funciona escanear ANTES del cierre y comprar al cierre?
# ═════════════════════════════════════════════════════════════════════════
def _prov_bars(it: Dict, fin: Dict, last_n: int = 900):
    """
    Reconstruye, con barras de 60 min, el diario 'tal como se veía a las 15:30 ET' (30 min antes del cierre):
    cierre provisional = apertura de la última barra horaria; máx/mín sin la última media hora; volumen sin ella.
    Devuelve (serie_final_recortada, serie_provisional, válida) alineadas por fecha, o None.
    """
    off = it['off']
    loc = it['t'] + off
    dday = (loc // 86400).astype(np.int64)
    sec = loc % 86400
    fday = ((fin['t'] + off) // 86400).astype(np.int64)
    n = len(fday)
    a0 = max(0, n - last_n)
    fin_s = {k: v[a0:] for k, v in fin.items()}
    fday = fday[a0:]
    pos = {int(d): i for i, d in enumerate(fday)}
    prov = {k: v.copy() for k, v in fin_s.items()}
    valid = np.zeros(len(fday), bool)
    ud, st = np.unique(dday, return_index=True)
    en = np.concatenate([st[1:], [len(dday)]])
    for d, a, b in zip(ud, st, en):
        i = pos.get(int(d))
        if i is None or b - a < 5 or not (15.25 * 3600 <= sec[b - 1] <= 15.75 * 3600):
            continue
        fac = fin_s['c'][i] / it['c'][b - 1] if it['c'][b - 1] > 0 else 0
        if not (0.5 < fac < 2.0):
            continue
        hi = max(it['h'][a:b - 1].max(), it['o'][b - 1]); lo = min(it['l'][a:b - 1].min(), it['o'][b - 1])
        prov['o'][i] = it['o'][a] * fac; prov['h'][i] = hi * fac; prov['l'][i] = lo * fac
        prov['c'][i] = it['o'][b - 1] * fac; prov['v'][i] = it['v'][a:b - 1].sum()
        valid[i] = True
    return fin_s, prov, valid


def moc_check(uni: Dict, data: Dict, out_dir: str):
    import data as D
    import feats
    import model as MD
    import setups as SU
    here = os.path.dirname(os.path.abspath(__file__))
    reg = json.load(open(os.path.join(here, 'model', 'validated.json')))
    strategies = reg['strategies']
    models = [MD.LogitModel.from_json(s['model']) for s in strategies]
    pattern_ids = [p['id'] for p in reg['patterns']]
    variants = {s['id']: SM.Variant(s['exit']['kind'], s['exit']['T'], s['exit']['S'], s['exit']['H']) for s in strategies}
    us = {'us_large', 'us_mid', 'us_small', 'etf', 'thematic'}
    import re
    syms = [x for x in uni if not uni[x].get('aux') and uni[x]['group'] in us and x in data and not re.search(r'\.[A-Z]{1,3}$', x)]   # solo cotizadas en EE.UU. (sin sufijo de bolsa)
    print(f'\n######## MOC · {len(syms)} símbolos de EE.UU.: descargando barras de 60 min…', flush=True)
    intr = D.fetch_intraday_many(syms + ['SPY', '^VIX'], workers=8)
    print(f'  intradía OK: {len(intr)}/{len(syms) + 2}', flush=True)
    pairs = {}
    for x in syms + ['SPY', '^VIX']:
        if x in intr and x in data:
            r = _prov_bars(intr[x], data[x])
            if r is not None and r[2].sum() > 100:
                pairs[x] = r
    print(f'  días provisionales reconstruidos para {len(pairs)} símbolos · mediana {int(np.median([p[2].sum() for p in pairs.values()]))} días', flush=True)
    if 'SPY' not in pairs:
        print('  sin SPY intradía: no se puede continuar'); return
    # régimen provisional / final
    vix_f = pairs['^VIX'][0] if '^VIX' in pairs else None
    spy_f, spy_p, _ = pairs['SPY']
    vix_p = pairs['^VIX'][1] if '^VIX' in pairs else spy_f
    reg_f = feats.regime_series(spy_f, vix_f if vix_f is not None else {**spy_f}) if vix_f is not None else None
    reg_p = feats.regime_series(spy_p, vix_p) if vix_f is not None else None
    if reg_f is None:
        print('  sin VIX intradía: se usa VIX final en ambos'); vix_f = data['^VIX']
        reg_f = feats.regime_series(spy_f, {k: v[-len(spy_f['c']):] for k, v in vix_f.items()}); reg_p = feats.regime_series(spy_p, {k: v[-len(spy_f['c']):] for k, v in vix_f.items()})
    # amplitud (dos pasadas para no guardar todo)
    bre_f, bre_p = feats.Breadth(), feats.Breadth()
    cache = {}
    for x in syms:
        if x not in pairs:
            continue
        Ff, Fp = feats.build(pairs[x][0]), feats.build(pairs[x][1])
        bre_f.add(Ff); bre_p.add(Fp)
    bd_f, bd_p = bre_f.finalize(), bre_p.finalize()
    rows = {sid: {'fin': [], 'prov': []} for sid in variants}
    n_cand = {'fin': 0, 'prov': 0}
    for x in syms:
        if x not in pairs:
            continue
        fin_s, prov_s, valid = pairs[x]
        g = uni[x]['group']; gi = feats.GROUPS.index(g)
        cost = SM.COST_RT.get(g, 0.30)
        Ff, Fp = feats.build(fin_s), feats.build(prov_s)
        feats.ensure_regime(Ff, reg_f); feats.ensure_regime(Fp, reg_p)
        feats.align_breadth(bd_f, Ff['t'], Ff); feats.align_breadth(bd_p, Fp['t'], Fp)
        for tag, F in (('fin', Ff), ('prov', Fp)):
            idx, FLm = SU.candidate_bars(F, g, pattern_ids)
            keep = valid[idx] & (idx < len(F['c']) - 12) if len(idx) else np.zeros(0, bool)
            idx, FLm = idx[keep], FLm[keep]
            if len(idx) == 0:
                continue
            n_cand[tag] += len(idx)
            Xm = feats.event_matrix(F, idx)
            for s, mdl in zip(strategies, models):
                if g not in s['groups']:
                    continue
                p_raw = mdl.predict_raw(Xm, FLm, np.full(len(idx), gi))
                sel = p_raw >= s['floorRaw']
                if not sel.any():
                    continue
                ii = idx[sel]
                v = variants[s['id']]
                pc = SM.simulate(Ff, ii, +1, [v], cost, entry_mode='close')[v.name].pnl     # resultado SIEMPRE con el cierre FINAL
                po = SM.simulate(Ff, ii, +1, [v], cost, entry_mode='open')[v.name].pnl
                dd = (Ff['t'][ii] // 86400).astype(np.int64)
                rows[s['id']][tag].append((dd, np.full(len(ii), hash(x) % 10 ** 9), p_raw[sel], pc, po, np.full(len(ii), gi)))
    print(f"  candidatos en la ventana: finales {n_cand['fin']:,} · provisionales {n_cand['prov']:,}", flush=True)
    for sid in variants:
        res = {}
        for tag in ('fin', 'prov'):
            if not rows[sid][tag]:
                continue
            dd, ss, pp, pc, po, gg = [np.concatenate(z) for z in zip(*rows[sid][tag])]
            ok = np.isfinite(pc) & np.isfinite(po)
            dd, ss, pp, pc, po, gg = dd[ok], ss[ok], pp[ok], pc[ok], po[ok], gg[ok]
            m = np.ones(len(dd), bool)
            top = _topn_mask(pp, dd, m, 5)
            res[tag] = (dd[top], ss[top], pc[top], po[top], gg[top])
        print(f"\n=== {sid} · política top 5/día (ventana ≈ {int(np.median([p[2].sum() for p in pairs.values()]))} días; el modelo vio estos años al entrenar → valen las DIFERENCIAS, no el nivel) ===")
        for tag, lbl in (('fin', 'señal con CIERRE FINAL'), ('prov', 'señal PROVISIONAL (15:30 ET)')):
            if tag not in res:
                continue
            dd, ss, pc, po, gg = res[tag]
            sc, so = stats(pc, dd, ss), stats(po, dd, ss)
            print(f"  {lbl:30s} n={sc['n']:5d} | compra al CIERRE: WR={sc['wr']:4.1f}% μ={sc['mean']:+5.2f}% PF={sc['pf']:4.2f} | compra a la APERTURA: WR={so['wr']:4.1f}% μ={so['mean']:+5.2f}% PF={so['pf']:4.2f}")
            for gname_ in sorted({GROUPS[g] for g in np.unique(gg)}):
                mg = gg == GROUPS.index(gname_)
                if mg.sum() >= 10:
                    sc_, so_ = stats(pc[mg], dd[mg], ss[mg]), stats(po[mg], dd[mg], ss[mg])
                    dlt = pc[mg] - po[mg]
                    print(f"      · {gname_:9s} n={sc_['n']:4d} | CIERRE: WR={sc_['wr']:4.1f}% μ={sc_['mean']:+5.2f}% | APERTURA: WR={so_['wr']:4.1f}% μ={so_['mean']:+5.2f}% | cierre−apertura {np.nanmean(dlt):+.2f}% ± {1.96 * np.nanstd(dlt) / math.sqrt(max(1, mg.sum())):.2f}")
        if 'fin' in res and 'prov' in res:
            kf = set(zip(res['fin'][0].tolist(), res['fin'][1].tolist())); kp = set(zip(res['prov'][0].tolist(), res['prov'][1].tolist()))
            for gname_ in ('thematic',):
                gi_ = GROUPS.index(gname_)
                kpg = {(d, s_) for d, s_, g in zip(res['prov'][0].tolist(), res['prov'][1].tolist(), res['prov'][4].tolist()) if g == gi_}
                if kpg:
                    print(f"  {gname_}: {len(kpg)} señales provisionales · {len(kpg & kf) / len(kpg) * 100:.0f}% se confirman con el cierre final")
            print(f"  solapamiento: {len(kf & kp) / max(1, len(kp)) * 100:.0f}% de las señales provisionales también lo son al cierre final · {len(kf & kp) / max(1, len(kf)) * 100:.0f}% de las finales ya estaban avisadas a las 15:30")
            both = np.array([(d, s) in kf for d, s in zip(res['prov'][0].tolist(), res['prov'][1].tolist())])
            dd, ss, pc, po, gg = res['prov']
            for lbl, mm in (('provisionales que se CONFIRMAN al cierre', both), ('provisionales que se DESVANECEN', ~both)):
                if mm.sum() >= 20:
                    s_ = stats(pc[mm], dd[mm], ss[mm]); print(f"    {lbl:42s} n={s_['n']:5d} WR={s_['wr']:4.1f}% μ={s_['mean']:+5.2f}%")
    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════
#  IMPROVE: batería PRE-ESPECIFICADA de mejoras de fiabilidad y rentabilidad sobre los planes desplegados
# ═════════════════════════════════════════════════════════════════════════
def improve_explore(EV: Dict, out_dir: str, data: Dict | None = None, uni: Dict | None = None):
    """
    Cada idea se evalúa con la MISMA tubería walk-forward del plan desplegado (modelo con ventana móvil de 3 años,
    N mejores por día sobre la tasa base, grupos del registro) y se compara con la base año a año.
    Criterio para adoptar un cambio: mejora la media en ≥ 6 de 8 años, el cambio es coherente a priori y la mejora
    no depende de un solo año. Con tantas pruebas, una mejora aislada «significativa» puede ser azar: por eso se exige consistencia.
    """
    import model as MD
    import math as _m
    reg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'validated.json')))
    groups_of = {s['id']: s['groups'] for s in reg['strategies']}
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym, gid = EV['day'][rows], EV['sym'][rows], EV['grp'][rows]
    years = year_of(day)
    X = EV['X'][rows]
    last_day = int(day.max())
    yrs = [Y for Y in sorted(set(years)) if Y >= 2019]
    sector_of = {}
    if uni is not None:
        for k, sname in enumerate(EV['sym_names']):
            sector_of[k] = uni.get(sname, {}).get('sector') or f'#{k}'
    sec_ids = {s: i for i, s in enumerate(sorted(set(sector_of.values())))}
    sec = np.array([sec_ids[sector_of[k]] if k in sector_of else k for k in sym], dtype=np.int64) if sector_of else np.arange(len(rows))
    fi = lambda fs: [FEATURES.index(f) for f in fs]

    def wf_lin(y, ok, window=3, l2=30.0, fnames=None):
        idxf = fi(fnames or MD.MODEL_FEATURES)
        p = np.full(len(rows), np.nan); base = np.full(len(rows), np.nan)
        for Y in yrs:
            lo = day_of(f'{Y - window}-01-01') if window else -10 ** 9
            trm = (day < day_of(f'{Y}-01-01')) & (day >= lo) & ok
            tem = (years == Y)
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LinearLogitModel(idxf, names, GROUPS, l2=l2).fit(X[trm], FL[trm], gid[trm], y[trm])
            p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem]); base[tem] = float(y[trm].mean())
        return p, base

    def pick(p, base, ok, allowed, N=5, margin=0.0, cap=None):
        cand = ok & np.isfinite(p) & (p >= base + margin) & np.isin(gid, allowed)
        if cap is None:
            return _topn_mask(p, day, cand, N)
        idx = np.flatnonzero(cand)
        o = idx[np.lexsort((-p[idx], day[idx]))]
        out = np.zeros(len(p), bool)
        cur, cnt, sc = None, 0, {}
        for i in o:
            if day[i] != cur:
                cur, cnt, sc = day[i], 0, {}
            if cnt >= N:
                continue
            s_ = sec[i]
            if sc.get(s_, 0) >= cap:
                continue
            sc[s_] = sc.get(s_, 0) + 1; cnt += 1; out[i] = True
        return out

    def ymeans(sel, pnl):
        return {Y: float(np.nanmean(pnl[sel & (years == Y)])) for Y in yrs if (sel & (years == Y)).sum() >= 20}

    def line(tag, sel, pnl, base_ym=None):
        st = stats(pnl[sel], day[sel], sym[sel])
        if st.get('n', 0) == 0:
            print(f'   {tag:34s} —'); return
        ym = ymeans(sel, pnl)
        dev = sel & (years <= 2023); rec = sel & (years >= 2024)
        d_ = stats(pnl[dev]); r_ = stats(pnl[rec])
        ud, inv = np.unique(day[sel], return_inverse=True)
        da = np.bincount(inv, weights=pnl[sel]) / np.bincount(inv)
        cum = np.cumsum(da); dd = float((np.maximum.accumulate(cum) - cum).max())
        worst = min(ym.values()) if ym else float('nan')
        cmp_ = ''
        if base_ym is not None:
            both = [Y for Y in ym if Y in base_ym]
            cmp_ = f" | mejor que base {sum(ym[Y] > base_ym[Y] for Y in both)}/{len(both)} años"
        print(f"   {tag:34s} n={st['n']:5d} WR={st['wr']:4.1f}% μ={st['mean']:+5.2f}% PF={st['pf']:4.2f} t={st['t_day']:4.1f} | ≤2023 μ={d_.get('mean', float('nan')):+5.2f} · 2024+ μ={r_.get('mean', float('nan')):+5.2f} | peor año {worst:+5.2f} · años+ {sum(v > 0 for v in ym.values())}/{len(ym)} · DD {dd:5.1f}{cmp_}")
        return ym

    print(f'\n######## IMPROVE · {len(rows):,} candidatos largos · {len(yrs)} años de walk-forward ({yrs[0]}–{yrs[-1]})', flush=True)
    spy = data.get('SPY') if data else None
    for plan in PLANS:
        vi = variants.index(plan['id'])
        pnl = EV['pnl'][rows, vi]; bars = EV['bars'][rows, vi]
        ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        allowed = [GROUPS.index(g) for g in groups_of[plan['id']]]
        p0, b0 = wf_lin(y, ok)
        sel0 = pick(p0, b0, ok, allowed)
        print(f"\n================ {plan['id']} · {plan['label']} · grupos {groups_of[plan['id']]} ================")
        base_ym = line('BASE (desplegado)', sel0, pnl)
        # ── riesgo constante (posición ∝ 1/ATR) y calibración por año ─────
        atrp = np.maximum(X[:, FEATURES.index('atrp')], 0.5)
        S_ = SM.default_variants()[vi].S
        w_ = 1.0 / atrp
        mu_w = float(np.nansum(w_[sel0] * pnl[sel0]) / np.nansum(w_[sel0]))
        ymw = {Y: float(np.nansum((w_ * pnl)[sel0 & (years == Y)]) / np.nansum(w_[sel0 & (years == Y)])) for Y in yrs if (sel0 & (years == Y)).sum() >= 20}
        ud, inv = np.unique(day[sel0], return_inverse=True)
        daw = np.bincount(inv, weights=(w_ * pnl)[sel0]) / np.bincount(inv, weights=w_[sel0])
        cumw = np.cumsum(daw)
        print(f"   riesgo constante (posición ∝ 1/ATR): μ ponderada={mu_w:+.2f}% · R medio (resultado/distancia al stop)={np.nanmean(pnl[sel0] / (S_ * atrp[sel0])):+.3f} · peor año {min(ymw.values()):+.2f} · DD {float((np.maximum.accumulate(cumw) - cumw).max()):.1f}")
        print('   calibración por año (p cruda media de las seleccionadas → acierto real): ' + ' '.join(
            f"{str(Y)[2:]}:{np.mean(p0[sel0 & (years == Y)]) * 100:.0f}→{np.mean(y[sel0 & (years == Y)]) * 100:.0f}" for Y in yrs if (sel0 & (years == Y)).sum() >= 20))
        # ── referencia sin modelo ─────────────────────────────────────────
        elig = ok & np.isfinite(p0) & np.isin(gid, allowed)
        rr = []
        for sd in (1, 2, 3):
            rnd = np.random.default_rng(sd).random(len(rows))
            rr.append(_topn_mask(rnd, day, elig, 5))
        ymr = [ymeans(s, pnl) for s in rr]
        mu_r = np.mean([np.nanmean(pnl[s]) for s in rr])
        print(f"   {'SIN modelo (5 al azar/día, 3 semillas)':34s} n={int(np.mean([s.sum() for s in rr])):5d} μ={mu_r:+5.2f}% | años: " + ' '.join(f"{str(Y)[2:]}:{np.mean([m_.get(Y, np.nan) for m_ in ymr]):+.2f}" for Y in yrs))
        print(f"   {'valor añadido del modelo por año':34s} " + ' '.join(f"{str(Y)[2:]}:{base_ym.get(Y, np.nan) - np.mean([m_.get(Y, np.nan) for m_ in ymr]):+.2f}" for Y in yrs))
        bi = EV['setups'].index('B_up') if 'B_up' in EV['setups'] else None
        if bi is not None:
            mb = (EV['sid'] == bi) & np.isfinite(EV['pnl'][:, vi]) & np.isin(EV['grp'], allowed)
            print(f"   {'referencia: entrar sobre SMA200 al azar':34s} n={mb.sum():7d} μ={np.mean(EV['pnl'][mb, vi]):+5.2f}%")
        # ── alfa frente al S&P 500 (misma ventana de cada operación) ──────
        if spy is not None:
            sd_ = (spy['t'] // 86400).astype(np.int64)
            ii = np.searchsorted(sd_, day[sel0], side='left')
            e = np.minimum(ii + 1, len(sd_) - 1); x = np.minimum(ii + np.maximum(bars[sel0], 1), len(sd_) - 1)
            sr = (spy['c'][x] / spy['o'][e] - 1) * 100
            alp = pnl[sel0] - sr
            s_a = stats(alp, day[sel0], sym[sel0])
            ud, inv = np.unique(day[sel0], return_inverse=True)
            da_p = np.bincount(inv, weights=pnl[sel0]) / np.bincount(inv); da_s = np.bincount(inv, weights=sr) / np.bincount(inv)
            print(f"   S&P 500 en las mismas ventanas: μ={np.nanmean(sr):+.2f}% · alfa (operación − S&P) μ={s_a['mean']:+.2f}% t={s_a['t_day']:.1f} · correlación diaria con el S&P {np.corrcoef(da_p, da_s)[0, 1]:+.2f}")
        # ── X1: N por día y margen sobre la tasa base ─────────────────────
        print('  [X1] número de señales por día / margen sobre la tasa base')
        opts1 = {}
        for N in (1, 2, 3, 5, 8, 12):
            s_ = pick(p0, b0, ok, allowed, N=N); opts1[f'N={N}'] = s_
            line(f'N={N}', s_, pnl, base_ym)
        for mg in (0.02, 0.04, 0.06):
            s_ = pick(p0, b0, ok, allowed, margin=mg); opts1[f'N=5, p ≥ base+{mg:.2f}'] = s_
            line(f'N=5, p ≥ base+{mg:.2f}', s_, pnl, base_ym)
        # ── X2: ventana de entrenamiento y regularización ─────────────────
        print('  [X2] ventana de entrenamiento y regularización')
        for w in (2, 4, 5, None):
            pw, bw = wf_lin(y, ok, window=w); line(f"ventana {w if w else 'creciente'}", pick(pw, bw, ok, allowed), pnl, base_ym)
        for l2 in (10.0, 100.0, 300.0):
            pw, bw = wf_lin(y, ok, l2=l2); line(f'l2={l2:g}', pick(pw, bw, ok, allowed), pnl, base_ym)
        # ── X3: variables adicionales (una familia cada vez) ─────────────
        print('  [X3] variables añadidas al modelo (AUC OOS entre paréntesis; base ' + f"{MD.auc(p0[ok & np.isfinite(p0)], y[ok & np.isfinite(p0)]):.3f})")
        fams = {'choque/noticias (max_gap5, shock5, max_vr5, rel5)': ['max_gap5', 'shock5', 'max_vr5', 'rel5'],
                'momentum y fuerza relativa (ret60, rs60, rs120)': ['ret60', 'rs60', 'rs120'],
                'corto plazo (ret2, ret3, rsi3, up_streak)': ['ret2', 'ret3', 'rsi3', 'up_streak'],
                'mercado (spy_ret5, dist20, ret20)': ['spy_ret5', 'dist20', 'ret20']}
        for fn, fl in fams.items():
            fl = [f for f in fl if f in FEATURES]
            pw, bw = wf_lin(y, ok, fnames=list(MD.MODEL_FEATURES) + fl)
            mm = ok & np.isfinite(pw)
            line(f'+ {fn[:30]} ({MD.auc(pw[mm], y[mm]):.3f})', pick(pw, bw, ok, allowed), pnl, base_ym)
        # ── X4: puertas de régimen sobre las operaciones seleccionadas ───
        print('  [X4] puertas de régimen (se descartan las señales seleccionadas que cumplen la condición)')
        col = lambda n: X[:, FEATURES.index(n)]
        gates = {'VIX z > 1,0': col('vix_z') > 1.0, 'VIX z > 1,5': col('vix_z') > 1.5, 'VIX z > 2,0': col('vix_z') > 2.0,
                 'SPY bajo su SMA200': col('spy_up') == 0, 'amplitud < 30% sobre SMA200': col('b_up200') < 30,
                 'amplitud < 40%': col('b_up200') < 40, 'sobreventa amplia > 15%': col('b_os') > 15, 'sobreventa amplia > 25%': col('b_os') > 25,
                 'ATR > 5% del precio': col('atrp') > 5, 'ATR > 7% del precio': col('atrp') > 7,
                 'SPY cae > 4% en 5 sesiones': col('spy_ret5') < -4 if 'spy_ret5' in FEATURES else np.zeros(len(rows), bool)}
        opts4 = {'sin puerta': sel0}
        for gname, gm in gates.items():
            gm = np.nan_to_num(gm.astype(float)).astype(bool)
            keep, drop = sel0 & ~gm, sel0 & gm
            s_d = stats(pnl[drop]) if drop.sum() else {}
            ym = line(f'sin: {gname}', keep, pnl, base_ym)
            print(f"      descartadas n={int(drop.sum())} ({drop.sum() / sel0.sum() * 100:.0f}%) μ={s_d.get('mean', float('nan')):+.2f}% WR={s_d.get('wr', float('nan')):.0f}%")
            opts4[gname] = keep
        # ── X5: tope por sector ──────────────────────────────────────────
        print('  [X5] máximo de señales por sector y día (diversificación)')
        for cap in (1, 2, 3):
            line(f'≤ {cap} por sector', pick(p0, b0, ok, allowed, cap=cap), pnl, base_ym)
        # ── X6: elección ADAPTATIVA (cada año se elige con los años anteriores) ──
        for lbl, opts in (('N y margen', opts1), ('puertas de régimen', opts4)):
            tot, tb = [], []
            for Y in yrs:
                if Y < 2022:
                    continue
                prior = {k: np.nanmean(pnl[s & (years < Y)]) for k, s in opts.items() if (s & (years < Y)).sum() >= 200}
                if not prior:
                    continue
                best = max(prior, key=prior.get)
                tot.append(pnl[opts[best] & (years == Y)]); tb.append(pnl[sel0 & (years == Y)])
            if tot:
                a, b = np.concatenate(tot), np.concatenate(tb)
                print(f"  [X6] elección adaptativa de «{lbl}» (cada año se elige con los años previos) 2022–{yrs[-1]}: μ={np.nanmean(a):+.2f}% (n={len(a)}) frente a base {np.nanmean(b):+.2f}% (n={len(b)})")
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════
#  IMPROVE2: puestos del día, réplica de mejoras sospechosas y SIMULACIÓN DE CARTERA con capital limitado
# ═════════════════════════════════════════════════════════════════════════
def improve2_explore(EV: Dict, out_dir: str, data: Dict | None = None, uni: Dict | None = None):
    import model as MD
    reg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'validated.json')))
    groups_of = {s['id']: s['groups'] for s in reg['strategies']}
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym, gid = EV['day'][rows], EV['sym'][rows], EV['grp'][rows]
    years = year_of(day)
    X = EV['X'][rows]
    yrs = [Y for Y in sorted(set(years)) if Y >= 2019]
    fi = lambda fs: [FEATURES.index(f) for f in fs]

    def wf_lin(y, ok, window=3, fnames=None):
        idxf = fi(fnames or MD.MODEL_FEATURES)
        p = np.full(len(rows), np.nan); base = np.full(len(rows), np.nan)
        for Y in yrs:
            lo = day_of(f'{Y - window}-01-01') if window else -10 ** 9
            trm = (day < day_of(f'{Y}-01-01')) & (day >= lo) & ok
            tem = (years == Y)
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LinearLogitModel(idxf, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
            p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem]); base[tem] = float(y[trm].mean())
        return p, base

    def pick(p, base, ok, allowed, N=5):
        return _topn_mask(p, day, ok & np.isfinite(p) & (p >= base) & np.isin(gid, allowed), N)

    def ymeans(sel, pnl):
        return {Y: float(np.nanmean(pnl[sel & (years == Y)])) for Y in yrs if (sel & (years == Y)).sum() >= 20}

    def rank_in_day(p, sel):
        idx = np.flatnonzero(sel)
        o = idx[np.lexsort((-p[idx], day[idx]))]
        d = day[o]
        first = np.concatenate([[0], np.flatnonzero(np.diff(d)) + 1])
        rk = np.arange(len(o)) - np.repeat(first, np.diff(np.concatenate([first, [len(o)]])))
        out = np.zeros(len(p), np.int16); out[o] = rk + 1
        return out

    spy = data.get('SPY') if data else None
    sd_ = (spy['t'] // 86400).astype(np.int64) if spy is not None else None

    def portfolio(sel, p, pnl, bars, N, M, f, label, alt=None):
        """Cartera con capital limitado: como mucho M posiciones abiertas, cada una el % f del capital ACTUAL; se atienden las señales por puesto."""
        rk = rank_in_day(p, sel)
        use = sel & (rk <= N) & np.isfinite(pnl)
        idx = np.flatnonzero(use)
        idx = idx[np.lexsort((rk[idx], day[idx]))]
        ti = np.searchsorted(sd_, day[idx], side='left')           # índice de la sesión de la señal
        nd = len(sd_)
        eq = 1.0
        open_ = []                                                   # (salida_ti, importe, pnl%)
        curve = np.full(nd, np.nan)
        pos_i = 0
        n_taken = 0
        res = pnl if alt is None else alt
        order = {}
        for k, t_ in zip(idx, ti):
            order.setdefault(int(t_), []).append(k)
        eq_hist = np.ones(nd)
        for t in range(int(ti.min()) if len(ti) else 0, nd):
            # cierres de las operaciones cuya sesión de salida es t
            still = []
            for ex_t, amt, r in open_:
                if ex_t <= t:
                    eq += amt * r / 100.0
                else:
                    still.append((ex_t, amt, r))
            open_ = still
            for k in order.get(t - 1, []):                           # señal ayer → entrada hoy
                if len(open_) >= M:
                    break
                b = int(max(bars[k], 1))
                if not np.isfinite(res[k]):
                    continue
                open_.append((t - 1 + b, eq * f, float(res[k]))); n_taken += 1
            eq_hist[t] = eq
        eq_hist = eq_hist[int(ti.min()):]
        dd = float(((np.maximum.accumulate(eq_hist) - eq_hist) / np.maximum.accumulate(eq_hist)).max())
        rets = np.diff(eq_hist) / eq_hist[:-1]
        yrs_n = len(eq_hist) / 252.0
        cagr = eq_hist[-1] ** (1 / yrs_n) - 1
        sh = rets.mean() / (rets.std() + 1e-12) * math.sqrt(252)
        mo = np.array([eq_hist[min(i + 21, len(eq_hist) - 1)] / eq_hist[i] - 1 for i in range(0, len(eq_hist) - 1, 21)])
        print(f"   {label:34s} CAGR {cagr * 100:+6.1f}% · DD máx {dd * 100:5.1f}% · Sharpe {sh:4.2f} · meses+ {np.mean(mo > 0) * 100:3.0f}% · peor mes {mo.min() * 100:+5.1f}% · {n_taken / yrs_n:5.0f} op/año")
        return eq_hist

    print(f'\n######## IMPROVE2 · {len(rows):,} candidatos largos', flush=True)
    if spy is not None:
        i0 = np.searchsorted(sd_, day_of('2019-01-01'))
        c = spy['c'][i0:]; r_ = np.diff(c) / c[:-1]; ddm = float(((np.maximum.accumulate(c) - c) / np.maximum.accumulate(c)).max())
        print(f"  Referencia S&P 500 comprar y mantener desde 2019: CAGR {((c[-1] / c[0]) ** (252 / len(c)) - 1) * 100:+.1f}% · DD máx {ddm * 100:.1f}% · Sharpe {r_.mean() / r_.std() * math.sqrt(252):.2f}")
    for plan in PLANS:
        vi = variants.index(plan['id'])
        pnl = EV['pnl'][rows, vi]; bars = EV['bars'][rows, vi]
        ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        allowed = [GROUPS.index(g) for g in groups_of[plan['id']]]
        p0, b0 = wf_lin(y, ok)
        sel0 = pick(p0, b0, ok, allowed)
        print(f"\n================ {plan['id']} · {plan['label']} ================")
        # ── puestos del día ──────────────────────────────────────────────
        rk = rank_in_day(p0, sel0)
        print('  [R] resultado por PUESTO en el ranking del día (fuera de muestra, plan desplegado, N=5):')
        for r in range(1, 6):
            m = sel0 & (rk == r)
            if m.sum() >= 50:
                st = stats(pnl[m], day[m], sym[m]); ym = ymeans(m, pnl)
                print(f"     puesto {r}: n={st['n']:5d} WR={st['wr']:4.1f}% μ={st['mean']:+5.2f}% PF={st['pf']:4.2f} t={st['t_day']:4.1f} · años+ {sum(v > 0 for v in ym.values())}/{len(ym)}")
        cnt = np.bincount(np.unique(day[sel0], return_inverse=True)[1])
        print(f"     señales por día con alguna: media {cnt.mean():.1f} · mediana {np.median(cnt):.0f} · días con señal {len(cnt)} de {len(np.unique(day[ok]))}")
        # ── réplica de la mejora «variables de mercado» y de la ventana 2 con salidas vecinas ──
        print('  [Z] réplica: ¿la mejora aislada de X2/X3 se repite con salidas vecinas? (μ de la política y años mejores que la base)')
        neigh = [v for v in variants if v.startswith(plan['id'].split('_')[0] + '_')][:0]
        nb = {'rsi_S4_H10': ['rsi_S4_H10', 'rsi_S2.5_H10', 'rsi_S4_H5', 'rsi_S1.5_H10'],
              'atr_T1_S4_H10': ['atr_T1_S4_H10', 'atr_T0.5_S4_H10', 'atr_T1.5_S4_H10', 'atr_T1_S2.5_H10', 'atr_T1_S4_H5']}[plan['id']]
        cfgs = {'base': dict(), 'ventana 2': dict(window=2), '+mercado (3)': dict(fnames=list(MD.MODEL_FEATURES) + ['spy_ret5', 'dist20', 'ret20']),
                '+spy_ret5': dict(fnames=list(MD.MODEL_FEATURES) + ['spy_ret5']), '+dist20': dict(fnames=list(MD.MODEL_FEATURES) + ['dist20']),
                '+ret20': dict(fnames=list(MD.MODEL_FEATURES) + ['ret20'])}
        for vn in nb:
            vj = variants.index(vn)
            pn = EV['pnl'][rows, vj]; okj = np.isfinite(pn); yj = (pn > 0).astype(np.float32)
            base_ym = None; out = []
            for cn, kw in cfgs.items():
                pp, bb = wf_lin(yj, okj, **kw)
                sl = pick(pp, bb, okj, allowed)
                ym = ymeans(sl, pn)
                if cn == 'base':
                    base_ym = ym
                better = sum(ym[Y] > base_ym[Y] for Y in ym if Y in base_ym)
                out.append(f"{cn}: {np.nanmean(pn[sl]):+.2f}%" + ('' if cn == 'base' else f" ({better}/{len(ym)})"))
            print(f"     {vn:18s} " + ' | '.join(out), flush=True)
        # ── cartera con capital limitado ─────────────────────────────────
        if spy is not None:
            print('  [P] CARTERA con capital limitado (a partir de 2019; M posiciones máx., cada una el f% del capital actual; resultados netos de costes; curva a precio realizado)')
            for N in (1, 3, 5):
                for M, f in ((5, 0.20), (10, 0.10), (20, 0.05)):
                    portfolio(sel0, p0, pnl, bars, N, M, f, f'N={N} · máx {M} pos. · {int(f * 100)}% c/u')
            # el mismo calendario de entradas pero comprando el S&P 500 en lugar de la acción
            ii = np.searchsorted(sd_, day, side='left')
            e = np.minimum(ii + 1, len(sd_) - 1); xx = np.minimum(ii + np.maximum(bars.astype(int), 1), len(sd_) - 1)
            alt = (spy['c'][xx] / spy['o'][e] - 1) * 100 - 0.03
            print('   — y con el MISMO calendario de entradas/salidas pero comprando el S&P 500 (SPY) en vez de la acción:')
            for N, M, f in ((3, 10, 0.10), (5, 10, 0.10)):
                portfolio(sel0, p0, pnl, bars, N, M, f, f'SPY · N={N} · máx {M} pos. · {int(f * 100)}% c/u', alt=alt)
            ms = sel0 & np.isfinite(alt)
            print(f"   SPY en las mismas ventanas: WR={np.mean(alt[ms] > 0) * 100:.1f}% μ={np.nanmean(alt[ms]):+.2f}% · plan en acciones: WR={np.mean(pnl[ms] > 0) * 100:.1f}% μ={np.nanmean(pnl[ms]):+.2f}%")
        sys.stdout.flush()
