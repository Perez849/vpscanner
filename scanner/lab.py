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
                      topn_mask as _topn_mask)

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


