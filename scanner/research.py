#!/usr/bin/env python3
"""
research.py — Laboratorio de validación fuera de muestra.

Fases temporales (fijadas ANTES de mirar resultados, para no sobreajustar):
    TRAIN  : hasta 2021-12-31      (se explora y se ajusta)
    VAL    : 2022-01-01 .. 2023-12-31 (se confirma)
    TEST   : desde 2024-01-01      (BLOQUEADO: solo se destapa con --final)

Subcomandos:
    explore  → tabla setup × salida sobre TRAIN+VAL (nada de TEST)
    final    → selecciona con TRAIN+VAL, entrena el modelo de probabilidad y mide UNA vez en TEST
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D
import feats
import setups as SU
import simulate as SM
import universe as UV

TRAIN_END = '2021-12-31'
VAL_END = '2023-12-31'

FEATURES = feats.FEATURES
GROUPS = feats.GROUPS


def day_of(s: str) -> int:
    return int(np.datetime64(s).astype('datetime64[D]').astype(np.int64))


def year_of(day: np.ndarray) -> np.ndarray:
    return (np.datetime64('1970-01-01') + day.astype('timedelta64[D]')).astype('datetime64[Y]').astype(int) + 1970


# ═════════════════════════════════════════════════════════════════════════
#  Construcción de la tabla de eventos
# ═════════════════════════════════════════════════════════════════════════
def synthetic_universe(n=60, ar=0.0):
    import synth
    return synth.universe(n, ar, n_bars=3400)


def build_events(data: Dict, uni: Dict, setup_list: List[SU.Setup], variants: List[SM.Variant],
                 only_last: bool = False, verbose: bool = True):
    """
    Recorre el universo y devuelve la tabla de eventos en arrays NumPy:
      sid, sym_i, grp_i, day, X[n, len(FEATURES)], pnl[n,V], kind[n,V], bars[n,V]
    """
    spy, vix = data.get('SPY'), data.get('^VIX')
    reg = feats.regime_series(spy, vix) if (spy is not None and vix is not None) else None
    sym_list = [s for s in uni if not uni[s].get('aux') and s in data]
    sid_of = {s.id: k for k, s in enumerate(setup_list)}
    # Pasada 1: amplitud de mercado del universo (fracción sobre SMA200, sobreventa, rentabilidad media)
    bre = feats.Breadth()
    for sym in sym_list:
        bre.add(feats.build(data[sym]))
    bd = bre.finalize()
    print('  amplitud de mercado lista', flush=True)
    acc = {k: [] for k in ('sid', 'sym', 'grp', 'day', 'bi', 'X', 'pnl', 'kind', 'bars')}
    t0 = time.time()
    for n_s, sym in enumerate(sym_list):
        g = uni[sym]['group']
        F = feats.build(data[sym])
        feats.ensure_regime(F, reg)
        feats.align_breadth(bd, F['t'], F)
        ev = SU.detect(F, g, setup_list)
        cost = SM.COST_RT.get(g, 0.30)
        # un único cálculo de simulación por dirección sobre la unión de índices
        for sgn in (+1, -1):
            ids = [s.id for s in setup_list if s.dir == sgn and len(ev[s.id])]
            if not ids:
                continue
            uni_idx = np.unique(np.concatenate([ev[i] for i in ids]))
            res = SM.simulate(F, uni_idx, sgn, variants, cost)
            P = np.stack([res[v.name].pnl for v in variants], 1).astype(np.float32)
            K = np.stack([res[v.name].kind for v in variants], 1)
            B = np.stack([res[v.name].bars for v in variants], 1)
            Xs = feats.event_matrix(F, uni_idx)
            for sid in ids:
                pos = np.searchsorted(uni_idx, ev[sid])
                acc['sid'].append(np.full(len(pos), sid_of[sid], np.int16))
                acc['sym'].append(np.full(len(pos), n_s, np.int32))
                acc['grp'].append(np.full(len(pos), GROUPS.index(g), np.int8))
                acc['day'].append((F['t'][uni_idx][pos] // 86400).astype(np.int32))
                acc['bi'].append(uni_idx[pos].astype(np.int32))
                acc['X'].append(Xs[pos]); acc['pnl'].append(P[pos])
                acc['kind'].append(K[pos]); acc['bars'].append(B[pos])
        if verbose and (n_s + 1) % 300 == 0:
            print(f'  eventos: {n_s + 1}/{len(sym_list)} símbolos ({time.time() - t0:.0f}s)', flush=True)
    EV = {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}
    EV['sym_names'] = sym_list
    EV['setups'] = [s.id for s in setup_list]
    EV['variants'] = [v.name for v in variants]
    return EV


# ═════════════════════════════════════════════════════════════════════════
#  Estadísticos
# ═════════════════════════════════════════════════════════════════════════
def wilson_lo(w: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


def stats(pnl: np.ndarray, day: np.ndarray | None = None, sym: np.ndarray | None = None) -> Dict:
    x = pnl[np.isfinite(pnl)]
    n = len(x)
    if n == 0:
        return {'n': 0}
    w = int((x > 0).sum())
    gp, gl = x[x > 0].sum(), -x[x < 0].sum()
    d = {'n': n, 'wr': w / n * 100, 'wr_lo': wilson_lo(w, n) * 100, 'mean': float(x.mean()),
         'med': float(np.median(x)), 'pf': float(gp / gl) if gl > 0 else 99.0,
         'avg_win': float(x[x > 0].mean()) if w else 0.0, 'avg_loss': float(x[x <= 0].mean()) if n - w else 0.0}
    if day is not None:
        dd = day[np.isfinite(pnl)]
        u, inv = np.unique(dd, return_inverse=True)
        # error estándar robusto por clústeres de día (las señales de un mismo día están correlacionadas)
        r = np.bincount(inv, weights=x - x.mean())
        se = math.sqrt(float((r ** 2).sum())) / n
        if sym is not None:        # y por símbolo: ventanas solapadas de un mismo activo comparten trayectoria
            _, inv2 = np.unique(sym[np.isfinite(pnl)], return_inverse=True)
            r2 = np.bincount(inv2, weights=x - x.mean())
            se = max(se, math.sqrt(float((r2 ** 2).sum())) / n)
        d['se'] = se
        d['t_day'] = float(x.mean() / (se + 1e-12))
    return d


def split_masks(day: np.ndarray):
    tr, va = day_of(TRAIN_END), day_of(VAL_END)
    return day <= tr, (day > tr) & (day <= va), day > va


def fmt(s: Dict) -> str:
    if s.get('n', 0) == 0:
        return '        —'
    return f"n={s['n']:6d} WR={s['wr']:5.1f}% (lo {s['wr_lo']:4.1f}) μ={s['mean']:+6.2f}% PF={s['pf']:4.2f}"


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



# ═════════════════════════════════════════════════════════════════════════
#  META: ¿un modelo de contexto separa las buenas de las malas señales?
# ═════════════════════════════════════════════════════════════════════════
UNION_COOLDOWN = 5


def union_events(EV: Dict, sgn: int):
    """Un evento por (símbolo, día) uniendo todos los patrones de esa dirección; banderas de qué patrones saltaron."""
    setups_ = EV['setups']
    ids = [k for k, sn in enumerate(setups_) if SU.BY_ID[sn].dir == sgn and SU.BY_ID[sn].family not in ('baseline', 'control')]
    m = np.isin(EV['sid'], ids)
    sid_m = EV['sid'][m]
    key = EV['sym'][m].astype(np.int64) * 100000 + EV['day'][m]
    uk, first = np.unique(key, return_index=True)
    rows = np.flatnonzero(m)[first]
    FL = np.zeros((len(uk), len(ids)), np.float32)
    for c, k in enumerate(ids):
        FL[np.isin(uk, key[sid_m == k]), c] = 1.0
    # enfriamiento sin estado: solo el primer evento si no hubo otro en las 4 barras previas del mismo símbolo
    sym, bi = EV['sym'][rows], EV['bi'][rows]
    o = np.lexsort((bi, sym))
    keep = np.ones(len(rows), bool)
    sy, b = sym[o], bi[o]
    same = np.concatenate([[False], sy[1:] == sy[:-1]])
    gap = np.concatenate([[10 ** 6], np.diff(b)])
    keep_o = ~(same & (gap < UNION_COOLDOWN))
    keep[o] = keep_o
    return rows[keep], FL[keep], [setups_[k] for k in ids]


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


# ═════════════════════════════════════════════════════════════════════════
#  FINAL: estrategias (variante de salida + umbral de probabilidad) y prueba ciega
# ═════════════════════════════════════════════════════════════════════════
# Se despliega UNA sola regla de salida (elegida con datos de desarrollo). Las alternativas solo se evalúan para el informe.
FINAL_VARIANTS = ['rsi_S2.5_H10']
ALT_VARIANTS = ['rsi_S4_H10', 'sig_S4_H10', 'ph_S4_H10']
CRIT = dict(min_n=600, min_wr=66.0, min_mean=0.35, min_pf=1.30, min_t=2.5, min_margin=3.0, min_years_pos=0.8)
THR_GRID = [round(x, 2) for x in np.arange(0.55, 0.91, 0.01)]
FIRST_WF_YEAR = 2019


def wf_oos_p(EV, rows, FL, y, ok, upto_day):
    """Probabilidad cruda fuera de muestra (walk-forward anual) para todos los eventos con día <= upto_day."""
    import model as MD
    day = EV['day'][rows]; years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    p = np.full(len(rows), np.nan)
    last_year = int(year_of(np.array([upto_day]))[0])
    for Y in range(FIRST_WF_YEAR, last_year + 1):
        trm = (day < day_of(f'{Y}-01-01')) & ok
        tem = (years == Y) & (day <= upto_day)
        if trm.sum() < 3000 or not tem.any():
            continue
        m = MD.LogitModel(fidx, ['f%d' % i for i in range(FL.shape[1])], GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
        p[tem] = m.predict_raw(X[tem], FL[tem], gid[tem])
    return p


def pick_threshold(pc, pnl, day, sym, years, m, crit=CRIT):
    """Umbral de probabilidad más bajo que cumple todos los criterios en OOS (más alertas con la misma calidad)."""
    for thr in THR_GRID:
        mm = m & (pc >= thr)
        if mm.sum() < crit['min_n']:
            return None
        st = stats(pnl[mm], day[mm], sym[mm])
        be = -st['avg_loss'] / (st['avg_win'] - st['avg_loss']) * 100 if st['avg_win'] > 0 and st['avg_loss'] < 0 else 100.0
        yrs = [int(y) for y in np.unique(years[mm]) if (mm & (years == y)).sum() >= 30]
        pos = sum(1 for y in yrs if np.nanmean(pnl[mm & (years == y)]) > 0)
        if (st['wr'] >= crit['min_wr'] and st['mean'] >= crit['min_mean'] and st['pf'] >= crit['min_pf'] and st['t_day'] >= crit['min_t']
                and st['wr_lo'] - be >= crit['min_margin'] and len(yrs) >= 3 and pos / len(yrs) >= crit['min_years_pos']):
            return thr, st, be, f'{pos}/{len(yrs)}'
    return None


def run_procedure(EV: Dict, cutoff_day: int, test: bool):
    """Procedimiento completo con datos hasta cutoff_day. Si test=True evalúa en los eventos posteriores."""
    import model as MD
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    dev = day <= cutoff_day
    strategies = []
    print(f"\n########  PROCEDIMIENTO con datos hasta {np.datetime64('1970-01-01') + np.timedelta64(cutoff_day, 'D')}  ·  {dev.sum():,} eventos candidatos  ########", flush=True)
    for vname in FINAL_VARIANTS:
        vi = variants.index(vname)
        pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        p = wf_oos_p(EV, rows, FL, y, ok, cutoff_day)
        m = dev & ok & np.isfinite(p)
        if m.sum() < 5000:
            continue
        cal = MD.LogitModel(fidx, names, GROUPS)
        cal.set_calibration(p[m], y[m])
        xs = np.array([c[0] for c in cal.calib]); ys = np.array([c[1] for c in cal.calib])
        pc = np.interp(p, xs, ys) if len(xs) >= 2 else p
        print(f"\n  [{vname}] AUC OOS={MD.auc(p[m], y[m]):.3f} · sin modelo: {fmt(stats(pnl[m], day[m], sym[m]))}")
        pk = pick_threshold(pc, pnl, day, sym, years, m)
        if pk is None:
            print('     ✘ ningún umbral cumple los criterios → variante descartada')
            continue
        # grupos de activos en los que de verdad funciona (el resto se excluye automáticamente)
        allowed = []
        for gi, gname in enumerate(GROUPS):
            mg = m & (pc >= pk[0]) & (gid == gi)
            if mg.sum() >= 80:
                sg_ = stats(pnl[mg], day[mg], sym[mg])
                ok_g = sg_['mean'] > 0.05 and sg_['wr'] >= CRIT['min_wr'] - 6
                print(f"       grupo {gname:9s} {fmt(sg_)} → {'ok' if ok_g else 'EXCLUIDO'}")
                if ok_g:
                    allowed.append(gi)
            elif mg.sum() > 0:
                print(f"       grupo {gname:9s} n={mg.sum()} (<80, sin evidencia suficiente) → EXCLUIDO")
        m2 = m & np.isin(gid, allowed)
        pk = pick_threshold(pc, pnl, day, sym, years, m2)
        if pk is None:
            print('     ✘ tras excluir grupos ningún umbral cumple → descartada')
            continue
        m = m2
        thr, st, be, yr_ok = pk
        print(f"     ✔ umbral calibrado P≥{thr:.2f} en {[GROUPS[g] for g in allowed]}: {fmt(st)} · t={st['t_day']:.1f} · acierto de equilibrio {be:.1f}% · años positivos {yr_ok}")
        mm = m & (pc >= thr)
        by_year = {}
        for Y in np.unique(years[mm]):
            my = mm & (years == Y)
            if my.sum() >= 30:
                sy = stats(pnl[my]); by_year[str(int(Y))] = {'n': sy['n'], 'wr': round(sy['wr'], 1), 'mean': round(sy['mean'], 2)}
        by_group = {}
        for gi, gname in enumerate(GROUPS):
            mg = mm & (gid == gi)
            if mg.sum() >= 30:
                sg_ = stats(pnl[mg]); by_group[gname] = {'n': sg_['n'], 'wr': round(sg_['wr'], 1), 'mean': round(sg_['mean'], 2)}
        final_m = MD.LogitModel(fidx, names, GROUPS).fit(X[dev & ok], FL[dev & ok], gid[dev & ok], y[dev & ok])
        final_m.calib = cal.calib
        v = SM.default_variants()[vi]
        strategies.append({
            'id': vname, 'exit': {'kind': v.kind, 'T': v.T, 'S': v.S, 'H': v.H}, 'thr': thr, 'vi': vi,
            'groups': [GROUPS[g] for g in allowed], 'gids': allowed,
            'ev': {'aw': round(st['avg_win'], 3), 'al': round(st['avg_loss'], 3)},
            'stats': {'oos': {k: round(float(st[k]), 3) for k in ('n', 'wr', 'wr_lo', 'mean', 'pf', 't_day', 'avg_win', 'avg_loss')},
                      'be': round(be, 1), 'byYear': by_year, 'byGroup': by_group},
            'model': final_m})
    out = {'strategies': strategies, 'names': names}
    if test:   # alternativas SOLO informativas (no se despliegan)
        te_ = day > cutoff_day
        print('\n  (informativo) otras reglas de salida con el mismo procedimiento, evaluadas en la prueba ciega:')
        for vname in ALT_VARIANTS:
            vi = variants.index(vname)
            pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl); y = (pnl > 0).astype(np.float32)
            p = wf_oos_p(EV, rows, FL, y, ok, cutoff_day)
            m = dev & ok & np.isfinite(p)
            if m.sum() < 5000:
                continue
            cal = MD.LogitModel(fidx, names, GROUPS); cal.set_calibration(p[m], y[m])
            xs = np.array([c[0] for c in cal.calib]); ys = np.array([c[1] for c in cal.calib])
            pc = np.interp(p, xs, ys)
            pk = pick_threshold(pc, pnl, day, sym, years, m)
            if pk is None:
                print(f'     {vname}: no supera los criterios en desarrollo'); continue
            fm = MD.LogitModel(fidx, names, GROUPS).fit(X[dev & ok], FL[dev & ok], gid[dev & ok], y[dev & ok]); fm.calib = cal.calib
            pt = fm.predict(X, FL, gid); mt = te_ & ok & (pt >= pk[0])
            print(f"     {vname}: umbral {pk[0]:.2f} · desarrollo {fmt(pk[1])} · TEST {fmt(stats(pnl[mt], day[mt], sym[mt]))}")
    if test and strategies:
        te = (day > cutoff_day)
        print(f'\n  ====== PRUEBA CIEGA (eventos posteriores al corte: {te.sum():,}) ======', flush=True)
        best_pc = np.full(len(rows), -1.0); best_pnl = np.full(len(rows), np.nan); best_k = np.full(len(rows), -1)
        for k, sg_ in enumerate(strategies):
            vi = sg_['vi']; pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl)
            p = sg_['model'].predict(X, FL, gid)           # con calibración OOS
            mt = te & ok & np.isin(gid, sg_['gids'])
            for thr in (sg_['thr'], 0.70, 0.75, 0.80):
                mm = mt & (p >= thr)
                if mm.sum() >= 30:
                    print(f"   {sg_['id']:14s} P≥{thr:.2f}: {fmt(stats(pnl[mm], day[mm], sym[mm]))}")
            sel = mt & (p >= sg_['thr']) & (p > best_pc)
            best_pc[sel] = p[sel]; best_pnl[sel] = pnl[sel]; best_k[sel] = k
        mm = best_k >= 0
        # tope diario: solo las N mejores señales por día (en un día de pánico hay cientos, todas correlacionadas)
        for cap in (5, 15):
            idx_ = np.flatnonzero(mm)
            o_ = idx_[np.lexsort((-best_pc[idx_], day[idx_]))]
            d_ = day[o_]
            first = np.concatenate([[0], np.flatnonzero(np.diff(d_)) + 1])
            rank = np.arange(len(o_)) - np.repeat(first, np.diff(np.concatenate([first, [len(o_)]])))
            sel_ = o_[rank < cap]
            if len(sel_) >= 30:
                print(f"   con tope de {cap} señales/día: {fmt(stats(best_pnl[sel_], day[sel_], sym[sel_]))}")
        if mm.sum() < 5:
            print('   (sin señales en la prueba ciega)')
            return out
        st = stats(best_pnl[mm], day[mm], sym[mm])
        ny = max(1, len(np.unique(year_of(day[te])))) 
        ndays = max(1, len(np.unique(day[te])))
        print(f"   COMBINADO (mejor estrategia por señal, P≥umbral): {fmt(st)} · t={st['t_day']:.1f} · {mm.sum() / ndays:.2f} alertas/día de mercado")
        for Y in sorted(set(year_of(day[te]))):
            my = mm & (year_of(day) == Y)
            sy = stats(best_pnl[my]); print(f"      {Y}: n={sy.get('n', 0)} WR={sy.get('wr', 0):.1f}% μ={sy.get('mean', 0):+.2f}% PF={sy.get('pf', 0):.2f}")
        tab = {}
        for g in (0.65, 0.70, 0.75, 0.80):
            sel = (best_k >= 0) | True
            # umbral global sobre la mejor probabilidad calibrada de cualquier estrategia
            pcs = np.full(len(rows), -1.0); pn = np.full(len(rows), np.nan)
            for k, sg_ in enumerate(strategies):
                vi = sg_['vi']; pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl) & te & np.isin(gid, sg_['gids'])
                p = sg_['model'].predict(X, FL, gid)
                better = ok & (p > pcs)
                pcs[better] = p[better]; pn[better] = pnl[better]
            mq = (pcs >= g) & np.isfinite(pn)
            if mq.sum() >= 30:
                sq = stats(pn[mq], day[mq], sym[mq]); tab[str(g)] = {k: round(float(sq[k]), 3) for k in ('n', 'wr', 'wr_lo', 'mean', 'pf')}
                print(f"   P≥{g:.2f} (mejor prob. de cualquier estrategia): {fmt(sq)}")
        out['test_table'] = tab
        out['test_stats'] = {k: round(float(st[k]), 3) for k in ('n', 'wr', 'wr_lo', 'mean', 'pf')} if st.get('n') else None
    return out


def registry_json(proc: Dict, last_day_str: str, universe_n: int, honest: Dict | None, names: List[str]) -> Dict:
    strategies = []
    for sg_ in proc['strategies']:
        d = {k: v for k, v in sg_.items() if k not in ('model', 'vi', 'gids')}
        d['model'] = sg_['model'].to_json()
        strategies.append(d)
    return {'version': 3, 'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'dataThrough': last_day_str,
            'universe': universe_n, 'periods': {'firstWalkForward': FIRST_WF_YEAR, 'blindFrom': '2024-01-01'},
            'criteria': CRIT,
            'rules': {'maxAlerts': 15, 'watchMargin': 0.05},
            'patterns': [{'id': n, 'label': SU.BY_ID[n].label} for n in names],
            'summary': honest, 'strategies': strategies}


def final(EV: Dict, out_dir: str, universe_n: int):
    cut = day_of(VAL_END)
    # 1) Procedimiento con corte 2023-12-31 → prueba ciega 2024+ (cifras honestas del procedimiento)
    p1 = run_procedure(EV, cut, test=True)
    honest = None
    if p1.get('test_table') is not None:
        honest = {'procedureCut': VAL_END, 'test': p1['test_table'], 'combined': p1.get('test_stats')}
    # 2) Procedimiento con TODOS los datos → registro que se despliega
    last_day = int(EV['day'].max())
    last_str = str(np.datetime64('1970-01-01') + np.timedelta64(last_day, 'D'))
    p2 = run_procedure(EV, last_day, test=False)
    reg = registry_json(p2, last_str, universe_n, honest, p2['names'] if p2.get('names') else [])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'validated.json')
    json.dump(reg, open(path, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'), default=float)
    print(f'\nregistro escrito: {path} ({os.path.getsize(path) / 1024:.0f} KB) · {len(reg["strategies"])} estrategias', flush=True)



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


def load_data(args):
    if args.synthetic:
        return synthetic_universe(args.synthetic, args.synthetic_ar)
    uni = UV.load()
    syms = list(uni.keys())
    if args.limit:
        keep = [s for s in syms if uni[s].get('aux')] + [s for s in syms if not uni[s].get('aux')][:args.limit]
        syms = keep
    bars, failed = D.fetch_many(syms, rng=args.range, workers=args.workers, cache_path=args.cache, max_age_h=args.max_age_h)
    uni = {s: uni[s] for s in syms if s in bars}
    return uni, bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['explore', 'meta', 'meta2', 'meta3', 'regime', 'final'])
    ap.add_argument('--range', default='10y')
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache', 'prices_10y.pkl.gz'))
    ap.add_argument('--max-age-h', type=float, default=24 * 14)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--synthetic', type=int, default=0)
    ap.add_argument('--synthetic-ar', type=float, default=0.0)
    ap.add_argument('--out', default=os.path.join(HERE, 'research_out'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    uni, data = load_data(args)
    print(f'universo con datos: {len([s for s in uni if not uni[s].get("aux")])} activos · {time.time() - t0:.0f}s', flush=True)
    variants = SM.default_variants()
    EV = build_events(data, uni, SU.SETUPS, variants)
    print(f'tabla de eventos lista · {time.time() - t0:.0f}s', flush=True)
    if args.cmd == 'explore':
        explore(EV, args.out)
    elif args.cmd == 'meta':
        meta_explore(EV, args.out)
    elif args.cmd == 'meta2':
        meta2_explore(EV, args.out)
    elif args.cmd == 'meta3':
        meta3_explore(EV, args.out)
    elif args.cmd == 'regime':
        regime_explore(EV, args.out)
    else:
        final(EV, args.out, len([s for s in uni if not uni[s].get('aux')]))
    print(f'\nfin · {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
