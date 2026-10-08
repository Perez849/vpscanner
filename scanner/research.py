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


# ═════════════════════════════════════════════════════════════════════════
#  Construcción de la tabla de eventos
# ═════════════════════════════════════════════════════════════════════════
def synthetic_universe(n=60):
    """Datos sintéticos para probar el pipeline sin red."""
    sys.path.insert(0, os.path.join(HERE, '..'))
    rng = np.random.default_rng(1)
    uni, data = {}, {}
    for k in range(n):
        N = 2600
        ret = rng.normal(0.0003, 0.016, N)
        c = 100 * np.exp(np.cumsum(ret))
        o = np.concatenate([[100], c[:-1]]) * np.exp(rng.normal(0, 0.006, N))
        h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.008, N)))
        l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.008, N)))
        v = rng.lognormal(16, 0.4, N)
        t = (np.datetime64('2016-01-04').astype('datetime64[s]').astype(np.int64) + np.arange(N) * 86400).astype(float)
        sym = f'SYN{k}'
        data[sym] = {'t': t, 'o': o, 'h': h, 'l': l, 'c': c, 'v': v}
        uni[sym] = {'yahoo': sym, 'group': GROUPS[k % 5]}
    spy = data['SYN0']
    data['SPY'] = spy
    data['^VIX'] = {**spy, 'c': 15 + 5 * np.abs(np.sin(np.arange(len(spy['c'])) / 50))}
    uni['SPY'] = {'yahoo': 'SPY', 'group': 'etf', 'aux': True}
    uni['^VIX'] = {'yahoo': '^VIX', 'group': 'index', 'aux': True}
    return uni, data


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
#  FINAL: procedimiento de selección + modelo de probabilidad
# ═════════════════════════════════════════════════════════════════════════
SEL = dict(min_n=400, min_syms=40, min_t_edge=2.5, min_edge=0.10, min_mean=0.20, min_wr=60.0,
           min_year_frac=0.70, min_group_n=80, min_margin=4.0, min_pf=1.20)
P_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def year_of(day: np.ndarray) -> np.ndarray:
    return (np.datetime64('1970-01-01') + day.astype('timedelta64[D]')).astype('datetime64[Y]').astype(int) + 1970


def select_cells(EV: Dict, mask: np.ndarray, sel: Dict = SEL, verbose: bool = True) -> List[Dict]:
    """Celdas (setup × salida) que superan los umbrales sobre `mask`. Una por setup."""
    setups_, variants = EV['setups'], EV['variants']
    sid, grp, day, sym = EV['sid'], EV['grp'], EV['day'], EV['sym']
    years = year_of(day)
    base_i = {+1: setups_.index('B_up'), -1: setups_.index('B_dn')}
    chosen: List[Dict] = []
    for si, sname in enumerate(setups_):
        st_ = SU.BY_ID[sname]
        if st_.family in ('baseline', 'control'):
            continue
        ms = (sid == si) & mask
        if ms.sum() < sel['min_n']:
            continue
        nsyms = len(np.unique(sym[ms]))
        if nsyms < sel['min_syms']:
            continue
        best = None
        for vi, vname in enumerate(variants):
            p = EV['pnl'][:, vi]
            st = stats(p[ms], day[ms], sym[ms])
            if st['n'] < sel['min_n']:
                continue
            mb = (sid == base_i[st_.dir]) & mask
            bs = stats(p[mb], day[mb], sym[mb])
            edge = st['mean'] - bs['mean']
            t_edge = edge / math.sqrt(st['se'] ** 2 + bs['se'] ** 2 + 1e-12)
            if not (st['wr'] >= sel['min_wr'] and st['mean'] >= sel['min_mean'] and edge >= sel['min_edge'] and t_edge >= sel['min_t_edge']):
                continue
            be = -st['avg_loss'] / (st['avg_win'] - st['avg_loss']) * 100      # acierto de equilibrio
            if st['wr_lo'] - be < sel['min_margin'] or st['pf'] < sel['min_pf']:
                continue
            # consistencia anual: % de años con media > 0 y ventaja > 0 frente al benchmark
            ok_y = tot_y = 0
            for y in np.unique(years[ms]):
                my = ms & (years == y)
                if my.sum() < 30:
                    continue
                by = (sid == base_i[st_.dir]) & mask & (years == y)
                tot_y += 1
                ok_y += (p[my][np.isfinite(p[my])].mean() > 0) and (p[my][np.isfinite(p[my])].mean() > p[by][np.isfinite(p[by])].mean())
            if tot_y < 3 or ok_y / tot_y < sel['min_year_frac']:
                continue
            cand = {'setup': sname, 'variant': vname, 'vi': vi, 'st': st, 'base': bs, 'edge': edge, 't_edge': t_edge,
                    'year_ok': f'{ok_y}/{tot_y}', 'nsyms': nsyms, 'be': be}
            if best is None or st['wr_lo'] > best['st']['wr_lo']:
                best = cand
        if best is None:
            continue
        # grupos en los que funciona
        vi = best['vi']; p = EV['pnl'][:, vi]
        groups_ok, by_group = [], {}
        for gi, gname in enumerate(GROUPS):
            mg = ms & (grp == gi)
            if mg.sum() < 20:
                continue
            sg = stats(p[mg], day[mg], sym[mg])
            by_group[gname] = {k: round(float(sg[k]), 2) for k in ('n', 'wr', 'mean')}
            if sg['n'] >= sel['min_group_n'] and sg['mean'] > 0.05 and sg['wr'] >= sel['min_wr'] - 5:
                groups_ok.append(gname)
        if not groups_ok:
            continue
        # recalcular estadísticos solo con los grupos válidos
        gm = ms & np.isin(grp, [GROUPS.index(g) for g in groups_ok])
        stg = stats(p[gm], day[gm], sym[gm])
        v = SM.default_variants()[vi]
        by_year = {}
        for y in np.unique(years[gm]):
            my = gm & (years == y)
            if my.sum() >= 30:
                sy = stats(p[my])
                by_year[str(int(y))] = {'n': sy['n'], 'wr': round(sy['wr'], 1), 'mean': round(sy['mean'], 2)}
        chosen.append({
            'id': f'{sname}|{best["variant"]}', 'setup': sname, 'label': st_.label, 'family': st_.family, 'dir': st_.dir,
            'exit': {'kind': v.kind, 'T': v.T, 'S': v.S, 'H': v.H}, 'groups': groups_ok, 'vi': vi,
            'stats': {'dev': {'n': stg['n'], 'wr': round(stg['wr'], 2), 'wr_lo': round(stg['wr_lo'], 2),
                               'mean': round(stg['mean'], 3), 'pf': round(stg['pf'], 3),
                               'avg_win': round(stg['avg_win'], 3), 'avg_loss': round(stg['avg_loss'], 3)},
                      'base': {'wr': round(best['base']['wr'], 2), 'mean': round(best['base']['mean'], 3)},
                      'edge': round(best['edge'], 3), 't_edge': round(best['t_edge'], 2), 'year_ok': best['year_ok'],
                      'byYear': by_year, 'byGroup': by_group}})
        if verbose:
            print(f"  ✔ {sname:13s} {best['variant']:20s} n={stg['n']:6d} WR={stg['wr']:5.1f}% μ={stg['mean']:+.2f}% edge={best['edge']:+.2f} t={best['t_edge']:.1f} margen={best['st']['wr_lo'] - best['be']:+.1f} años {best['year_ok']} grupos={groups_ok}", flush=True)
    return chosen


def train_model(EV: Dict, cells: List[Dict], train_mask: np.ndarray):
    """Entrena el modelo sobre los eventos de las celdas elegidas (etiqueta: gana con su propia salida)."""
    import model as MD
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    sel_rows, y, sid_m, gid = [], [], [], []
    cell_ids = [c['id'] for c in cells]
    for k, c in enumerate(cells):
        si = EV['setups'].index(c['setup'])
        m = (EV['sid'] == si) & train_mask & np.isin(EV['grp'], [GROUPS.index(g) for g in c['groups']])
        m &= np.isfinite(EV['pnl'][:, c['vi']])
        idx = np.flatnonzero(m)
        sel_rows.append(idx); y.append((EV['pnl'][idx, c['vi']] > 0).astype(np.float32))
        sid_m.append(np.full(len(idx), k)); gid.append(EV['grp'][idx])
    rows = np.concatenate(sel_rows); y = np.concatenate(y); sid_m = np.concatenate(sid_m); gid = np.concatenate(gid)
    mdl = MD.LogitModel(fidx, cell_ids, GROUPS)
    mdl.fit(EV['X'][rows], sid_m, gid, y)
    return mdl, rows, y, sid_m, gid


def walk_forward_oos(EV: Dict, cells: List[Dict], upto_day: int, first_year: int = 2019):
    """Predicciones fuera de muestra: para cada año Y, modelo entrenado solo con eventos anteriores."""
    import model as MD
    years = year_of(EV['day'])
    out_rows, out_p, out_y, out_sid = [], [], [], []
    last_year = int(year_of(np.array([upto_day]))[0])
    for Y in range(first_year, last_year + 1):
        trm = EV['day'] < day_of(f'{Y}-01-01')
        tem = (years == Y) & (EV['day'] <= upto_day)
        if trm.sum() < 5000 or tem.sum() == 0:
            continue
        mdl, *_ = train_model(EV, cells, trm)
        for k, c in enumerate(cells):
            si = EV['setups'].index(c['setup'])
            m = (EV['sid'] == si) & tem & np.isin(EV['grp'], [GROUPS.index(g) for g in c['groups']]) & np.isfinite(EV['pnl'][:, c['vi']])
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            p = mdl.predict_raw(EV['X'][idx], np.full(len(idx), k), EV['grp'][idx])
            out_rows.append(idx); out_p.append(p); out_y.append((EV['pnl'][idx, c['vi']] > 0).astype(np.float32)); out_sid.append(np.full(len(idx), k))
    if not out_rows:
        return None
    return (np.concatenate(out_rows), np.concatenate(out_p), np.concatenate(out_y), np.concatenate(out_sid))


def thr_table(title: str, p: np.ndarray, pnl: np.ndarray, day: np.ndarray, sym: np.ndarray, grid=P_GRID, base_wr: float | None = None):
    print(f'\n--- {title}')
    print(f"{'P>=':>5s} {'n':>7s} {'WR%':>6s} {'loWR':>6s} {'μ%':>7s} {'PF':>5s} {'avgW':>6s} {'avgL':>6s} {'alertas/día':>11s}")
    ndays = max(1, len(np.unique(day)))
    st0 = stats(pnl, day, sym)
    print(f"{'todas':>5s} {st0['n']:7d} {st0['wr']:6.1f} {st0['wr_lo']:6.1f} {st0['mean']:+7.2f} {st0['pf']:5.2f} {st0['avg_win']:+6.2f} {st0['avg_loss']:+6.2f} {st0['n']/ndays:11.2f}")
    res = {}
    for g in grid:
        m = p >= g
        if m.sum() < 30:
            continue
        st = stats(pnl[m], day[m], sym[m])
        res[g] = st
        print(f"{g:5.2f} {st['n']:7d} {st['wr']:6.1f} {st['wr_lo']:6.1f} {st['mean']:+7.2f} {st['pf']:5.2f} {st['avg_win']:+6.2f} {st['avg_loss']:+6.2f} {st['n']/ndays:11.2f}")
    return res


def run_procedure(EV: Dict, cutoff: str, evaluate_from: str | None, verbose: bool = True):
    """Aplica el procedimiento con datos hasta `cutoff`. Si evaluate_from, mide en test (>cutoff)."""
    cut = day_of(cutoff)
    mask = EV['day'] <= cut
    print(f'\n########  PROCEDIMIENTO con datos hasta {cutoff}  ########', flush=True)
    cells = select_cells(EV, mask, verbose=verbose)
    print(f'  → {len(cells)} celdas seleccionadas', flush=True)
    if not cells:
        return {'cells': [], 'model': None}
    # probabilidad: walk-forward para calibrar, modelo final sobre todo hasta cutoff
    oos = walk_forward_oos(EV, cells, cut)
    mdl, *_ = train_model(EV, cells, mask)
    info = {}
    if oos is not None:
        rows, p, y, k = oos
        import model as MD
        print(f'  walk-forward {len(rows):,} eventos OOS · AUC={MD.auc(p, y):.3f}')
        mdl.set_calibration(p, y)
        pc = np.interp(p, [c[0] for c in mdl.calib], [c[1] for c in mdl.calib]) if len(mdl.calib) >= 2 else p
        pnl_sel = np.array([EV['pnl'][r, cells[kk]['vi']] for r, kk in zip(rows, k)])
        base_wr = float(y.mean() * 100)
        info['wf'] = thr_table('Walk-forward DENTRO de muestra (años previos al corte) · probabilidad calibrada', pc, pnl_sel, EV['day'][rows], EV['sym'][rows])
    out = {'cells': cells, 'model': mdl, 'cut': cut}
    if evaluate_from:
        te = EV['day'] > cut
        rows_l, p_l, pnl_l, sid_l = [], [], [], []
        for kk, c in enumerate(cells):
            si = EV['setups'].index(c['setup'])
            m = (EV['sid'] == si) & te & np.isin(EV['grp'], [GROUPS.index(g) for g in c['groups']]) & np.isfinite(EV['pnl'][:, c['vi']])
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            rows_l.append(idx); sid_l.append(np.full(len(idx), kk)); pnl_l.append(EV['pnl'][idx, c['vi']])
            p_l.append(mdl.predict(EV['X'][idx], np.full(len(idx), kk), EV['grp'][idx]))
            st = stats(EV['pnl'][idx, c['vi']], EV['day'][idx], EV['sym'][idx])
            print(f"  TEST {c['id']:34s} n={st['n']:5d} WR={st['wr']:5.1f}% (DEV {c['stats']['dev']['wr']:.1f}%) μ={st['mean']:+.2f}% (DEV {c['stats']['dev']['mean']:+.2f}%)", flush=True)
        if rows_l:
            rows = np.concatenate(rows_l); p = np.concatenate(p_l); pnl = np.concatenate(pnl_l)
            info['test'] = thr_table('TEST CIEGO (después del corte) · todas las celdas · probabilidad calibrada', p, pnl, EV['day'][rows], EV['sym'][rows])
            # mejor día-a-día: máx. N alertas por día
            out['test_rows'] = (rows, p, pnl)
    out['info'] = info
    return out


def registry_json(proc: Dict, last_day_str: str, universe_n: int, honest: Dict | None) -> Dict:
    cells = []
    for c in proc['cells']:
        cc = {k: v for k, v in c.items() if k != 'vi'}
        cells.append(cc)
    reg = {'version': 2, 'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'dataThrough': last_day_str,
           'universe': universe_n, 'periods': {'trainEnd': TRAIN_END, 'valEnd': VAL_END},
           'rules': {'minP': 0.70, 'minEv': 0.15, 'watchP': 0.62, 'maxAlerts': 30},
           'summary': honest, 'cells': cells}
    if proc.get('model') is not None:
        reg['model'] = proc['model'].to_json()
    return reg


def final(EV: Dict, out_dir: str, universe_n: int):
    # 1) Procedimiento con corte en 2023-12-31 → test ciego 2024+
    p1 = run_procedure(EV, VAL_END, evaluate_from=VAL_END)
    honest = None
    if p1.get('info', {}).get('test'):
        t = p1['info']['test']
        honest = {'procedureCut': VAL_END, 'test': {str(k): {kk: round(float(vv), 3) for kk, vv in v.items() if kk in ('n', 'wr', 'wr_lo', 'mean', 'pf')} for k, v in t.items()}}
    # 2) Procedimiento con TODOS los datos → registro que se despliega
    last_day = int(EV['day'].max())
    last_str = str(np.datetime64('1970-01-01') + np.timedelta64(last_day, 'D'))
    p2 = run_procedure(EV, last_str, evaluate_from=None)
    reg = registry_json(p2, last_str, universe_n, honest)
    os.makedirs(os.path.join(HERE, 'model'), exist_ok=True)
    path = os.path.join(out_dir, 'validated.json')
    json.dump(reg, open(path, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'), default=float)
    print(f'\nregistro escrito: {path} ({os.path.getsize(path) / 1024:.0f} KB) · {len(reg["cells"])} celdas', flush=True)



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


def load_data(args):
    if args.synthetic:
        return synthetic_universe(args.synthetic)
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
    ap.add_argument('cmd', choices=['explore', 'meta', 'final'])
    ap.add_argument('--range', default='10y')
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache', 'prices_10y.pkl.gz'))
    ap.add_argument('--max-age-h', type=float, default=24 * 14)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--synthetic', type=int, default=0)
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
    else:
        final(EV, args.out, len([s for s in uni if not uni[s].get('aux')]))
    print(f'\nfin · {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
