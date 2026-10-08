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

FEATURES = ['rsi2', 'rsi3', 'rsi14', 'ibs', 'ret1', 'ret2', 'ret3', 'ret5', 'ret10', 'ret20', 'atrp', 'atr_rel',
            'dist200', 'dist50', 'dist20', 'slope200', 'dd20', 'dd52', 'pos52', 'bbz', 'vol_ratio',
            'dn_streak', 'up_streak', 'gap',
            'spy_up', 'spy_dist200', 'spy_rsi2', 'spy_ret5', 'spy_dd60', 'vix', 'vix_z', 'vix_chg5',
            'vp_poc_atr', 'vp_val_atr', 'vp_pos']
GROUPS = ['us_large', 'us_mid', 'us_small', 'thematic', 'etf', 'crypto', 'fx', 'futures', 'eu', 'index']


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
    acc = {k: [] for k in ('sid', 'sym', 'grp', 'day', 'X', 'pnl', 'kind', 'bars')}
    t0 = time.time()
    for n_s, sym in enumerate(sym_list):
        g = uni[sym]['group']
        F = feats.build(data[sym])
        if reg is not None:
            F.update(feats.align_regime(reg, F['t']))
        else:
            for k in ('spy_up', 'spy_dist200', 'spy_rsi2', 'spy_ret5', 'spy_dd60', 'vix', 'vix_z', 'vix_chg5'):
                F[k] = np.full(len(F['c']), np.nan)
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
            poc, vah, val = feats.vp_levels(F, uni_idx)
            atr = F['atr'][uni_idx]
            c = F['c'][uni_idx]
            with np.errstate(invalid='ignore', divide='ignore'):
                vp_poc_atr = (poc - c) / atr
                vp_val_atr = (c - val) / atr
                vp_pos = (c - val) / (vah - val)
            Xs = np.stack([F[f][uni_idx] if f in F else np.full(len(uni_idx), np.nan) for f in FEATURES[:-3]]
                          + [vp_poc_atr, vp_val_atr, vp_pos], 1).astype(np.float32)
            for sid in ids:
                pos = np.searchsorted(uni_idx, ev[sid])
                acc['sid'].append(np.full(len(pos), sid_of[sid], np.int16))
                acc['sym'].append(np.full(len(pos), n_s, np.int32))
                acc['grp'].append(np.full(len(pos), GROUPS.index(g), np.int8))
                acc['day'].append((F['t'][uni_idx][pos] // 86400).astype(np.int32))
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
    ap.add_argument('cmd', choices=['explore', 'final'])
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
    print(f'\nfin · {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
