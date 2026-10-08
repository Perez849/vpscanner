#!/usr/bin/env python3
"""
research.py — Construye la tabla de eventos del universo y VALIDA el sistema (comando `final`).

    python research.py final --out model        → escribe model/validated.json (lo usa scan.py)
    python research.py <cmd>                    → laboratorio exploratorio (ver lab.py)

Procedimiento de `final` (walk-forward, sin mirar nunca al futuro):
  1. Candidatos: unión de patrones de sobreventa en tendencia alcista (largos), un evento por (activo, día).
  2. Por cada plan de salida, para cada año Y≥2019 se entrena el modelo SOLO con los 3 años anteriores y se
     predice la probabilidad de acierto de los candidatos de Y (fuera de muestra).
  3. Política: de cada día, las N mejores señales cuya probabilidad supere la tasa base de acierto.
     Los grupos de activos sin evidencia (n<80 o media≤0) se excluyen automáticamente.
  4. Calibración isotónica de la probabilidad con esas predicciones fuera de muestra.
  5. Chequeo de salud: si los últimos 12 meses (fuera de muestra) no son rentables, el plan se PAUSA.
  6. El modelo desplegado se ajusta con los últimos 3 años de datos.
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

TRAIN_END = '2021-12-31'      # (solo laboratorio)
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


MOC_VARIANTS = [SM.Variant('rsi', 0.0, 4.0, 10), SM.Variant('atr', 1.0, 4.0, 10)]


def build_events(data: Dict, uni: Dict, setup_list: List[SU.Setup], variants: List[SM.Variant],
                 only_last: bool = False, verbose: bool = True, with_moc: bool = True):
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
    acc = {k: [] for k in ('sid', 'sym', 'grp', 'day', 'bi', 'X', 'pnl', 'kind', 'bars', 'gapin', 'moc')}
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
            # hueco de entrada (apertura de i+1 frente al cierre de i, en ATR) y entrada AL CIERRE para los planes desplegados
            nxt = np.minimum(uni_idx + 1, len(F['c']) - 1)
            gap_in = np.where(uni_idx + 1 < len(F['c']), (F['o'][nxt] - F['c'][uni_idx]) / F['atr'][uni_idx], np.nan).astype(np.float32)
            if with_moc:
                moc_res = SM.simulate(F, uni_idx, sgn, MOC_VARIANTS, cost, entry_mode='close')
                MO = np.stack([moc_res[v.name].pnl for v in MOC_VARIANTS], 1).astype(np.float32)
            else:
                MO = np.full((len(uni_idx), 2), np.nan, np.float32)
            for sid in ids:
                pos = np.searchsorted(uni_idx, ev[sid])
                acc['sid'].append(np.full(len(pos), sid_of[sid], np.int16))
                acc['sym'].append(np.full(len(pos), n_s, np.int32))
                acc['grp'].append(np.full(len(pos), GROUPS.index(g), np.int8))
                acc['day'].append((F['t'][uni_idx][pos] // 86400).astype(np.int32))
                acc['bi'].append(uni_idx[pos].astype(np.int32))
                acc['X'].append(Xs[pos]); acc['pnl'].append(P[pos])
                acc['kind'].append(K[pos]); acc['bars'].append(B[pos])
                acc['gapin'].append(gap_in[pos]); acc['moc'].append(MO[pos])
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


def portfolio_sim(cal: np.ndarray, ti: np.ndarray, bars: np.ndarray, res: np.ndarray, rank: np.ndarray, M: int, f: float) -> Dict:
    """
    Cartera con capital limitado. cal = días (enteros, ordenados) de la sesión de cada barra; ti = índice en cal de la sesión de la señal;
    bars = sesiones hasta la salida; res = resultado % neto de cada operación; rank = puesto en el ranking del día.
    Como mucho M posiciones abiertas, cada una el f del capital ACTUAL; las señales entran al día siguiente por orden de puesto.
    La curva de capital se actualiza al CERRAR cada operación (sin valorar a mercado), por lo que el DD queda algo subestimado.
    """
    ok = np.isfinite(res)
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
    taken = 0
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
            open_.append((t - 1 + int(max(bars[k], 1)), eq * f, float(res[k]))); taken += 1
        hist[t] = eq
    h = hist[t0:]
    peak = np.maximum.accumulate(h)
    rets = np.diff(h) / h[:-1]
    yrs_n = len(h) / 252.0
    mo = np.array([h[min(i + 21, len(h) - 1)] / h[i] - 1 for i in range(0, len(h) - 1, 21)])
    act = mo[np.abs(mo) > 1e-9]                                  # meses con alguna operación cerrada (un mes sin cierres no cuenta como positivo ni negativo)
    return {'cagr': float(h[-1] ** (1 / yrs_n) - 1) * 100, 'dd': float(((peak - h) / peak).max()) * 100,
            'sharpe': float(rets.mean() / (rets.std() + 1e-12) * math.sqrt(252)), 'monthsPos': float(np.mean(act > 0) * 100) if len(act) else float('nan'),
            'worstMonth': float(mo.min() * 100), 'tradesYear': float(taken / yrs_n), 'curve': h}


def split_masks(day: np.ndarray):
    tr, va = day_of(TRAIN_END), day_of(VAL_END)
    return day <= tr, (day > tr) & (day <= va), day > va


def fmt(s: Dict) -> str:
    if s.get('n', 0) == 0:
        return '        —'
    return f"n={s['n']:6d} WR={s['wr']:5.1f}% (lo {s['wr_lo']:4.1f}) μ={s['mean']:+6.2f}% PF={s['pf']:4.2f}"



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



def topn_mask(p, day, m, n):
    """Las n mejores señales por día (por probabilidad), dentro de la máscara m."""
    idx = np.flatnonzero(m)
    o = idx[np.lexsort((-p[idx], day[idx]))]
    d = day[o]
    first = np.concatenate([[0], np.flatnonzero(np.diff(d)) + 1])
    rank = np.arange(len(o)) - np.repeat(first, np.diff(np.concatenate([first, [len(o)]])))
    out = np.zeros(len(p), bool); out[o[rank < n]] = True
    return out




# ═════════════════════════════════════════════════════════════════════════
#  FINAL: planes de salida, política "N mejores por día", salud y registro
# ═════════════════════════════════════════════════════════════════════════
PLANS = [
    {'id': 'rsi_S4_H10', 'role': 'principal', 'label': 'Equilibrado', 'blurb': 'sale cuando el RSI(2) supera 70 (stop 4×ATR, máx. 10 sesiones)'},
    {'id': 'atr_T1_S4_H10', 'role': 'alta_prob', 'label': 'Alta probabilidad', 'blurb': 'objetivo +1×ATR (stop 4×ATR, máx. 10 sesiones): acierta más, gana menos por operación'},
]
N_PER_DAY = 5
WINDOW_YEARS = 3
FIRST_WF_YEAR = 2019
MIN_GROUP_N = 80
HEALTH_DAYS = 365


def wf_predict_linear(EV, rows, FL, y, ok, upto_day: int):
    """Probabilidad cruda OOS: para cada año Y, modelo lineal entrenado con los 3 años previos. Devuelve (p, base_rate_train)."""
    import model as MD
    day = EV['day'][rows]; years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    p = np.full(len(rows), np.nan); base = np.full(len(rows), np.nan)
    last_year = int(year_of(np.array([upto_day]))[0])
    for Y in range(FIRST_WF_YEAR, last_year + 1):
        trm = (day < day_of(f'{Y}-01-01')) & (day >= day_of(f'{Y - WINDOW_YEARS}-01-01')) & ok
        tem = (years == Y) & (day <= upto_day)
        if trm.sum() < 3000 or not tem.any():
            continue
        m = MD.LinearLogitModel(fidx, ['f%d' % i for i in range(FL.shape[1])], GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
        p[tem] = m.predict_raw(X[tem], FL[tem], gid[tem])
        base[tem] = float(y[trm].mean())
    return p, base


def select_policy(p, base, day, gid, ok, allowed_gids):
    """Las N mejores por día (probabilidad), entre las que superan la tasa base y están en grupos permitidos."""
    cand = ok & np.isfinite(p) & (p >= base) & np.isin(gid, allowed_gids)
    return topn_mask(p, day, cand, N_PER_DAY)


def _block(pnl, day, sym):
    st = stats(pnl, day, sym)
    return {k: round(float(st[k]), 3) for k in ('n', 'wr', 'wr_lo', 'mean', 'pf', 't_day', 'avg_win', 'avg_loss') if k in st}


def build_plan(EV, rows, FL, names, plan, upto_day: int, verbose=True):
    import model as MD
    variants = EV['variants']
    vi = variants.index(plan['id'])
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    fidx = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
    pnl = EV['pnl'][rows, vi]; ok = np.isfinite(pnl) & (day <= upto_day)
    y = (pnl > 0).astype(np.float32)
    p, base = wf_predict_linear(EV, rows, FL, y, ok, upto_day)
    m = ok & np.isfinite(p)
    print(f"\n  [{plan['id']} · {plan['label']}] AUC OOS={MD.auc(p[m], y[m]):.3f} · todos los candidatos: {fmt(stats(pnl[m], day[m], sym[m]))}", flush=True)
    # grupos permitidos: evidencia fuera de muestra con la política aplicada a todos los grupos
    sel0 = select_policy(p, base, day, gid, ok, list(range(len(GROUPS))))
    allowed = []
    for gi, gname in enumerate(GROUPS):
        mg = sel0 & (gid == gi)
        if mg.sum() >= MIN_GROUP_N:
            sg_ = stats(pnl[mg], day[mg], sym[mg])
            good = sg_['mean'] > 0.05
            print(f"       grupo {gname:9s} {fmt(sg_)} → {'ok' if good else 'EXCLUIDO'}")
            if good:
                allowed.append(gi)
        elif mg.sum() > 0:
            print(f"       grupo {gname:9s} n={mg.sum()} (<{MIN_GROUP_N}: sin evidencia) → EXCLUIDO")
    sel = select_policy(p, base, day, gid, ok, allowed)
    st = stats(pnl[sel], day[sel], sym[sel])
    print(f"     política top {N_PER_DAY}/día en {[GROUPS[g] for g in allowed]}: {fmt(st)} · t={st['t_day']:.1f}", flush=True)
    # por año, por grupo, reciente, sensibilidad a costes ×2
    by_year = {}
    for Y in sorted(set(years[sel])):
        my = sel & (years == Y)
        if my.sum() >= 20:
            sy = stats(pnl[my]); by_year[str(int(Y))] = {'n': sy['n'], 'wr': round(sy['wr'], 1), 'mean': round(sy['mean'], 2)}
    print('     por año: ' + ' '.join(f"{y_}:{v['wr']:.0f}%/{v['mean']:+.2f}" for y_, v in by_year.items()))
    by_group = {GROUPS[g]: _block(pnl[sel & (gid == g)], day[sel & (gid == g)], sym[sel & (gid == g)]) for g in allowed if (sel & (gid == g)).sum() >= 20}
    cost_arr = np.array([SM.COST_RT.get(GROUPS[g], 0.30) for g in range(len(GROUPS))])[gid]
    st2 = stats(pnl[sel] - cost_arr[sel], day[sel], sym[sel])
    recent = sel & (day >= day_of('2024-01-01'))
    st_rec = stats(pnl[recent], day[recent], sym[recent]) if recent.sum() >= 50 else None
    if st_rec:
        print(f"     desde 2024: {fmt(st_rec)}")
    print(f"     con costes DOBLES: WR={st2['wr']:.1f}% μ={st2['mean']:+.2f}% PF={st2['pf']:.2f}")
    # salud: últimos 12 meses con resultado conocido
    last_known = int(day[sel].max()) if sel.any() else upto_day
    hm = sel & (day > last_known - HEALTH_DAYS)
    st_h = stats(pnl[hm], day[hm], sym[hm]) if hm.sum() >= 30 else None
    paused = bool(st_h and st_h['n'] >= 200 and st_h['mean'] <= 0)
    print(f"     salud (últimos 12 meses): {fmt(st_h) if st_h else 'datos insuficientes'} → {'PAUSADO' if paused else 'activo'}", flush=True)
    # calibración isotónica (todas las predicciones OOS) y modelo desplegado (últimos 3 años)
    cal = MD.LogitModel(fidx, names, GROUPS); cal.set_calibration(p[m], y[m])
    trm = (day >= upto_day - 365 * WINDOW_YEARS) & (day <= upto_day) & ok
    fm = MD.LinearLogitModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
    fm.calib = cal.calib
    base_now = float(y[trm].mean())
    pc = np.interp(p, [c[0] for c in cal.calib], [c[1] for c in cal.calib]) if len(cal.calib) >= 2 else p
    v = SM.default_variants()[vi]
    return {
        'id': plan['id'], 'role': plan['role'], 'label': plan['label'], 'blurb': plan['blurb'],
        'exit': {'kind': v.kind, 'T': v.T, 'S': v.S, 'H': v.H}, 'groups': [GROUPS[g] for g in allowed], 'gids': allowed,
        'nPerDay': N_PER_DAY, 'floorRaw': round(base_now, 4), 'paused': paused,
        'ev': {'aw': round(st['avg_win'], 3), 'al': round(st['avg_loss'], 3)},
        'stats': {'oos': _block(pnl[sel], day[sel], sym[sel]), 'recent': _block(pnl[recent], day[recent], sym[recent]) if st_rec else None,
                  'cost2x': {k: round(float(st2[k]), 3) for k in ('wr', 'mean', 'pf')}, 'health': _block(pnl[hm], day[hm], sym[hm]) if st_h else None,
                  'baseWR': round(float(y[m].mean() * 100), 1), 'meanPCal': round(float(pc[sel].mean() * 100), 1), 'byYear': by_year, 'byGroup': by_group},
        'model': fm}


# ═════════════════════════════════════════════════════════════════════════
#  PELOTAZOS (experimental): cola gruesa, baja tasa de acierto
# ═════════════════════════════════════════════════════════════════════════
PEL_PLAN = {'id': 'pel_trl_T5_S5_H60', 'variant': 'trl_T5_S5_H60', 'label': 'Pelotazo (experimental)',
            'blurb': 'trailing stop de 5×ATR hasta 60 sesiones: acierta ~48%, la mediana pierde, pero ~1 de cada 4 operaciones supera +20%'}
PEL_VARIANTS = [SM.Variant('trl', 5.0, 5.0, 60)]
PEL_GROUPS = ['us_large', 'us_mid', 'us_small']       # por diseño: temáticos y cripto son listas elegidas a posteriori (sesgo extremo)
PEL_N_PER_DAY = 3
PEL_BIG = 12.0
PEL_EXTRA = ['ret60', 'ret120', 'rs60', 'rs120', 'max_gap5', 'shock5', 'max_vr5']


def build_pelotazo(EV: Dict, upto_day: int):
    import model as MD
    vi = EV['variants'].index(PEL_PLAN['variant'])
    rows, FL, names = union_events(EV, +1)
    day, sym = EV['day'][rows], EV['sym'][rows]
    years = year_of(day)
    X = EV['X'][rows]; gid = EV['grp'][rows]
    feats_ = list(MD.MODEL_FEATURES) + PEL_EXTRA
    fidx = [FEATURES.index(f) for f in feats_]
    pnl = EV['pnl'][rows, vi]; bars = EV['bars'][rows, vi]
    ok = np.isfinite(pnl) & (day <= upto_day)
    allowed = [GROUPS.index(g) for g in PEL_GROUPS]
    ing = np.isin(gid, allowed)
    y = (pnl >= PEL_BIG).astype(np.float32)
    p = np.full(len(rows), np.nan); base = np.full(len(rows), np.nan)
    last_year = int(year_of(np.array([upto_day]))[0])
    for Y in range(FIRST_WF_YEAR, last_year + 1):
        trm = (day < day_of(f'{Y}-01-01')) & (day >= day_of(f'{Y - WINDOW_YEARS}-01-01')) & ok & ing
        tem = (years == Y) & (day <= upto_day) & ing
        if trm.sum() < 3000 or not tem.any():
            continue
        m1 = MD.LinearLogitModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
        p[tem] = m1.predict_raw(X[tem], FL[tem], gid[tem])
        base[tem] = float(y[trm].mean())
    m = ok & ing & np.isfinite(p)
    cand = m & (p >= base)
    sel = topn_mask(p, day, cand, PEL_N_PER_DAY) if cand.sum() else np.zeros(len(rows), bool)
    if sel.sum() < 300:
        print(f'\n  [{PEL_PLAN["id"]}] sin eventos suficientes ({int(sel.sum())}): no se despliega el plan de pelotazos', flush=True)
        return None
    x = pnl[sel]
    st = stats(x, day[sel], sym[sel])
    print(f"\n  [{PEL_PLAN['id']}] AUC(≥{PEL_BIG:.0f}%) OOS={MD.auc(p[m], y[m]):.3f} · base {y[m].mean() * 100:.1f}% · todos los candidatos: {fmt(stats(pnl[m], day[m], sym[m]))}")
    print(f"     política top {PEL_N_PER_DAY}/día en {PEL_GROUPS}: {fmt(st)} · t={st['t_day']:.1f} · mediana {np.median(x):+.2f}% · P≥20%={np.mean(x >= 20) * 100:.0f}% · duración {np.nanmean(bars[sel]):.0f} ses.", flush=True)
    by_year = {}
    for Y in sorted(set(years[sel])):
        my = sel & (years == Y)
        if my.sum() >= 20:
            sy = stats(pnl[my]); by_year[str(int(Y))] = {'n': sy['n'], 'wr': round(sy['wr'], 1), 'mean': round(sy['mean'], 2), 'p20': round(float(np.mean(pnl[my] >= 20) * 100), 1)}
    print('     por año: ' + ' '.join(f"{y_}:{v['mean']:+.1f}%" for y_, v in by_year.items()))
    by_group = {}
    for g in PEL_GROUPS:
        mg = sel & (gid == GROUPS.index(g))
        if mg.sum() >= 30:
            sg_ = stats(pnl[mg]); by_group[g] = {'n': sg_['n'], 'wr': round(sg_['wr'], 1), 'mean': round(sg_['mean'], 2)}
    mu = float(np.mean(x))
    haircut = {f'{int(f * 100)}%': round((1 - f) * mu + f * -60.0, 2) for f in (0.02, 0.05, 0.08)}
    cost_arr = np.array([SM.COST_RT.get(GROUPS[g], 0.30) for g in range(len(GROUPS))])[gid]
    # benchmark honesto: entrar al azar sobre la SMA200 con la misma salida
    bi = EV['setups'].index('B_up')
    bmask = (EV['sid'] == bi) & np.isfinite(EV['pnl'][:, vi]) & np.isin(EV['grp'], allowed)
    bench = float(np.mean(EV['pnl'][bmask, vi])) if bmask.sum() else float('nan')
    recent = sel & (day >= day_of('2024-01-01'))
    last_known = int(day[sel].max()) if sel.any() else upto_day
    hm = sel & (day > last_known - HEALTH_DAYS)
    st_h = stats(pnl[hm], day[hm], sym[hm]) if hm.sum() >= 30 else None
    paused = bool(st_h and st_h['n'] >= 150 and st_h['mean'] <= 0)
    print(f"     benchmark (entrar al azar sobre SMA200, misma salida): μ={bench:+.2f}% · descuento por quiebras: {haircut} · costes dobles: μ={np.mean(x - cost_arr[sel]):+.2f}%")
    print(f"     salud (12 meses): {fmt(st_h) if st_h else 'datos insuficientes'} → {'PAUSADO' if paused else 'activo'}", flush=True)
    cal = MD.LogitModel(fidx, names, GROUPS); cal.set_calibration(p[sel], y[sel])
    trm = (day >= upto_day - 365 * WINDOW_YEARS) & (day <= upto_day) & ok & ing
    fm = MD.LinearLogitModel(fidx, names, GROUPS).fit(X[trm], FL[trm], gid[trm], y[trm])
    fm.calib = cal.calib
    v = PEL_VARIANTS[0]
    return {
        'id': PEL_PLAN['id'], 'label': PEL_PLAN['label'], 'blurb': PEL_PLAN['blurb'], 'role': 'pelotazo',
        'exit': {'kind': v.kind, 'T': v.T, 'S': v.S, 'H': v.H}, 'groups': PEL_GROUPS, 'nPerDay': PEL_N_PER_DAY,
        'floorRaw': round(float(y[trm].mean()), 4), 'bigPct': PEL_BIG, 'paused': paused,
        'patterns': [{'id': n, 'label': SU.BY_ID[n].label} for n in names],
        'stats': {'oos': {**_block(x, day[sel], sym[sel]), 'median': round(float(np.median(x)), 2), 'p10': round(float(np.mean(x >= 10) * 100), 1),
                          'p20': round(float(np.mean(x >= 20) * 100), 1), 'p30': round(float(np.mean(x >= 30) * 100), 1), 'p95': round(float(np.percentile(x, 95)), 1),
                          'hold': round(float(np.nanmean(bars[sel])), 1)},
                  'recent': _block(pnl[recent], day[recent], sym[recent]) if recent.sum() >= 50 else None, 'benchmark': round(bench, 2),
                  'haircut': haircut, 'cost2x': round(float(np.mean(x - cost_arr[sel])), 2),
                  'health': _block(pnl[hm], day[hm], sym[hm]) if st_h else None, 'byYear': by_year, 'byGroup': by_group,
                  'pBigCal': round(float(y[sel].mean() * 100), 1)},
        'model': fm}


def final(EV: Dict, EV_pel: Dict | None, out_dir: str, universe_n: int):
    rows, FL, names = union_events(EV, +1)
    last_day = int(EV['day'].max())
    last_str = str(np.datetime64('1970-01-01') + np.timedelta64(last_day, 'D'))
    print(f'\n########  VALIDACIÓN FINAL · {len(rows):,} candidatos · datos hasta {last_str}  ########', flush=True)
    plans = [build_plan(EV, rows, FL, names, pl, last_day) for pl in PLANS]
    pel = build_pelotazo(EV_pel, last_day) if EV_pel is not None else None
    reg = {'version': 5, 'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'dataThrough': last_str, 'universe': universe_n,
           'design': {'windowYears': WINDOW_YEARS, 'firstWalkForwardYear': FIRST_WF_YEAR, 'nPerDay': N_PER_DAY, 'healthDays': HEALTH_DAYS},
           'rules': {'nPerDay': N_PER_DAY, 'watchMargin': 0.03},
           'patterns': [{'id': n, 'label': SU.BY_ID[n].label} for n in names],
           'strategies': [{**{k: v for k, v in pl.items() if k not in ('model', 'gids')}, 'model': pl['model'].to_json()} for pl in plans],
           'pelotazo': ({**{k: v for k, v in pel.items() if k != 'model'}, 'model': pel['model'].to_json()} if pel else None)}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'validated.json')
    json.dump(reg, open(path, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'), default=float)
    print(f'\nregistro escrito: {path} ({os.path.getsize(path) / 1024:.0f} KB) · {len(plans)} planes · pausados: {[p["id"] for p in plans if p["paused"]]}', flush=True)


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
    n_ok = len([s for s in uni if not uni[s].get('aux')])
    if not args.limit and n_ok < 0.7 * (len(syms)):
        raise SystemExit(f'Descarga incompleta ({n_ok}/{len(syms)} activos): se aborta para no sobrescribir el registro con datos pobres.')
    return uni, bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['final', 'explore', 'meta', 'meta2', 'meta3', 'meta4', 'meta5', 'regime', 'gap', 'pelotazo', 'pelotazo2', 'moc', 'improve', 'improve2', 'index'])
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
    if args.cmd == 'index':
        import lab
        lab.index_check(uni, data, args.out)
        print(f'\nfin · {time.time() - t0:.0f}s', flush=True)
        return
    if args.cmd == 'moc':
        import lab
        lab.moc_check(uni, data, args.out)
        print(f'\nfin · {time.time() - t0:.0f}s', flush=True)
        return
    if args.cmd in ('pelotazo', 'pelotazo2'):
        import lab
        variants = lab.PEL_VARIANTS
        EV = build_events(data, uni, SU.PEL_SETUPS + [SU.BY_ID['B_up']], variants, with_moc=False)
    else:
        variants = SM.default_variants()
        EV = build_events(data, uni, SU.SETUPS, variants)
    print(f'tabla de eventos lista · {time.time() - t0:.0f}s', flush=True)
    if args.cmd == 'final':
        EV_pel = build_events(data, uni, SU.PEL_SETUPS + [SU.BY_ID['B_up']], PEL_VARIANTS, with_moc=False)
        final(EV, EV_pel, args.out, len([s for s in uni if not uni[s].get('aux')]))
    elif args.cmd in ('improve', 'improve2'):
        import lab
        getattr(lab, args.cmd + '_explore')(EV, args.out, data, uni)
    else:
        import lab
        getattr(lab, {'explore': 'explore', 'meta': 'meta_explore', 'meta2': 'meta2_explore', 'meta3': 'meta3_explore',
                      'meta4': 'meta4_explore', 'meta5': 'meta5_explore', 'regime': 'regime_explore', 'gap': 'gap_explore', 'pelotazo': 'pelotazo_explore', 'pelotazo2': 'pelotazo2_explore'}[args.cmd])(EV, args.out)
    print(f'\nfin · {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

