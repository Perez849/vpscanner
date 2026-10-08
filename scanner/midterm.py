#!/usr/bin/env python3
"""
midterm.py — Estrategias a MEDIO plazo (semanas–meses) frente al sistema de rebotes de corto plazo (comando `midterm` de research.py).
NO interviene en producción.

Pregunta del usuario: ¿más tiempo = menos ruido y más fiabilidad? Se prueban, con parámetros FIJADOS de antemano (sin optimizar):

  A. Tendencia en índices (SPY/QQQ/IWM/EFA): invertido solo si el precio está sobre su media de 200 sesiones; si no, en liquidez (BIL).
  B. Rotación de ETF por momentum (sectores, países, bonos, oro): cada mes, los K con mejor momentum Y positivo; si no, liquidez.
  C. Momentum de acciones (12-1 meses) en S&P 500: cada mes, las K mejores. OJO: sesgo de supervivencia (componentes actuales); se compara con
     la media igual-ponderada del mismo universo, que arrastra el mismo sesgo.
  D. Combinación con el sistema de rebotes (Equilibrado): ¿mejora la cartera conjunta?

Reglas honestas: la señal se calcula con el cierre del último día del mes y se ejecuta al cierre del DÍA SIGUIENTE (1 sesión de retraso);
costes por rotación (ETF 0,10 % ida y vuelta, acciones 0,15 %); la liquidez rinde lo que BIL; todos los resultados son netos.
"""
from __future__ import annotations
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

COST_ETF = 0.0005          # por lado (0,10 % ida y vuelta)
COST_STK = 0.00075         # por lado (0,15 % ida y vuelta)

ROT_ETFS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC',
            'TLT', 'IEF', 'HYG', 'GLD', 'DBC', 'VNQ']


def _px(data: Dict, syms: List[str], cal: np.ndarray) -> pd.DataFrame:
    idx = pd.Index(cal)
    cols = {}
    for s in syms:
        b = data.get(s)
        if b is None:
            continue
        d = (b['t'] // 86400).astype(np.int64)
        ser = pd.Series(b['c'], index=d)
        ser = ser[~ser.index.duplicated(keep='last')]
        cols[s] = ser.reindex(idx).ffill(limit=5)
    return pd.DataFrame(cols, index=idx)


def metrics(ret: np.ndarray, days: np.ndarray) -> Dict:
    ret = np.nan_to_num(ret)
    eq = np.cumprod(1 + ret)
    n = len(ret)
    yrs_n = n / 252.0
    dt = pd.to_datetime(days * 86400, unit='s')
    s = pd.Series(eq, index=dt)
    m = s.resample('ME').last().pct_change().dropna()
    y = s.resample('YE').last().pct_change()
    y.iloc[0] = s.resample('YE').last().iloc[0] / 1.0 - 1 if len(y) else np.nan
    return {'cagr': float(eq[-1] ** (1 / yrs_n) - 1) * 100, 'vol': float(ret.std() * math.sqrt(252) * 100),
            'sharpe': float(ret.mean() / (ret.std() + 1e-12) * math.sqrt(252)), 'dd': float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
            'monthsPos': float((m > 0).mean() * 100), 'worstMonth': float(m.min() * 100), 'years': {int(k.year): float(v * 100) for k, v in y.items()}}


def simulate(rets: np.ndarray, targets: Dict[int, np.ndarray], cash: int, cost: float):
    """Cartera con rebalanceos en los días `targets` (índice del día en que ya rigen los nuevos pesos). Devuelve (rentabilidad diaria, índice de inicio)."""
    T, N = rets.shape
    w = np.zeros(N); w[cash] = 1.0
    v = 1.0
    out = np.zeros(T)
    start = min(targets) if targets else T
    for t in range(start, T):
        if t in targets:
            tw = targets[t]
            v *= (1 - float(np.abs(tw - w).sum()) * cost)
            w = tw.copy()
        r = np.nan_to_num(rets[t])
        rp = float((w * r).sum())
        out[t] = rp
        v_new = v * (1 + rp)
        w = w * (1 + r) / (1 + rp)
        out[t] = v_new / v - 1
        v = v_new
    return out, start


def month_ends(days: np.ndarray) -> np.ndarray:
    dt = pd.to_datetime(days * 86400, unit='s')
    df = pd.DataFrame({'i': np.arange(len(days)), 'ym': dt.year * 100 + dt.month})
    return df.groupby('ym')['i'].max().to_numpy()


def _line(tag: str, m: Dict, spy: Dict | None = None) -> None:
    print(f"   {tag:46s} anual {m['cagr']:+6.1f}% · vol {m['vol']:4.1f}% · Sharpe {m['sharpe']:4.2f} · caída máx. {m['dd']:6.1f}% · meses+ {m['monthsPos']:3.0f}% · peor mes {m['worstMonth']:+6.1f}%", flush=True)


def etf_suite(data: Dict, cal: np.ndarray, cash_sym: str, pool: List[str], cost: float = COST_ETF):
    """Tendencia en índices, rotación por momentum y momentum dual, sobre el calendario `cal` (días del S&P 500)."""
    T = len(cal)
    syms = [s for s in dict.fromkeys(pool + [cash_sym, 'SPY']) if s in data]
    px = _px(data, syms, cal)
    rets = px.pct_change().to_numpy()
    cols = list(px.columns)
    ci = {c: i for i, c in enumerate(cols)}
    cash = ci[cash_sym]
    me = month_ends(cal)
    sma200 = px.rolling(200, min_periods=200).mean()
    above = (px > sma200).to_numpy()
    pv = px.to_numpy()
    res: Dict[str, np.ndarray] = {}
    starts: Dict[str, int] = {}

    def trend(assets: List[str]):
        tg, prev = {}, None
        for t in range(201, T):
            ok_ = [a for a in assets if a in ci and above[t - 2, ci[a]]]       # señal con el cierre de t-2, se ejecuta al cierre de t-1: rige desde t
            key = tuple(sorted(ok_))
            if key != prev:
                w = np.zeros(len(cols))
                for a in ok_:
                    w[ci[a]] = 1.0 / len(assets)
                w[cash] = 1.0 - w.sum()
                tg[t] = w; prev = key
        return tg

    def rotation(look, K, assets=None):
        assets = [a for a in (assets or pool) if a in ci]
        lbs = list(look) if isinstance(look, (list, tuple)) else [look]
        tg = {}
        for i in me:
            if i < 260 or i + 2 >= T:
                continue
            sc = {}
            for a in assets:
                p = pv[:, ci[a]]
                if all(np.isfinite(p[i]) and np.isfinite(p[i - lb]) and p[i - lb] > 0 for lb in lbs):
                    sc[a] = float(np.mean([p[i] / p[i - lb] - 1 for lb in lbs]))
            best = sorted([a for a in sc if sc[a] > 0.0], key=lambda a: -sc[a])[:K]
            w = np.zeros(len(cols))
            for a in best:
                w[ci[a]] = 1.0 / K
            w[cash] = 1.0 - w.sum()
            tg[i + 2] = w
        return tg

    def sim(tg):
        r, st = simulate(rets, tg, cash, cost)
        return r, st
    for nm, assets in (('Tendencia SPY (sobre su media de 200)', ['SPY']), ('Tendencia QQQ', ['QQQ']),
                       ('Tendencia 4 índices (SPY, QQQ, IWM, EFA)', ['SPY', 'QQQ', 'IWM', 'EFA'])):
        if all(a in ci for a in assets):
            res[nm], starts[nm] = sim(trend(assets))
    for look, K in (([63, 126, 252], 3), (126, 3)):
        nm = f"Rotación ETF momentum {look if isinstance(look, int) else 'media 3-6-12m'}, {K} mejores"
        res[nm], starts[nm] = sim(rotation(look, K))
    if all(a in ci for a in ('SPY', 'EFA', 'IEF')):
        res['Momentum dual clásico (SPY/EFA/IEF, 12 meses)'], starts['Momentum dual clásico (SPY/EFA/IEF, 12 meses)'] = sim(rotation(252, 1, ['SPY', 'EFA', 'IEF']))
    grid = {}
    for look in (63, 126, 252, [63, 126, 252]):
        for K in (2, 3, 5):
            r, st = sim(rotation(look, K))
            grid[(str(look), K)] = metrics(r[st:], cal[st:])
    return res, starts, grid, px


def _report_suite(title, res, starts, grid, cal, r_spy, s0, years_keys):
    print(f'\n=== {title} (periodo común {pd.to_datetime(cal[s0] * 86400, unit="s").date()} → hoy; todo neto de costes) ===')
    _line('S&P 500 comprar y mantener', metrics(r_spy[s0:], cal[s0:]))
    for nm, r in res.items():
        _line(nm, metrics(r[s0:], cal[s0:]))
    ms = metrics(r_spy[s0:], cal[s0:])
    mm = {nm: metrics(r[s0:], cal[s0:]) for nm, r in res.items()}
    ks = [k for k in years_keys if k in mm]
    print('   año   ' + ' '.join(f"{k[:16]:>16s}" for k in ['S&P 500'] + ks))
    for y_ in sorted(ms['years']):
        print(f"   {y_}  " + ' '.join(f"{(ms if k == 'S&P 500' else mm[k])['years'].get(y_, float('nan')):+16.1f}" for k in ['S&P 500'] + ks))
    print('   sensibilidad de la rotación (anual / Sharpe / caída máx.; no se elige la mejor, se muestra la dispersión):')
    for (look, K), m in grid.items():
        print(f"     momentum {look:14s} K={K}: {m['cagr']:+5.1f}% / {m['sharpe']:.2f} / {m['dd']:.0f}%")


def run(EV, out_dir: str, data: Dict, uni: Dict):
    spy = data['SPY']
    cal = (spy['t'] // 86400).astype(np.int64)
    T = len(cal)
    from research import day_of
    print(f'\n######## MEDIO PLAZO · calendario S&P 500 {T} sesiones ({pd.to_datetime(cal[0] * 86400, unit="s").date()} → {pd.to_datetime(cal[-1] * 86400, unit="s").date()})', flush=True)
    cash_sym = 'BIL' if 'BIL' in data else ('SHY' if 'SHY' in data else None)
    print(f'  liquidez: {cash_sym} · ETF disponibles: {[s for s in ROT_ETFS if s in data]}', flush=True)
    r_spy = np.nan_to_num(pd.Series(spy['c']).pct_change().to_numpy())
    res, starts, grid, px_etf = etf_suite(data, cal, cash_sym, ROT_ETFS)
    i19 = int(np.searchsorted(cal, day_of('2019-01-01')))
    names_main = ['Tendencia SPY (sobre su media de 200)', 'Rotación ETF momentum media 3-6-12m, 3 mejores', 'Momentum dual clásico (SPY/EFA/IEF, 12 meses)']
    s0 = max(max(starts.values()), i19)
    _report_suite('A y B · ETF, 10 AÑOS DE DATOS', res, starts, grid, cal, r_spy, s0, names_main)

    # ── historia LARGA (índices y ETF desde 2002–2005): incluye 2008 ──────
    long_res = None
    try:
        import data as D
        pool = [s for s in ROT_ETFS if s not in ('XLRE', 'XLC')] + ['SHY']
        bars, failed = D.fetch_many(list(dict.fromkeys(pool + ['SPY'])), workers=6, cache_path=None, verbose=False,
                                    period1=int(pd.Timestamp('2002-01-02').timestamp()))
        calL = (bars['SPY']['t'] // 86400).astype(np.int64)
        gaps = np.diff(calL)
        if np.median(gaps) > 3 or gaps.max() > 12:
            raise RuntimeError(f'barras no diarias (mediana {np.median(gaps)} días, máximo {gaps.max()})')
        r_spyL = np.nan_to_num(pd.Series(bars['SPY']['c']).pct_change().to_numpy())
        resL, startsL, gridL, _ = etf_suite(bars, calL, 'SHY', pool)
        i05 = int(np.searchsorted(calL, day_of('2005-01-01')))
        s0L = max(max(startsL.values()), i05)
        print(f"\n  historia larga: SPY desde {pd.to_datetime(calL[0] * 86400, unit='s').date()} · ETF con datos: {sorted(bars)}")
        _report_suite('A y B · ETF, HISTORIA LARGA (incluye 2008)', resL, startsL, gridL, calL, r_spyL, s0L, names_main)
    except Exception as e:                                                   # pragma: no cover
        print(f'  (historia larga no disponible: {e})')

    # ── C. momentum de acciones (S&P 500, historial completo) ────────────
    stk = [s for s in uni if uni[s]['group'] == 'us_large' and not uni[s].get('aux') and s in data]
    full = [s for s in stk if (data[s]['t'][0] // 86400) <= cal[0] + 40]
    print(f'\n  acciones S&P 500 con historial completo: {len(full)} de {len(stk)} (se excluyen las de salida a bolsa reciente para recortar sesgo)', flush=True)
    px_s = _px(data, full, cal)
    ret_s = px_s.pct_change().to_numpy()
    cash_col = np.nan_to_num(px_etf[cash_sym].pct_change().to_numpy())
    rets_s = np.column_stack([ret_s, cash_col])
    ns = ret_s.shape[1]
    sma_s = px_s.rolling(200, min_periods=150).mean().to_numpy()
    pv = px_s.to_numpy()
    me = month_ends(cal)

    def stock_mom(look, skip, K):
        tg = {}
        for i in me:
            if i < 300 or i + 2 >= T:
                continue
            a, b = pv[i - skip], pv[i - look]
            sc = np.where(np.isfinite(a) & np.isfinite(b) & (b > 0) & (pv[i] > sma_s[i]), a / b - 1, np.nan)
            ok_ = np.flatnonzero(np.isfinite(sc))
            w = np.zeros(ns + 1)
            if len(ok_) >= K:
                w[ok_[np.argsort(-sc[ok_])[:K]]] = 1.0 / K
            w[ns] = max(0.0, 1.0 - w[:ns].sum())
            tg[i + 2] = w
        return tg

    def ew_universe():
        tg = {}
        for i in me:
            if i < 300 or i + 2 >= T:
                continue
            ok_ = np.flatnonzero(np.isfinite(pv[i]) & np.isfinite(pv[i - 21]))
            w = np.zeros(ns + 1)
            w[ok_] = 1.0 / len(ok_)
            tg[i + 2] = w
        return tg
    r_ew, st_ew = simulate(rets_s, ew_universe(), ns, COST_STK)
    sg = {(look, K): simulate(rets_s, stock_mom(look, 21, K), ns, COST_STK) for look in (126, 252) for K in (20, 30, 50)}
    r_m, st_m = sg[(252, 30)]
    s1 = max(st_ew, st_m, i19)
    print(f'\n=== C · MOMENTUM DE ACCIONES (S&P 500 actual; periodo {pd.to_datetime(cal[s1] * 86400, unit="s").date()} → hoy) ===')
    _line('S&P 500 comprar y mantener', metrics(r_spy[s1:], cal[s1:]))
    _line('Igual-ponderado del mismo universo (con sesgo)', metrics(r_ew[s1:], cal[s1:]))
    _line('Momentum 12-1, 30 acciones sobre su SMA200', metrics(r_m[s1:], cal[s1:]))
    print('   sensibilidad (anual / Sharpe / caída máx., mismo periodo):')
    for (look, K), (r, st) in sg.items():
        m = metrics(r[s1:], cal[s1:])
        print(f"     {look}-21 sesiones, {K} acciones: {m['cagr']:+5.1f}% / {m['sharpe']:.2f} / {m['dd']:.0f}%")
    mw, mm_ = metrics(r_ew[s1:], cal[s1:]), metrics(r_m[s1:], cal[s1:])
    print(f"   ventaja del momentum sobre el igual-ponderado del mismo universo: {mm_['cagr'] - mw['cagr']:+.1f} puntos anuales (el sesgo de supervivencia afecta a los dos)")

    # ── D. combinación con el sistema de rebotes ─────────────────────────
    try:
        d_ret, d0 = _dip_returns(EV, data, cal)
    except Exception as e:                                       # pragma: no cover
        print(f'  (no se pudo reconstruir el sistema de rebotes: {e})')
        d_ret = None
    if d_ret is not None:
        s2 = max(d0, s0, i19)
        print(f'\n=== D · COMBINACIÓN con el sistema de rebotes (Equilibrado, cartera 10×10 %; curva a precio realizado) desde {pd.to_datetime(cal[s2] * 86400, unit="s").date()} ===')
        _line('S&P 500 comprar y mantener', metrics(r_spy[s2:], cal[s2:]))
        _line('Rebotes (Equilibrado) solo', metrics(d_ret[s2:], cal[s2:]))
        parts = {nm: res[nm] for nm in names_main if nm in res}
        for nm, r in parts.items():
            c_ = np.corrcoef(d_ret[s2:], r[s2:])[0, 1]
            _line(f'{nm[:32]} solo', metrics(r[s2:], cal[s2:]))
            print(f"      correlación diaria con rebotes: {c_:+.2f}")
            _line('  50 % rebotes + 50 % ' + nm[:20], metrics(0.5 * d_ret[s2:] + 0.5 * r[s2:], cal[s2:]))
        allr = np.mean([d_ret[s2:]] + [r[s2:] for r in parts.values()], axis=0)
        _line(f'  partes iguales: rebotes + {len(parts)} de medio plazo', metrics(allr, cal[s2:]))
    sys.stdout.flush()


def _dip_returns(EV, data, cal):
    """Rentabilidad diaria de la cartera del plan Equilibrado (3 mejores/día, ATR ≥ 2,5 %, 10 posiciones × 10 %) a partir de la tabla de eventos."""
    from research import (FEATURES, GROUPS, union_events, topn_mask, rank_in_day, portfolio_sim, year_of, day_of, wf_predict_linear, PLANS, MIN_ATR_PCT)
    reg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'validated.json')))
    plan = PLANS[0]
    groups = next(s['groups'] for s in reg['strategies'] if s['id'] == plan['id'])
    rows, FL, names = union_events(EV, +1)
    vi = EV['variants'].index(plan['id'])
    day, gid = EV['day'][rows], EV['grp'][rows]
    pnl, bars = EV['pnl'][rows, vi], EV['bars'][rows, vi]
    ok = np.isfinite(pnl)
    y = (pnl > 0).astype(np.float32)
    p, base = wf_predict_linear(EV, rows, FL, y, ok, int(day.max()))
    atrp = EV['X'][rows][:, FEATURES.index('atrp')]
    cand = ok & np.isfinite(p) & (p >= base) & np.isin(gid, [GROUPS.index(g) for g in groups]) & (atrp >= MIN_ATR_PCT)
    sel = topn_mask(p, day, cand, int(plan.get('n', 3)))
    rk = rank_in_day(p, day, sel)
    ti = np.searchsorted(cal, day, side='left')
    o = portfolio_sim(cal, ti, bars, np.where(sel & (day >= day_of('2019-01-01')), pnl, np.nan), rk, 10, 0.10)
    curve = o['curve']
    t0 = int(np.flatnonzero(curve != 1.0)[0]) if (curve != 1.0).any() else 0
    full = np.ones(len(cal)); full[len(cal) - len(curve):] = curve
    r = np.zeros(len(cal)); r[1:] = full[1:] / full[:-1] - 1
    st = len(cal) - len(curve)
    return r, st
