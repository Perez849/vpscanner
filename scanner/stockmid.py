#!/usr/bin/env python3
"""
stockmid.py — BÚSQUEDA SISTEMÁTICA a medio plazo con acciones: miles de variantes y control del «mejor de miles»
(comandos `stockmid` y `stockmid_long`). NO interviene en producción.

Por qué hace falta cuidado: si pruebas miles de variantes sobre ~9 años, SIEMPRE habrá alguna que parezca estupenda por puro azar.
Por eso, además de probarlas, se mide:
  1. Contraste de «mejor de N»: ¿el mejor resultado observado supera lo que daría el azar entre N variantes SIN habilidad?
     La habilidad se mide como EXCESO sobre comprar todo el universo a partes iguales (en un mercado alcista casi cualquier cartera
     de acciones gana dinero; lo que importa es si ELEGIR bien añade algo).
  2. Selección honesta (walk-forward): cada año se elige la mejor variante con los años anteriores y se mide en el año siguiente.
  3. Persistencia: ¿las variantes que fueron mejores en la primera mitad siguen siéndolo en la segunda?
  4. Correlación de rangos (IC) de cada señal con la rentabilidad del mes siguiente.
Universos: S&P 500 actual con historial completo, y «liquidez en cada fecha» (las 300 acciones más negociadas en ese momento).
Ninguno elimina el sesgo de supervivencia (las empresas que desaparecieron no están en los datos).

Variantes: señales (momentum con 8 ventanas × 3 saltos, momentum ajustado por volatilidad, baja volatilidad, cercanía a máximos,
reversión, distancia a medias, y combinaciones por promedio de rangos) × nº de acciones {10, 20, 30, 50, 100} × cartera de 1 mes o
de 3 meses solapados × filtro {ninguno, acción sobre su SMA200, mercado (SPY sobre su SMA200; si no, liquidez)}.
Señal con el cierre de fin de mes, ejecución al cierre del día siguiente, costes 0,15 % ida y vuelta sobre lo que rota. Rentabilidades
con dividendos (adjclose).
"""
from __future__ import annotations
import itertools
import math
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from midterm import _px, month_ends, ROT_ETFS

COST_RT = 0.0015
KS = (10, 20, 30, 50, 100)
HS = (1, 3)
FILTERS = ('none', 'sma200', 'market')
MIN_NAMES = 5


def _field(data: Dict, syms: List[str], cal: np.ndarray, key: str) -> pd.DataFrame:
    idx = pd.Index(cal)
    cols = {}
    for s in syms:
        b = data.get(s)
        if b is None:
            continue
        d = (b['t'] // 86400).astype(np.int64)
        ser = pd.Series(b[key], index=d)
        cols[s] = ser[~ser.index.duplicated(keep='last')].reindex(idx)
    return pd.DataFrame(cols, index=idx)


def cs_rank(M: np.ndarray) -> np.ndarray:
    return pd.DataFrame(M).rank(axis=1, pct=True).to_numpy()


def build_factors(P: np.ndarray, R: np.ndarray) -> Dict[str, np.ndarray]:
    """Matrices (nP × N) de cada señal en las fechas de decisión R (solo información hasta el cierre de R)."""
    df = pd.DataFrame(P)
    ret = df.pct_change(fill_method=None)
    F: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for L in (21, 42, 63, 84, 126, 189, 252, 315):
            for s in (0, 5, 21):
                if s < L:
                    F[f'mom{L}_{s}'] = P[R - s] / P[R - L] - 1
        vol = {L: ret.rolling(L, min_periods=int(L * 0.8)).std().to_numpy()[R] for L in (63, 126, 252)}
        for L in (63, 126, 252):
            F[f'mv{L}'] = (P[R - 21] / P[R - L] - 1) / vol[L]
            F[f'lv{L}'] = -vol[L]
        hi252 = df.rolling(252, min_periods=200).max().to_numpy()[R]
        hi126 = df.rolling(126, min_periods=100).max().to_numpy()[R]
        F['hi52'] = P[R] / hi252
        F['hi26'] = P[R] / hi126
        F['rev21'] = -(P[R] / P[R - 21] - 1)
        F['rev5'] = -(P[R] / P[R - 5] - 1)
        sma200 = df.rolling(200, min_periods=150).mean().to_numpy()[R]
        sma50 = df.rolling(50, min_periods=40).mean().to_numpy()[R]
        F['dist200'] = P[R] / sma200 - 1
        F['dist50'] = P[R] / sma50 - 1
    base = ['mom252_21', 'mom126_21', 'mv126', 'hi52', 'lv126', 'rev21', 'dist200']
    rk = {k: cs_rank(F[k]) for k in base}
    for k in (2, 3):
        for combo in itertools.combinations(base, k):
            F['+'.join(combo)] = np.nanmean(np.stack([rk[c] for c in combo]), axis=0)
    F['_above200'] = (P[R] > sma200).astype(float)
    return F


def family(name: str) -> str:
    if '+' in name:
        return 'combinación de señales'
    if name.startswith('mom'):
        return 'momentum (rentabilidad pasada)'
    if name.startswith('mv'):
        return 'momentum ajustado por volatilidad'
    if name.startswith('lv'):
        return 'baja volatilidad'
    if name.startswith('hi'):
        return 'cercanía a máximos'
    if name.startswith('rev'):
        return 'reversión (el que más cayó)'
    return 'distancia a la media'


def stats_m(r: np.ndarray, per_year: float = 12.0):
    r = np.nan_to_num(r)
    sd = r.std()
    sh = r.mean() / sd * math.sqrt(per_year) if sd > 0 else 0.0
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (per_year / len(r)) - 1 if len(r) else 0.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return sh, cagr * 100, dd * 100


def _select(sc: np.ndarray, K: int) -> List[np.ndarray]:
    out = []
    for p in range(sc.shape[0]):
        row = sc[p]
        nv = int(np.isfinite(row).sum())
        k = min(K, nv)
        out.append(np.array([], int) if k < MIN_NAMES else np.argpartition(-row, k - 1)[:k])
    return out


def _variant_returns(sel: List[np.ndarray], RET1: np.ndarray, cash_ret: np.ndarray, h: int) -> np.ndarray:
    """Rentabilidad mensual neta de costes. h=1: cartera nueva cada mes. h=3: tres selecciones solapadas (las de los últimos 3 meses)."""
    nP = RET1.shape[0]

    def month(p: int, lag: int) -> float:
        s_ = sel[p - lag] if p - lag >= 0 else np.array([], int)
        if not len(s_):
            return cash_ret[p]
        v = RET1[p, s_]
        return float(np.nanmean(v)) if np.isfinite(v).any() else cash_ret[p]

    r = np.zeros(nP)
    for p in range(nP):
        r[p] = np.mean([month(p, lag) for lag in range(h)])
    w = 1.0 / h
    for p in range(h, nP):
        a, b = sel[p - h], sel[p]
        if len(b):
            ov = len(np.intersect1d(a, b)) / len(b)
            r[p] -= COST_RT * (1 - ov) * w
    return r


def core(uni: Dict, data: Dict, stk: List[str], liquid_universe: bool, title: str):
    spy = data['SPY']
    cal = (spy['t'] // 86400).astype(np.int64)
    T = len(cal)
    me = month_ends(cal)
    Rall = np.array([i for i in me if i >= 320 and i + 1 < T])
    R = Rall[:-1]
    trade = R + 1
    nxt = Rall[1:] + 1
    nP = len(R)
    print(f'\n######## {title} · {nP} meses de decisión ({pd.to_datetime(cal[R[0]] * 86400, unit="s").date()} → {pd.to_datetime(cal[R[-1]] * 86400, unit="s").date()})', flush=True)
    spy_sma = pd.Series(spy['c']).rolling(200).mean().to_numpy()
    mkt_up = spy['c'][R] > spy_sma[R]
    cash_sym = 'BIL' if 'BIL' in data else ('SHY' if 'SHY' in data else None)
    if cash_sym:
        cp = _px(data, [cash_sym], cal)[cash_sym].to_numpy()
        cash_ret = np.nan_to_num(cp[nxt] / cp[trade] - 1)
    else:
        cash_ret = np.zeros(nP)
    spy_ret = spy['c'][nxt] / spy['c'][trade] - 1
    px = _px(data, stk, cal)
    stk = list(px.columns)
    P = px.to_numpy()
    vol_df = _field(data, stk, cal, 'v')
    dv60 = (px * vol_df).rolling(60, min_periods=40).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        RET1 = P[nxt] / P[trade] - 1                                      # de la ejecución (cierre del día siguiente a la señal) a la del mes siguiente
    large = np.array([uni[s]['group'] == 'us_large' for s in stk])
    print(f'  acciones con historial completo: {len(stk)} (S&P 500 actual: {int(large.sum())}) · liquidez: {cash_sym}', flush=True)
    F = build_factors(P, R)
    above = F.pop('_above200') > 0
    names = list(F)
    print(f'  señales: {len(names)} · variantes por universo: {len(names) * len(KS) * len(HS) * len(FILTERS):,}', flush=True)

    universes = {'S&P 500 actual': np.broadcast_to(large[None, :], (nP, len(stk))) & np.isfinite(P[R])}
    if liquid_universe:
        dv_at = np.where(np.isfinite(dv60[R]) & np.isfinite(P[R]), dv60[R], np.nan)
        rank_dv = pd.DataFrame(dv_at).rank(axis=1, ascending=False).to_numpy()
        universes['las 300 acciones más negociadas en cada fecha (EE.UU.)'] = rank_dv <= 300
    per_year = 12.0

    for uname, valid in universes.items():
        valid = np.asarray(valid, bool)
        print(f'\n================ UNIVERSO: {uname} (≈ {int(valid.sum(1).mean())} acciones por fecha) ================', flush=True)
        ew = np.array([np.nanmean(np.where(valid[p], RET1[p], np.nan)) for p in range(nP)])

        # ── IC por señal ─────────────────────────────────────────────────
        rows = []
        for k in names:
            ics = []
            for p in range(nP):
                m = valid[p] & np.isfinite(F[k][p]) & np.isfinite(RET1[p])
                if m.sum() >= 50:
                    ics.append(pd.Series(F[k][p][m]).rank().corr(pd.Series(RET1[p][m]).rank()))
            if len(ics) >= 24:
                a = np.array(ics)
                rows.append((k, a.mean(), (a > 0).mean() * 100, a.mean() / (a.std() / math.sqrt(len(a)) + 1e-12)))
        if rows:
            rows.sort(key=lambda r: -abs(r[3]))
            print(f'  [IC] correlación de rangos señal → rentabilidad del MES SIGUIENTE (las 10 con mayor |t|; con {len(names)} señales probadas, unas {len(names) // 20} saldrían con |t|>2 solo por azar):')
            for k, m_, pos, t in rows[:10]:
                print(f"    {k:34s} IC {m_:+.3f} · {pos:3.0f}% meses+ · t={t:+.1f}")
            for k in ('mom252_21', 'mom126_21', 'hi52', 'lv126', 'rev21'):
                r_ = [x for x in rows if x[0] == k]
                if r_:
                    print(f"    (referencia) {k:22s} IC {r_[0][1]:+.3f} · {r_[0][2]:3.0f}% meses+ · t={r_[0][3]:+.1f}")

        # ── todas las variantes ─────────────────────────────────────────
        series: Dict[tuple, np.ndarray] = {}
        for name in names:
            sc0 = np.where(valid & np.isfinite(F[name]), F[name], -np.inf)
            for flt in ('none', 'sma200'):
                sc = np.where(above, sc0, -np.inf) if flt == 'sma200' else sc0
                for K in KS:
                    sel = _select(sc, K)
                    for h in HS:
                        series[(name, flt, K, h)] = _variant_returns(sel, RET1, cash_ret, h)
        flips = np.r_[False, mkt_up[1:] != mkt_up[:-1]]
        for key in [k for k in series if k[1] == 'none']:
            name, _, K, h = key
            series[(name, 'market', K, h)] = np.where(mkt_up, series[key], cash_ret) - COST_RT * flips
        keys = list(series)
        M = np.column_stack([series[k][2:] for k in keys])
        ew2, spy2 = ew[2:], spy_ret[2:]
        n_m = M.shape[0]
        sh = M.mean(0) / (M.std(0) + 1e-12) * math.sqrt(per_year)
        cg = (np.prod(1 + M, axis=0) ** (per_year / n_m) - 1) * 100
        ew_s, ew_c, ew_d = stats_m(ew2)
        sp_s, sp_c, sp_d = stats_m(spy2)
        X = M - ew2[:, None]                                              # exceso sobre comprar todo el universo a partes iguales
        ir = X.mean(0) / (X.std(0) + 1e-12) * math.sqrt(per_year)
        print(f"\n  {len(keys):,} variantes · {n_m} meses · referencias: universo a partes iguales {ew_c:+.1f}% anual (Sharpe {ew_s:.2f}, caída {ew_d:.0f}%) · S&P 500 {sp_c:+.1f}% (Sharpe {sp_s:.2f}, caída {sp_d:.0f}%)")
        print(f"  Sharpe: mediana {np.median(sh):.2f} · p90 {np.quantile(sh, .9):.2f} · máx {sh.max():.2f} | rentabilidad anual: mediana {np.median(cg):+.1f}% · p90 {np.quantile(cg, .9):+.1f}% · máx {cg.max():+.1f}%")
        print(f"  superan al S&P 500 en rentabilidad: {np.mean(cg > sp_c) * 100:.0f}% · en Sharpe: {np.mean(sh > sp_s) * 100:.0f}% | superan al universo a partes iguales en rentabilidad: {np.mean(cg > ew_c) * 100:.0f}% · en Sharpe: {np.mean(sh > ew_s) * 100:.0f}%")
        print(f"  EXCESO sobre el universo a partes iguales (habilidad de elegir): mediana {np.median(X.mean(0) * 12 * 100):+.2f} puntos/año · ratio de información mediano {np.median(ir):+.2f} · >0 en {np.mean(ir > 0) * 100:.0f}% de las variantes")
        fam = np.array([family(k[0]) for k in keys])
        flt_ = np.array([k[1] for k in keys])
        print('  por filtro:  ' + ' · '.join(f"{f}: rent. mediana {np.median(cg[flt_ == f]):+.1f}%/Sharpe {np.median(sh[flt_ == f]):.2f}/exceso {np.median(X[:, flt_ == f].mean(0)) * 1200:+.1f} pts" for f in FILTERS))
        print('  por familia de señal (mediana de las variantes con filtro «ninguno»):')
        for f in sorted(set(fam)):
            m = (fam == f) & (flt_ == 'none')
            print(f"    {f:36s} rent. {np.median(cg[m]):+5.1f}% · Sharpe {np.median(sh[m]):.2f} · exceso {np.median(X[:, m].mean(0)) * 1200:+5.2f} pts/año · IR {np.median(ir[m]):+.2f} · supera al universo en Sharpe: {np.mean(sh[m] > ew_s) * 100:3.0f}%")
        order = np.argsort(-sh)[:8]
        print('  las 8 MEJORES por Sharpe (elegidas mirando el futuro: NO son operables tal cual):')
        for j in order:
            name, flt, K, h = keys[j]
            s_, c_, d_ = stats_m(M[:, j])
            print(f"    {name[:44]:44s} filtro={flt:6s} K={K:3d} cartera {h}m → {c_:+5.1f}% anual · Sharpe {s_:4.2f} · caída {d_:4.0f}% · exceso {X[:, j].mean() * 1200:+.1f} pts")
        print('  la IDEA CLÁSICA, sin elegir nada (momentum 12-1 meses, 50 acciones, mensual):')
        for flt in FILTERS:
            if ('mom252_21', flt, 50, 1) not in series:
                continue
            j = keys.index(('mom252_21', flt, 50, 1))
            s_, c_, d_ = stats_m(M[:, j])
            print(f"    filtro={flt:6s} → {c_:+5.1f}% anual · Sharpe {s_:4.2f} · caída {d_:4.0f}% · exceso {X[:, j].mean() * 1200:+.1f} pts · lugar {int((sh > sh[j]).sum()) + 1} de {len(keys):,}")

        # ── contraste del «mejor de N»: remuestreo por bloques sin habilidad ─
        rng = np.random.default_rng(3)
        Xc = X - X.mean(0)                                                  # sin habilidad: exceso medio cero en cada variante
        best_null = []
        blk = 3
        for _ in range(400):
            starts = rng.integers(0, n_m, size=int(math.ceil(n_m / blk)))
            ix = np.concatenate([(s_ + np.arange(blk)) % n_m for s_ in starts])[:n_m]
            Z = Xc[ix]
            best_null.append((Z.mean(0) / (Z.std(0) + 1e-12)).max() * math.sqrt(per_year))
        best_null = np.array(best_null)
        print(f"  CONTRASTE «mejor de {len(keys):,}» (habilidad = exceso sobre el universo): mejor ratio de información observado {ir.max():.2f} · sin habilidad el máximo mediano sería {np.median(best_null):.2f} (p95 {np.quantile(best_null, .95):.2f}) → p-valor ≈ {np.mean(best_null >= ir.max()):.3f}")

        # ── persistencia entre mitades ───────────────────────────────────
        half = n_m // 2
        ir1 = X[:half].mean(0) / (X[:half].std(0) + 1e-12) * math.sqrt(per_year)
        ir2 = X[half:].mean(0) / (X[half:].std(0) + 1e-12) * math.sqrt(per_year)
        sh1 = M[:half].mean(0) / (M[:half].std(0) + 1e-12) * math.sqrt(per_year)
        sh2 = M[half:].mean(0) / (M[half:].std(0) + 1e-12) * math.sqrt(per_year)
        rc = pd.Series(ir1).rank().corr(pd.Series(ir2).rank())
        rcs = pd.Series(sh1).rank().corr(pd.Series(sh2).rank())
        top1 = ir1 >= np.quantile(ir1, 0.9)
        print(f"  PERSISTENCIA: correlación de rangos entre la 1.ª y la 2.ª mitad de la muestra → exceso (IR) {rc:+.2f} · Sharpe {rcs:+.2f}. El 10 % mejor de la 1.ª mitad tiene en la 2.ª un exceso mediano de {np.median(X[half:][:, top1].mean(0)) * 1200:+.2f} pts/año (el conjunto entero: {np.median(X[half:].mean(0)) * 1200:+.2f}).")

        # ── selección honesta walk-forward ───────────────────────────────
        dates = pd.to_datetime(cal[R][2:] * 86400, unit='s')
        years = dates.year.to_numpy()
        yrs = [Y for Y in sorted(set(years)) if (years < Y).sum() >= 24 and (years == Y).sum() >= 6]
        if yrs:
            ixs, picks = [], {'sh': [], 'ir': [], 'top20': [], 'all': []}
            for Y in yrs:
                tr, te = years < Y, years == Y
                a = M[tr].mean(0) / (M[tr].std(0) + 1e-12)
                b = X[tr].mean(0) / (X[tr].std(0) + 1e-12)
                picks['sh'].append(M[te][:, np.argmax(a)])
                picks['ir'].append(M[te][:, np.argmax(b)])
                picks['top20'].append(M[te][:, np.argsort(-a)[:20]].mean(1))
                picks['all'].append(M[te].mean(1))
                ixs.append(np.flatnonzero(te))
            ix = np.concatenate(ixs)
            print(f"  SELECCIÓN HONESTA (cada año se elige con los años anteriores · {len(ix)} meses fuera de muestra, {yrs[0]}–{yrs[-1]}):")
            lab = {'sh': 'la de mejor Sharpe en los años previos', 'ir': 'la de mejor exceso (IR) en los años previos', 'top20': 'promedio de las 20 mejores en Sharpe', 'all': 'una variante cualquiera (promedio de todas)'}
            for k in ('sh', 'ir', 'top20', 'all'):
                s_, c_, d_ = stats_m(np.concatenate(picks[k]))
                print(f"    {lab[k]:46s} {c_:+5.1f}% anual · Sharpe {s_:4.2f} · caída {d_:4.0f}%")
            s_, c_, d_ = stats_m(ew2[ix]); print(f"    {'universo a partes iguales (mismos meses)':46s} {c_:+5.1f}% anual · Sharpe {s_:4.2f} · caída {d_:4.0f}%")
            s_, c_, d_ = stats_m(spy2[ix]); print(f"    {'S&P 500 comprar y mantener (mismos meses)':46s} {c_:+5.1f}% anual · Sharpe {s_:4.2f} · caída {d_:4.0f}%")
            print('    por año (la mejor por Sharpe previo | universo | S&P 500): ' + ' · '.join(
                f"{Y}: {np.prod(1 + p_) * 100 - 100:+.0f}|{np.prod(1 + ew2[i_]) * 100 - 100:+.0f}|{np.prod(1 + spy2[i_]) * 100 - 100:+.0f}" for Y, p_, i_ in zip(yrs, picks['sh'], ixs)))
        sys.stdout.flush()


def _etf_persistence(data: Dict):
    spy = data['SPY']
    cal = (spy['t'] // 86400).astype(np.int64)
    T = len(cal)
    me = month_ends(cal)
    Rall = np.array([i for i in me if i >= 320 and i + 1 < T])
    R = Rall[:-1]
    trade = R + 1
    nxt = Rall[1:] + 1
    etfs = [s for s in ROT_ETFS if s in data]
    if len(etfs) < 8:
        return
    pe = _px(data, etfs, cal).to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        reti = pe[nxt] / pe[trade] - 1
    print(f'\n[ETF] ¿PERSISTEN LOS «MEJORES ETF»? IC mensual entre el ranking de momentum y la rentabilidad del mes siguiente ({len(etfs)} ETF, {len(R)} meses)')
    for L, s in ((63, 0), (126, 0), (252, 0), (252, 21), (21, 0)):
        ics = []
        for p in range(len(R)):
            sc = pe[R[p] - s] / pe[R[p] - L] - 1
            m = np.isfinite(sc) & np.isfinite(reti[p])
            if m.sum() >= 12:
                ics.append(pd.Series(sc[m]).rank().corr(pd.Series(reti[p][m]).rank()))
        a = np.array(ics)
        print(f"    momentum {L:3d} sesiones (salto {s:2d}): IC {a.mean():+.3f} · {np.mean(a > 0) * 100:3.0f}% meses+ · t={a.mean() / (a.std() / math.sqrt(len(a))):+.1f}")
    sys.stdout.flush()


def _full_history(uni: Dict, data: Dict, cal0: int) -> List[str]:
    return [s for s in uni if uni[s]['group'] in ('us_large', 'us_mid', 'us_small') and not uni[s].get('aux') and s in data
            and (data[s]['t'][0] // 86400) <= cal0 + 40]


def run(uni: Dict, data: Dict, out_dir: str):
    cal0 = int(data['SPY']['t'][0] // 86400)
    stk = _full_history(uni, data, cal0)
    core(uni, data, stk, True, 'MEDIO PLAZO CON ACCIONES · 10 AÑOS')
    _etf_persistence(data)


def run_long(uni: Dict, data: Dict, out_dir: str):
    """Historia larga (desde 2004): solo S&P 500 actual. Descarga aparte (diaria, con period1) para incluir 2008–09."""
    import data as D
    syms = [s for s in uni if uni[s]['group'] == 'us_large' and not uni[s].get('aux')]
    pool = list(dict.fromkeys(syms + ['SPY', 'SHY', 'BIL']))
    bars, failed = D.fetch_many(pool, workers=8, cache_path=None, verbose=False, period1=int(pd.Timestamp('2004-01-02').timestamp()))
    calL = (bars['SPY']['t'] // 86400).astype(np.int64)
    gaps = np.diff(calL)
    if np.median(gaps) > 3 or gaps.max() > 12:
        raise SystemExit(f'barras no diarias (mediana {np.median(gaps)} días, máximo {gaps.max()})')
    print(f'  descargadas {len(bars)} series desde 2004 · fallidas {len(failed)}', flush=True)
    stk = _full_history(uni, bars, int(calL[0]))
    core(uni, bars, stk, False, 'MEDIO PLAZO CON ACCIONES · HISTORIA LARGA (S&P 500 actual desde 2004; incluye 2008-09)')
    _etf_persistence(bars)
