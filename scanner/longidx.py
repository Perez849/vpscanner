#!/usr/bin/env python3
"""
longidx.py — ¿ES REAL EL MECANISMO DE LOS REBOTES? Prueba en ÍNDICES con hasta 60 años de historia (comando `longidx`; NO interviene en producción).

El sistema de rebotes se afinó con acciones de 2016–2026, con sesgo de supervivencia y en una época concreta. Un índice no tiene sesgo de
supervivencia (el índice ES el mercado) y hay décadas distintas (60, 70, 80, 90, 2000, 2010, 2020), con mercados bajistas largos (1973-74,
2000-02, 2008). Si «sobreventa dentro de tendencia alcista» (los 17 patrones, mismo criterio que el sistema) también paga en 15 índices del
mundo y en todas las décadas, el mecanismo es real. Si solo pagó en 2016–2026, era un artefacto de esa época.

Cada candidato se compara con la entrada AL AZAR en tendencia alcista (cualquier barra sobre su SMA200) con la MISMA salida: solo cuenta la
diferencia. Señal con el cierre, entrada a la apertura siguiente, coste 0,10 % ida y vuelta (aproximación: antes de 1993 no había ETF; se opera
con futuros).
"""
from __future__ import annotations
import math
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import feats
import setups as SU
import simulate as SM

INDICES = ['^GSPC', '^IXIC', '^RUT', '^DJI', '^FTSE', '^GDAXI', '^FCHI', '^N225', '^HSI', '^GSPTSE', '^BVSP', '^AXJO', '^IBEX', '^STOXX50E', '^KS11', '^BSESN', '^TWII', '^MXX', '^NYA']
VARIANTS = [SM.Variant('rsi', 0.0, 4.0, 10), SM.Variant('atr', 1.0, 4.0, 10)]
COST = 0.10
MIN_BARS = 1500


def clean_series(b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray] | None:
    """Último tramo diario continuo con OHLC real (sin saltos de más de 15 días y con máximo > mínimo en casi todas las barras)."""
    t = b['t'] // 86400
    n = len(t)
    if n < MIN_BARS:
        return None
    cut = 0
    br = np.flatnonzero(np.diff(t) > 15)
    if len(br):
        cut = int(br[-1]) + 1
    rng_ok = (b['h'] > b['l']).astype(float)
    ma = pd.Series(rng_ok).rolling(250, min_periods=250).mean().to_numpy()
    good = np.flatnonzero(np.isfinite(ma) & (ma > 0.9))
    if not len(good):
        return None
    cut = max(cut, int(good[0]) - 249)
    if n - cut < MIN_BARS:
        return None
    seg = {k: v[cut:] for k, v in b.items() if isinstance(v, np.ndarray) and len(v) == n}
    gaps = np.diff(seg['t'] // 86400)
    return seg if np.median(gaps) <= 3 else None


def _stats(pnl: np.ndarray, day: np.ndarray):
    from research import stats
    return stats(pnl, day)


def run(uni: Dict, data: Dict, out_dir: str):
    import data as D
    since = int(pd.Timestamp('1950-01-03').timestamp())
    bars, failed = D.fetch_many(INDICES, workers=6, cache_path=None, verbose=False, period1=since)
    SU.MIN_ATRP = 0.0                                              # sin el suelo de volatilidad de las acciones: aquí se prueba el mecanismo
    pids = [p for p in SU.long_pattern_ids() if p not in ('L_vpval',)]
    print(f'\n######## ÍNDICES LARGOS · {len(bars)} descargados · fallidos {failed}', flush=True)
    rows = []                                                     # (índice, año, década, pnl_cand[v], pnl_base[v]) agregados luego
    cand: Dict[str, Dict] = {}
    for s in INDICES:
        if s not in bars:
            continue
        seg = clean_series(bars[s])
        if seg is None:
            print(f'  {s}: sin tramo diario utilizable', flush=True)
            continue
        F = feats.build(seg)
        idx, _ = SU.candidate_bars(F, 'index', pids)
        base = np.flatnonzero(np.isfinite(F['sma200']) & (F['c'] > F['sma200']) & np.isfinite(F['atr']) & (F['atr'] > 0))
        idx = idx[idx < len(F['c']) - 12]
        base = base[base < len(F['c']) - 12][::3]                  # cada 3.ª barra: menos solapamiento, misma media
        rc = SM.simulate(F, idx, +1, VARIANTS, COST)
        rb = SM.simulate(F, base, +1, VARIANTS, COST)
        day = (F['t'] // 86400).astype(np.int64)
        cand[s] = {'F': F, 'idx': idx, 'base': base, 'day': day, 'rc': rc, 'rb': rb}
        y0, y1 = pd.to_datetime(day[0] * 86400, unit='s').year, pd.to_datetime(day[-1] * 86400, unit='s').year
        print(f'  {s:10s} {y0}–{y1} · {len(F["c"]):6d} barras · {len(idx):4d} candidatos · {len(base):5d} barras de referencia', flush=True)

    for v in VARIANTS:
        name = v.name
        print(f'\n================ SALIDA {name} ================')
        print('  POR ÍNDICE (candidatos frente a entrar al azar en tendencia alcista con la misma salida):')
        pooled_c, pooled_b, pooled_cd, pooled_bd, pooled_sym = [], [], [], [], []
        wins = 0
        for s, e in cand.items():
            pc, pb = e['rc'][name].pnl, e['rb'][name].pnl
            ok_c, ok_b = np.isfinite(pc), np.isfinite(pb)
            if ok_c.sum() < 30:
                continue
            sc = _stats(pc[ok_c], e['day'][e['idx']][ok_c])
            mb = float(np.mean(pb[ok_b]))
            wins += sc['mean'] > mb
            print(f"    {s:10s} n={sc['n']:4d} acierto {sc['wr']:4.1f}% · media {sc['mean']:+5.2f}% (t={sc['t_day']:+.1f}) · PF {sc['pf']:.2f} | al azar: {np.mean(pb[ok_b] > 0) * 100:4.1f}% / {mb:+5.2f}% | Δ {sc['mean'] - mb:+5.2f} pts")
            pooled_c.append(pc[ok_c]); pooled_cd.append(e['day'][e['idx']][ok_c]); pooled_sym.append(np.full(ok_c.sum(), s))
            pooled_b.append(pb[ok_b]); pooled_bd.append(e['day'][e['base']][ok_b])
        n_idx = len(pooled_c)
        print(f'  → el mecanismo supera al azar en {wins} de {n_idx} índices')
        pc, pcd = np.concatenate(pooled_c), np.concatenate(pooled_cd)
        pb, pbd = np.concatenate(pooled_b), np.concatenate(pooled_bd)
        yc = pd.to_datetime(pcd * 86400, unit='s').year.to_numpy()
        yb = pd.to_datetime(pbd * 86400, unit='s').year.to_numpy()
        print('  AGRUPADO POR DÉCADA (todos los índices disponibles en cada década):')
        for d0 in range(1960, 2030, 10):
            mc, mb = (yc >= d0) & (yc < d0 + 10), (yb >= d0) & (yb < d0 + 10)
            if mc.sum() < 40:
                continue
            print(f"    {d0}s: n={int(mc.sum()):4d} acierto {np.mean(pc[mc] > 0) * 100:4.1f}% · media {np.mean(pc[mc]):+5.2f}% | al azar {np.mean(pb[mb] > 0) * 100:4.1f}% / {np.mean(pb[mb]):+5.2f}% | Δ {np.mean(pc[mc]) - np.mean(pb[mb]):+5.2f} pts (error típico ≈ {np.std(pc[mc]) / math.sqrt(mc.sum()):.2f})")
        st = _stats(pc, pcd)
        mb_all = float(np.mean(pb))
        print(f"    TODO: n={st['n']} acierto {st['wr']:.1f}% · media {st['mean']:+.2f}% (t={st['t_day']:+.1f}) · PF {st['pf']:.2f} | al azar {mb_all:+.2f}% → Δ {st['mean'] - mb_all:+.2f} pts")
        print('  EN MERCADOS BAJISTAS FUERTES (años 1973-74, 2000-02, 2008): ' + ' · '.join(
            f"{a}: n={int(((yc >= a) & (yc <= b_)).sum())}, media {np.mean(pc[(yc >= a) & (yc <= b_)]):+.2f}% (al azar {np.mean(pb[(yb >= a) & (yb <= b_)]):+.2f}%)" for a, b_ in ((1973, 1974), (2000, 2002), (2008, 2008)) if ((yc >= a) & (yc <= b_)).sum() >= 10))

    # ── S&P 500 desde 1962: una posición, todo el capital, frente a comprar y mantener ──
    if '^GSPC' in cand:
        e = cand['^GSPC']; F = e['F']
        print('\n================ CARTERA «solo S&P 500»: una posición con el 100 % del capital, el resto en liquidez al 0 % (aproximación) ================', flush=True)
        for v in VARIANTS:
            r = e['rc'][v.name]
            idx, pnl, nb = e['idx'], r.pnl, r.bars.astype(int)
            eq, last_exit, in_mkt, trades = 1.0, -1, 0, []
            for q in range(len(idx)):
                i = idx[q]
                if not np.isfinite(pnl[q]) or i + 1 <= last_exit:
                    continue
                eq *= 1 + pnl[q] / 100.0
                last_exit = i + max(nb[q], 1)
                in_mkt += max(nb[q], 1)
                trades.append((i, pnl[q]))
            yrs = (F['t'][-1] - F['t'][0]) / 86400 / 365.25
            bh = F['c'][-1] / F['c'][0]
            tr = np.array([p for _, p in trades])
            print(f"  salida {v.name}: {len(trades)} operaciones · en mercado {in_mkt / len(F['c']) * 100:.0f}% del tiempo · acierto {np.mean(tr > 0) * 100:.1f}% · media {np.mean(tr):+.2f}% · capital final x{eq:.1f} ({(eq ** (1 / yrs) - 1) * 100:+.1f}% anual) | comprar y mantener x{bh:.1f} ({(bh ** (1 / yrs) - 1) * 100:+.1f}% anual, sin dividendos) en {yrs:.0f} años")
            ti = np.array([i for i, _ in trades])
            yy = pd.to_datetime(F['t'][ti], unit='s').year.to_numpy()
            ya = pd.to_datetime(F['t'], unit='s').year.to_numpy()
            held = np.zeros(len(F['c']), bool)
            for i, _ in trades:
                q = int(np.searchsorted(idx, i))
                held[i + 1:i + 1 + max(nb[q], 1)] = True
            parts = []
            for d0 in range(1960, 2030, 10):
                mt, mb_ = (yy >= d0) & (yy < d0 + 10), (ya >= d0) & (ya < d0 + 10)
                if mt.sum() < 5:
                    continue
                bh_d = F['c'][np.flatnonzero(mb_)[-1]] / F['c'][max(np.flatnonzero(mb_)[0] - 1, 0)] - 1
                parts.append(f"{d0}s: {np.prod(1 + tr[mt] / 100.0) - 1:+.0%} (en mercado {held[mb_].mean() * 100:.0f}% · comprar y mantener {bh_d:+.0%})")
            print('    por década: ' + ' · '.join(parts))
    sys.stdout.flush()
