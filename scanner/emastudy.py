#!/usr/bin/env python3
"""
emastudy.py — Estudio de ideas con EMA 34 / 89 / 200 y RSI(14) (comando `ema` de research.py). NO interviene en producción.

Ideas del usuario (todas definidas de antemano, causales; la señal es el CIERRE de la barra i y se entra en la APERTURA de i+1):

  A. El precio CRUZA DE ABAJO A ARRIBA la EMA 34 y en ese mismo cruce el RSI(14) está en torno a 50 y SUBIENDO.
       Variantes: banda del RSI (48–52, 45–55, 40–60), sin exigir que suba, con filtros de tendencia (sobre EMA 200,
       EMA 89 > EMA 200, pila alcista EMA 34 > 89 > 200) y el RSI cruzando el 50 a la vez. Control: el cruce SIN RSI.
  B. CRUCES ENTRE EMAs: 34↗89, 89↗200, 34↗200 y «triple alineación» (se completa 34 > 89 > 200), con y sin RSI.
  C. El precio recupera la EMA 89 o la EMA 200 con el RSI en torno a 50 y subiendo.
  D. Retroceso hasta la EMA 34 dentro de una pila alcista que rebota con RSI subiendo.
  + los espejos cortos de las principales.

Cada setup se compara con una ENTRADA ALEATORIA en el mismo régimen y con la misma salida (benchmark): solo cuenta lo que lo supera.
Salidas probadas (seguimiento de tendencia): stop móvil 3×ATR / 4×ATR (máx. 60 sesiones), stop 3×ATR + 20 sesiones, objetivo 3×ATR con stop 2×ATR,
cierre bajo la EMA 34 y cierre bajo la EMA 89.
"""
from __future__ import annotations
import sys
import time
import zlib
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

import feats
import setups as SU
import simulate as SM
from research import day_of, year_of, stats

VARIANTS = [SM.Variant('trl', 3.0, 3.0, 60), SM.Variant('trl', 4.0, 4.0, 60), SM.Variant('atr', 99.0, 3.0, 20),
            SM.Variant('atr', 3.0, 2.0, 20), SM.Variant('e34', 0.0, 3.0, 60), SM.Variant('e89', 0.0, 4.0, 80)]
VNAMES = ['stop móvil 3×ATR (60 ses.)', 'stop móvil 4×ATR (60 ses.)', 'stop 3×ATR + 20 sesiones', 'objetivo 3×ATR / stop 2×ATR (20 ses.)',
          'sale al cerrar bajo EMA 34', 'sale al cerrar bajo EMA 89']
COOLDOWN = 5
CHUNK = 20000


def ema(c: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(c).ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()


def _sh(a: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full(len(a), np.nan)
    out[k:] = a[:-k]
    return out


def _cross_up(a, b):
    return np.isfinite(a) & np.isfinite(b) & np.isfinite(_sh(a)) & np.isfinite(_sh(b)) & (a > b) & (_sh(a) <= _sh(b))


def _cross_dn(a, b):
    return np.isfinite(a) & np.isfinite(b) & np.isfinite(_sh(a)) & np.isfinite(_sh(b)) & (a < b) & (_sh(a) >= _sh(b))


def build_masks(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Todas las señales (bool por barra) y las barras de referencia (entrada aleatoria por régimen)."""
    c, l, h = F['c'], F['l'], F['h']
    e34, e89, e200 = F['ema34'], F['ema89'], F['ema200']
    R = F['rsi14']; Rp = _sh(R)
    fin = np.isfinite(R) & np.isfinite(Rp)
    band = lambda lo, hi: fin & (R >= lo) & (R <= hi)
    rise, fall = fin & (R > Rp), fin & (R < Rp)
    stack = np.isfinite(e200) & (e34 > e89) & (e89 > e200)
    stack_dn = np.isfinite(e200) & (e34 < e89) & (e89 < e200)
    up200 = np.isfinite(e200) & (c > e200)
    dn200 = np.isfinite(e200) & (c < e200)
    x34, x89, x200 = _cross_up(c, e34), _cross_up(c, e89), _cross_up(c, e200)
    d34, d89, d200 = _cross_dn(c, e34), _cross_dn(c, e89), _cross_dn(c, e200)
    rcross50 = fin & (R >= 50) & (_sh(R, 3) < 50)                   # el RSI cruza el 50 hacia arriba en las últimas 3 barras
    rcross50_dn = fin & (R <= 50) & (_sh(R, 3) > 50)
    M: Dict[str, np.ndarray] = {}
    # ── A: precio cruza la EMA 34 de abajo arriba ──
    M['A0'] = x34                                                                    # control: solo el cruce
    M['A1'] = x34 & band(45, 55) & rise                                              # idea del usuario
    M['A2'] = x34 & band(40, 60) & rise
    M['A3'] = x34 & band(48, 52) & rise
    M['A4'] = x34 & band(45, 55)                                                     # sin exigir que suba
    M['A5'] = M['A1'] & up200
    M['A6'] = M['A1'] & stack
    M['A7'] = M['A1'] & np.isfinite(e200) & (e89 > e200)
    M['A8'] = x34 & rcross50 & (R <= 60)                                             # el RSI cruza el 50 a la vez que el precio la EMA 34
    M['A9'] = x34 & fin & (R > 50) & rise & up200                                    # RSI ya por encima de 50 y subiendo, sobre la EMA 200
    # ── B: cruces entre EMAs ──
    b34_89, b89_200, b34_200 = _cross_up(e34, e89), _cross_up(e89, e200), _cross_up(e34, e200)
    M['B1'] = b34_89
    M['B2'] = b89_200
    M['B3'] = b34_200
    M['B4'] = stack & ~(np.nan_to_num(_sh(stack.astype(float))) > 0)                         # se completa la pila 34 > 89 > 200
    M['B5'] = b34_89 & fin & (R > 50) & (R < 70) & rise
    M['B6'] = b89_200 & fin & (R > 50) & (R < 70) & rise
    M['B7'] = M['B4'] & fin & (R > 50) & (R < 70) & rise
    M['B8'] = b34_89 & up200
    M['B9'] = b34_89 & rcross50                                                      # cruce 34↗89 con el RSI cruzando el 50
    # ── C: el precio recupera la EMA 89 / 200 con RSI ~50 subiendo ──
    M['C0a'] = x89
    M['C0b'] = x200
    M['C1'] = x89 & band(45, 55) & rise
    M['C2'] = x200 & band(45, 55) & rise
    # ── D: retroceso a la EMA 34 dentro de una pila alcista y rebote ──
    touch = np.isfinite(e34) & (l <= e34) & (c > e34)
    M['D1'] = stack & touch & band(40, 60) & rise
    M['D2'] = stack & touch & band(40, 60) & rise & (R >= 45)
    # ── espejos CORTOS ──
    M['sA0'] = d34
    M['sA1'] = d34 & band(45, 55) & fall
    M['sA6'] = M['sA1'] & stack_dn
    M['sB1'] = _cross_up(e89, e34)                                                   # 34↘89
    M['sB4'] = stack_dn & ~(np.nan_to_num(_sh(stack_dn.astype(float))) > 0)
    M['sB5'] = M['sB1'] & fin & (R < 50) & (R > 30) & fall
    return {'masks': M, 'stack': stack, 'stack_dn': stack_dn, 'up200': up200, 'dn200': dn200}


# id → (dirección, etiqueta, referencia)
DEFS: Dict[str, tuple] = {
    'A0': (1, 'Precio cruza ↑ EMA34 (sin RSI) · CONTROL', 'all'),
    'A1': (1, 'Precio cruza ↑ EMA34 + RSI 45–55 subiendo  ◀ idea', 'all'),
    'A2': (1, 'Precio cruza ↑ EMA34 + RSI 40–60 subiendo', 'all'),
    'A3': (1, 'Precio cruza ↑ EMA34 + RSI 48–52 subiendo', 'all'),
    'A4': (1, 'Precio cruza ↑ EMA34 + RSI 45–55 (sin exigir subida)', 'all'),
    'A5': (1, 'A1 + precio sobre EMA200', 'up'),
    'A6': (1, 'A1 + pila alcista EMA34>89>200', 'stack'),
    'A7': (1, 'A1 + EMA89 > EMA200', 'up'),
    'A8': (1, 'Precio cruza ↑ EMA34 + RSI cruza ↑ 50 (≤60)', 'all'),
    'A9': (1, 'Cruce ↑ EMA34 + RSI>50 subiendo + sobre EMA200', 'up'),
    'B1': (1, 'EMA34 cruza ↑ EMA89 (sin RSI)', 'all'),
    'B2': (1, 'EMA89 cruza ↑ EMA200 (sin RSI)', 'all'),
    'B3': (1, 'EMA34 cruza ↑ EMA200 (sin RSI)', 'all'),
    'B4': (1, 'Se completa la pila EMA34>89>200 (triple cruce)', 'stack'),
    'B5': (1, 'EMA34↗89 + RSI 50–70 subiendo', 'all'),
    'B6': (1, 'EMA89↗200 + RSI 50–70 subiendo', 'all'),
    'B7': (1, 'Triple alineación + RSI 50–70 subiendo  ◀ idea', 'stack'),
    'B8': (1, 'EMA34↗89 con precio sobre EMA200', 'up'),
    'B9': (1, 'EMA34↗89 + RSI cruza ↑ 50', 'all'),
    'C0a': (1, 'Precio cruza ↑ EMA89 (sin RSI) · CONTROL', 'all'),
    'C0b': (1, 'Precio cruza ↑ EMA200 (sin RSI) · CONTROL', 'all'),
    'C1': (1, 'Precio cruza ↑ EMA89 + RSI 45–55 subiendo', 'all'),
    'C2': (1, 'Precio cruza ↑ EMA200 + RSI 45–55 subiendo', 'all'),
    'D1': (1, 'Pila alcista: toca EMA34 y rebota + RSI 40–60 subiendo', 'stack'),
    'D2': (1, 'D1 con RSI ≥ 45', 'stack'),
    'sA0': (-1, 'CORTO: precio cruza ↓ EMA34 (sin RSI) · CONTROL', 'dn_all'),
    'sA1': (-1, 'CORTO: precio cruza ↓ EMA34 + RSI 45–55 bajando', 'dn_all'),
    'sA6': (-1, 'CORTO: sA1 + pila bajista', 'stack_dn'),
    'sB1': (-1, 'CORTO: EMA34 cruza ↓ EMA89', 'dn_all'),
    'sB4': (-1, 'CORTO: se completa la pila bajista', 'stack_dn'),
    'sB5': (-1, 'CORTO: EMA34↘89 + RSI 30–50 bajando', 'dn_all'),
}
BASES = {'all': (1, 'REF · cualquier barra operable'), 'up': (1, 'REF · barra con precio sobre EMA200'), 'stack': (1, 'REF · barra con pila alcista'),
         'dn_all': (-1, 'REF corto · cualquier barra'), 'stack_dn': (-1, 'REF corto · barra con pila bajista')}


def _cooldown(m: np.ndarray, k: int) -> np.ndarray:
    cs = np.concatenate([[0], np.cumsum(m.astype(np.int64))])
    n = len(m)
    lo = np.maximum(np.arange(n) - (k - 1), 0)
    return m & ((cs[np.arange(n)] - cs[lo]) == 0)


def collect(uni: Dict, data: Dict, only_last_years: bool = False):
    """Recorre el universo y devuelve la tabla de eventos {sid, sym, grp, day, atrp, spy_up, vix_z, rsi, pnl[n,V], bars[n,V]}."""
    spy, vix = data.get('SPY'), data.get('^VIX')
    regime = feats.regime_series(spy, vix)
    ids = list(DEFS) + ['REF_' + k for k in BASES]
    sid_of = {k: i for i, k in enumerate(ids)}
    sym_list = [s for s in uni if not uni[s].get('aux') and s in data]
    acc = {k: [] for k in ('sid', 'sym', 'grp', 'day', 'atrp', 'spy_up', 'vix_z', 'rsi', 'pnl', 'bars')}
    t0 = time.time()
    for n_s, sym in enumerate(sym_list):
        g = uni[sym]['group']
        F = feats.build(data[sym])
        c = F['c']
        F['ema34'], F['ema89'], F['ema200'] = ema(c, 34), ema(c, 89), ema(c, 200)
        feats.ensure_regime(F, regime)
        trad = SU.tradable_mask(F, g)
        B = build_masks(F)
        M, n = B['masks'], len(c)
        sel: Dict[str, np.ndarray] = {}
        for k, (d, _lbl, _ref) in DEFS.items():
            idx = np.flatnonzero(_cooldown(M[k] & trad, COOLDOWN))
            idx = idx[idx < n - 1]
            if len(idx):
                sel[k] = idx
        off = zlib.crc32(sym.encode()) % 10
        sample = (np.arange(n) % 10) == off
        refmask = {'all': trad, 'up': trad & B['up200'], 'stack': trad & B['stack'], 'dn_all': trad, 'stack_dn': trad & B['stack_dn']}
        for k, m in refmask.items():
            idx = np.flatnonzero(m & sample)
            idx = idx[idx < n - 1]
            if len(idx):
                sel['REF_' + k] = idx
        cost = SM.COST_RT.get(g, 0.30)
        for sgn in (1, -1):
            keys = [k for k in sel if (DEFS[k][0] if k in DEFS else BASES[k[4:]][0]) == sgn]
            if not keys:
                continue
            uidx = np.unique(np.concatenate([sel[k] for k in keys]))
            P = np.empty((len(uidx), len(VARIANTS)), np.float32)
            Bk = np.empty((len(uidx), len(VARIANTS)), np.int16)
            for a in range(0, len(uidx), CHUNK):
                res = SM.simulate(F, uidx[a:a + CHUNK], sgn, VARIANTS, cost)
                for vi, v in enumerate(VARIANTS):
                    P[a:a + CHUNK, vi] = res[v.name].pnl
                    Bk[a:a + CHUNK, vi] = res[v.name].bars
            for k in keys:
                pos = np.searchsorted(uidx, sel[k])
                m_ = len(pos)
                acc['sid'].append(np.full(m_, sid_of[k], np.int16)); acc['sym'].append(np.full(m_, n_s, np.int32))
                acc['grp'].append(np.full(m_, feats.GROUPS.index(g), np.int8)); acc['day'].append((F['t'][sel[k]] // 86400).astype(np.int32))
                acc['atrp'].append(F['atrp'][sel[k]].astype(np.float32)); acc['spy_up'].append(F['spy_up'][sel[k]].astype(np.float32))
                acc['vix_z'].append(F['vix_z'][sel[k]].astype(np.float32)); acc['rsi'].append(F['rsi14'][sel[k]].astype(np.float32))
                acc['pnl'].append(P[pos]); acc['bars'].append(Bk[pos])
        if (n_s + 1) % 300 == 0:
            print(f'  eventos: {n_s + 1}/{len(sym_list)} símbolos ({time.time() - t0:.0f}s)', flush=True)
    E = {k: np.concatenate(v) for k, v in acc.items()}
    E['ids'] = ids
    return E


def run(uni: Dict, data: Dict, out_dir: str):
    E = collect(uni, data)
    ids = E['ids']
    print(f"\n######## EMA 34 / 89 / 200 + RSI(14): {len(E['sid']):,} eventos · {len(DEFS)} setups × {len(VARIANTS)} salidas · referencia = entrada aleatoria en el mismo régimen y con la misma salida", flush=True)
    day, sym, sid = E['day'], E['sym'], E['sid']
    years = year_of(day)
    yrs = [y for y in sorted(set(years)) if 2017 <= y <= 2026]
    sid_idx = {k: i for i, k in enumerate(ids)}
    nF = len(feats.GROUPS)
    stock_g = [feats.GROUPS.index(g) for g in ('us_large', 'us_mid', 'us_small')]

    def block(mask, vi):
        x = E['pnl'][mask, vi]; ok = np.isfinite(x)
        return x[ok], day[mask][ok], sym[mask][ok], years[mask][ok]

    summary = []
    for vi, vn in enumerate(VNAMES):
        print(f'\n=== SALIDA: {vn} ===')
        print(f"   {'setup':58s} {'n':>7s} {'WR%':>5s} {'μ%':>6s} {'PF':>4s} {'t':>4s} | {'ref μ':>6s} {'Δμ':>6s} {'años Δ>0':>8s} | {'≤2023':>6s} {'2024+':>6s}")
        for k, (d, lbl, ref) in DEFS.items():
            m = sid == sid_idx[k]
            if m.sum() < 300:
                continue
            x, dd, ss, yy = block(m, vi)
            mb = sid == sid_idx['REF_' + ref]
            xb, db, sb, yb = block(mb, vi)
            if len(x) < 300 or len(xb) < 300:
                continue
            st = stats(x, dd, ss)
            dy = []
            for Y in yrs:
                a, b = x[yy == Y], xb[yb == Y]
                if len(a) >= 15 and len(b) >= 100:
                    dy.append(a.mean() - b.mean())
            pos = sum(v > 0 for v in dy)
            dev, rec = x[yy <= 2023], x[yy >= 2024]
            print(f"   {k:3s} {lbl[:54]:54s} {st['n']:7d} {st['wr']:5.1f} {st['mean']:+6.2f} {st['pf']:4.2f} {st['t_day']:4.1f} | {xb.mean():+6.2f} {st['mean'] - xb.mean():+6.2f} {pos:>4d}/{len(dy):<3d} | {dev.mean() if len(dev) else float('nan'):+6.2f} {rec.mean() if len(rec) else float('nan'):+6.2f}")
            summary.append((k, vi, st['n'], st['wr'], st['mean'], st['t_day'], st['mean'] - xb.mean(), pos, len(dy), (dev.mean() if len(dev) else np.nan), (rec.mean() if len(rec) else np.nan)))
        sys.stdout.flush()

    # ── candidatos que pasan un criterio estricto ──
    print('\n=== SETUPS QUE SUPERAN LA REFERENCIA con criterio estricto: Δμ > 0, t ≥ 3, mejora en ≥ 6 años y μ > 0 tanto hasta 2023 como desde 2024 ===')
    hits = [r for r in summary if r[6] > 0 and r[5] >= 3 and r[8] >= 6 and r[7] >= 6 and r[9] > 0 and r[10] > 0]
    if not hits:
        print('   NINGUNO')
    for r in sorted(hits, key=lambda r: -r[4]):
        print(f"   {r[0]:3s} · {DEFS[r[0]][1][:50]:50s} · {VNAMES[r[1]][:30]:30s} n={r[2]:6d} WR={r[3]:4.1f}% μ={r[4]:+.2f}% (Δ {r[6]:+.2f}) t={r[5]:.1f} años Δ>0 {r[7]}/{r[8]} · ≤2023 {r[9]:+.2f} 2024+ {r[10]:+.2f}")
    # ── contribución del RSI: cruce solo frente a cruce + RSI ──
    print('\n=== ¿APORTA EL RSI? media por setup frente al cruce SIN RSI (misma salida; Δ = con RSI − sin RSI) ===')
    for a, b in (('A1', 'A0'), ('A2', 'A0'), ('A3', 'A0'), ('A4', 'A0'), ('A8', 'A0'), ('C1', 'C0a'), ('C2', 'C0b'), ('B5', 'B1'), ('B6', 'B2'), ('B7', 'B4'), ('B9', 'B1')):
        line = f'   {a} ({DEFS[a][1][:36]}) vs {b}:'
        for vi in range(len(VARIANTS)):
            xa, _, _, _ = block(sid == sid_idx[a], vi); xb, _, _, _ = block(sid == sid_idx[b], vi)
            if len(xa) >= 300 and len(xb) >= 300:
                line += f" {xa.mean() - xb.mean():+.2f}"
            else:
                line += '    —'
        print(line + '   (columnas = las 6 salidas, en orden)')
    # ── desglose de la idea A1 y B7 ──
    for k in ('A1', 'A6', 'B7', 'D1'):
        m = sid == sid_idx[k]
        if m.sum() < 500:
            continue
        print(f'\n=== DESGLOSE de {k} · {DEFS[k][1]} (salida: stop móvil 3×ATR) ===')
        x, dd, ss, yy = block(m, 0)
        gg = E['grp'][m][np.isfinite(E['pnl'][m, 0])]
        up = E['spy_up'][m][np.isfinite(E['pnl'][m, 0])]
        print('   por año: ' + ' '.join(f"{str(Y)[2:]}:{x[yy == Y].mean():+.2f}(n={int((yy == Y).sum())})" for Y in yrs if (yy == Y).sum() >= 30))
        for gname in ('us_large', 'us_mid', 'us_small', 'etf', 'crypto', 'thematic', 'eu'):
            mg = gg == feats.GROUPS.index(gname)
            if mg.sum() >= 200:
                print(f"   {gname:9s} n={int(mg.sum()):6d} WR={np.mean(x[mg] > 0) * 100:4.1f}% μ={x[mg].mean():+.2f}%")
        for lbl, mm in (('S&P 500 sobre su SMA200', up == 1), ('S&P 500 bajo su SMA200', up == 0)):
            if mm.sum() >= 200:
                print(f"   {lbl:26s} n={int(mm.sum()):6d} WR={np.mean(x[mm] > 0) * 100:4.1f}% μ={x[mm].mean():+.2f}%")
        at = E['atrp'][m][np.isfinite(E['pnl'][m, 0])]
        for lo, hi in ((0, 2), (2, 3), (3, 4.5), (4.5, 99)):
            mm = (at >= lo) & (at < hi)
            if mm.sum() >= 200:
                print(f"   ATR% [{lo:g},{hi:g}) n={int(mm.sum()):6d} WR={np.mean(x[mm] > 0) * 100:4.1f}% μ={x[mm].mean():+.2f}%")
    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════
#  Segunda tanda (comando `ema2`): ¿sobrevive lo que mejor salió a tiempo en mercado, cartera, selección honesta y universo sin sesgo?
# ═════════════════════════════════════════════════════════════════════════
def run2(uni: Dict, data: Dict, out_dir: str):
    from research import portfolio_sim
    E = collect(uni, data)
    ids = E['ids']
    day, sym, sid, grp = E['day'], E['sym'], E['sid'], E['grp']
    years = year_of(day)
    sid_idx = {k: i for i, k in enumerate(ids)}
    spy = data['SPY']
    cal = (spy['t'] // 86400).astype(np.int64)
    ti_all = np.searchsorted(cal, day, side='left')
    rng = np.random.default_rng(7)
    prio = rng.random(len(day))
    g_stock = np.isin(grp, [feats.GROUPS.index(g) for g in ('us_large', 'us_mid', 'us_small')])
    g_etf = grp == feats.GROUPS.index('etf')
    print(f"\n######## EMA2 · {len(day):,} eventos", flush=True)

    def rows(k, extra=None):
        m = sid == sid_idx[k]
        return m if extra is None else (m & extra)

    # ── S1: exposición en el mercado ──
    print('\n[S1] TIEMPO EN EL MERCADO: sesiones medias por operación y rentabilidad por 20 sesiones (salida / setup / referencia con la misma salida)')
    keys = ['A1', 'A5', 'A6', 'A9', 'B1', 'B3', 'B4', 'B8', 'D1', 'C2']
    for vi in (1, 4, 5):
        print(f'  salida: {VNAMES[vi]}')
        for k in keys + ['REF_all', 'REF_stack', 'REF_up']:
            kk = k if k.startswith('REF') else k
            m = rows(kk)
            x = E['pnl'][m, vi]; b = E['bars'][m, vi].astype(float); ok = np.isfinite(x)
            if ok.sum() < 300:
                continue
            nm = DEFS[kk][1][:46] if kk in DEFS else BASES[kk[4:]][1]
            print(f"    {kk:9s} {nm:46s} n={int(ok.sum()):7d} WR={np.mean(x[ok] > 0) * 100:4.1f}% μ={x[ok].mean():+5.2f}% sesiones medias={b[ok].mean():5.1f} → μ por 20 sesiones {x[ok].mean() / b[ok].mean() * 20:+5.2f}%")

    # ── S2: carteras con capital limitado ──
    print('\n[S2] CARTERA con capital limitado (máx. 20 posiciones × 5 % del capital, prioridad al azar entre señales del mismo día; netas de costes; desde 2017-01). Referencia: S&P 500 comprar y mantener.')
    i0 = int(np.searchsorted(cal, day_of('2017-01-01')))
    c_ = spy['c'][i0:]; r_ = np.diff(c_) / c_[:-1]
    print(f"   S&P 500 comprar y mantener: anual {((c_[-1] / c_[0]) ** (252 / len(c_)) - 1) * 100:+.1f}% · caída máx. {float(((np.maximum.accumulate(c_) - c_) / np.maximum.accumulate(c_)).max()) * 100:.1f}% · Sharpe {r_.mean() / r_.std() * np.sqrt(252):.2f}")

    def port(k, vi, scope, M=20, f=0.05):
        m = rows(k, scope) & np.isfinite(E['pnl'][:, vi]) & (day >= day_of('2017-01-01'))
        if m.sum() < 500:
            return None
        res = np.where(m, E['pnl'][:, vi], np.nan)
        o = portfolio_sim(cal, ti_all, E['bars'][:, vi].astype(np.int64), res, prio, M, f)
        return o
    for scope_name, scope in (('TODAS las acciones y ETF', None), ('SOLO ETF (menos sesgo de supervivencia)', g_etf), ('SOLO acciones de EE.UU.', g_stock)):
        print(f'  — {scope_name}')
        for vi in (1, 4, 5):
            for k in ['A6', 'A9', 'B1', 'B4', 'B8', 'REF_stack', 'REF_up']:
                o = port(k, vi, scope)
                if o is None:
                    continue
                nm = DEFS[k][1][:34] if k in DEFS else BASES[k[4:]][1][:34]
                print(f"    {VNAMES[vi][:26]:26s} {k:9s} {nm:34s} anual {o['cagr']:+6.1f}% · caída {o['dd']:5.1f}% · Sharpe {o['sharpe']:4.2f} · meses+ {o['monthsPos']:3.0f}% · peor mes {o['worstMonth']:+5.1f}% · {o['tradesYear']:5.0f} op/año")
        sys.stdout.flush()

    # ── S3: selección HONESTA: se elige con 2016–2020 y se mide en 2021–2026 ──
    print('\n[S3] SELECCIÓN HONESTA: se elige el mejor (setup × salida) con los datos 2016–2020 (Δμ frente a su referencia con t ≥ 3) y se mide en 2021–2026, sin tocar nada')
    cands = []
    for vi in range(len(VARIANTS)):
        for k, (d, lbl, ref) in DEFS.items():
            if d < 0:
                continue
            for per, lo, hi in (('A', 2016, 2020), ('B', 2021, 2026)):
                pass
            m = (sid == sid_idx[k]) & np.isfinite(E['pnl'][:, vi])
            mb = (sid == sid_idx['REF_' + ref]) & np.isfinite(E['pnl'][:, vi])
            tr, te = m & (years <= 2020), m & (years >= 2021)
            trb, teb = mb & (years <= 2020), mb & (years >= 2021)
            if tr.sum() < 500 or te.sum() < 500 or trb.sum() < 500 or teb.sum() < 500:
                continue
            st = stats(E['pnl'][tr, vi], day[tr], sym[tr])
            cands.append((E['pnl'][tr, vi].mean() - E['pnl'][trb, vi].mean(), st['t_day'], k, vi,
                          E['pnl'][te, vi].mean() - E['pnl'][teb, vi].mean(), E['pnl'][te, vi].mean(), E['pnl'][teb, vi].mean(), int(te.sum())))
    cands.sort(key=lambda r: -r[0])
    print('   mejores 12 en 2016–2020 → su resultado en 2021–2026:')
    for d_tr, t_tr, k, vi, d_te, mu_te, mu_ref, n_te in [(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in cands if c[1] >= 3][:12]:
        print(f"     {k:3s} · {VNAMES[vi][:30]:30s} · 2016–20 Δμ {d_tr:+.2f} → 2021–26 Δμ {d_te:+.2f} (μ {mu_te:+.2f}% frente a la referencia {mu_ref:+.2f}%, n={n_te})")
    top = [c for c in cands if c[1] >= 3][:12]
    if top:
        print(f"   media de Δμ en 2021–2026 de esas 12 elecciones: {np.mean([c[4] for c in top]):+.2f}% (positivo = la elección anterior se sostiene)  ·  de las 12, {sum(c[4] > 0 for c in top)} siguen superando a la referencia")

    # ── S4: ¿importa en qué zona está el RSI? ──
    print('\n[S4] ¿IMPORTA LA ZONA DEL RSI(14) en la señal? μ por tramo de RSI (todos los cruces precio↑EMA34, A0, y triple alineación, B4)')
    for k, vi in (('A0', 1), ('A0', 5), ('B4', 5), ('B1', 5)):
        m = rows(k) & np.isfinite(E['pnl'][:, vi])
        r_ = E['rsi'][m]; x = E['pnl'][m, vi]
        line = f"   {k} / {VNAMES[vi][:26]:26s}:"
        for lo, hi in ((0, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 70), (70, 101)):
            mm = (r_ >= lo) & (r_ < hi)
            if mm.sum() >= 300:
                line += f" [{lo},{hi}) n={int(mm.sum()):6d} μ={x[mm].mean():+.2f}% |"
        print(line)
    sys.stdout.flush()
