"""Estrategias de otras familias, medidas con el mismo rigor que la regla de rebote.

Familias: factores de sección cruzada (momento, reversión, cercanía al máximo, baja volatilidad), eventos (huecos con volumen, rupturas de máximo anual) y mercado (tendencia, pánico). Todo en
carteras diarias con los precios reales: la decisión se toma al cierre, se ejecuta en la apertura siguiente, se paga el coste por cada lado y no hay apalancamiento. Los parámetros están
fijados de antemano (no se ajusta ninguno) y todas se comparan con mantener el universo líquido a partes iguales.

Qué se mide de cada una: rentabilidad anual, caída máxima, Sharpe, beta y alfa frente al universo a partes iguales (con la t del alfa por bloques de 10 sesiones), correlación con la regla de
rebote, y qué ocurre al mezclarlas con el mismo riesgo. Es DESCRIPTIVO: son unas 30 carteras miradas a la vez, así que una t de alfa por debajo de ~3,1 no distingue entre mérito y suerte
de elegir la mejor de varias.

  python -m vp factores [--out data]"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import cartera, ml, search

TOP_N = 300                    # las más negociadas de cada día (dinero medio de 60 sesiones, con datos hasta ese día)
PRICE_MIN = 5.0
COST_SIDE = 0.075 / 100.0      # por lado: ida y vuelta 0,15 %, como la regla de rebote
N_XS = 30                      # valores de las carteras de sección cruzada
SLOTS = 20                     # huecos de las carteras de eventos (cada uno el 1/20 de la cuenta)
BLOCK = 10                     # sesiones por bloque para la t del alfa
START = '2020-01-02'           # ventana común con la regla de rebote (el primer año con fiabilidad fuera de muestra)
LEV_COST = 0.04                # coste anual del dinero prestado al apalancar (sobre la parte prestada)
DD_TARGET = 0.25               # caída máxima a la que se compara la rentabilidad de todo (con apalancamiento o a medio gas)
FAMILY_T = 3.1                 # t mínima del alfa para que una de ~30 carteras destaque con p < 0,05 en el conjunto (Bonferroni)
ASSETS = ('SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD', 'ETH-USD', 'TQQQ', 'UPRO', 'SOXL')
ASSET_COST = {'BTC-USD': 0.15 / 100.0, 'ETH-USD': 0.15 / 100.0}     # por lado; el resto de activos, COST_SIDE
TREND_LEN = 200                # media de la tendencia (sesiones)
TREND_SIGNAL = {'TQQQ': 'QQQ', 'UPRO': 'SPY'}      # el ETF apalancado se compra cuando su subyacente supera su media

DESC = {
    'mercado': 'universo líquido a partes iguales, reequilibrado cada mes (la referencia)',
    'mom_12_1': 'momento: los 30 que más han subido de hace 12 meses a hace 1 mes, cada mes',
    'mom_6_1': 'momento: los 30 que más han subido de hace 6 meses a hace 1 mes, cada mes',
    'rev_1m': 'reversión mensual: los 30 que más han caído en el último mes, cada mes',
    'rev_1s': 'reversión semanal: los 30 que más han caído en la última semana, cada semana',
    'cerca_max': 'cercanía al máximo anual: los 30 más cerca de su máximo de 252 sesiones, cada mes',
    'baja_vol': 'baja volatilidad: los 30 menos volátiles de 60 sesiones, cada mes',
    'tendencia': 'tendencia del mercado: el universo a partes iguales solo mientras supera su media de 200 sesiones',
    'panico_todo': 'pánico: el universo a partes iguales mientras la amplitud (% sobre su media de 200) baja de 30 % hasta que pasa de 45 %',
    'panico_30': 'pánico: en esos mismos tramos, los 30 que más han caído en 20 sesiones',
    'hueco_alza': 'hueco al alza con volumen: sube ≥ 4 % en la apertura, con volumen ≥ 2 veces la media y cierra por encima de la apertura; se mantiene 20 sesiones',
    'hueco_baja': 'hueco a la baja con volumen: cae ≥ 5 % en la apertura con volumen ≥ 2 veces la media; se mantiene 10 sesiones',
    'ruptura_max': 'ruptura del máximo anual con volumen ≥ 1,5 veces la media; se mantiene 20 sesiones',
}


# ── panel ─────────────────────────────────────────────────────────────────
class Panel:
    """Precios alineados por día (filas) y valor (columnas), y qué valores son operables cada día."""

    def __init__(self, bars: Dict[str, Dict[str, np.ndarray]], join: Optional[Dict[str, str]] = None, top: int = TOP_N):
        import pandas as pd
        self.syms = sorted(bars)
        days = np.unique(np.concatenate([np.asarray(b['t']) // 86400 for b in bars.values()])).astype(np.int64)
        self.days = days
        self.date = np.array([str(x) for x in days.astype('datetime64[D]')])
        nd, ns = len(days), len(self.syms)
        self.nd, self.ns = nd, ns
        self.O, self.H, self.L, self.C, self.V = (np.full((nd, ns), np.nan) for _ in range(5))
        for j, s in enumerate(self.syms):
            b = bars[s]
            di = np.searchsorted(days, np.asarray(b['t']) // 86400)
            for arr, k in ((self.O, 'o'), (self.H, 'h'), (self.L, 'l'), (self.C, 'c'), (self.V, 'v')):
                arr[di, j] = b[k]
        member = np.ones((nd, ns), bool)
        if join:
            for j, s in enumerate(self.syms):
                if join.get(s):
                    member[:, j] = days >= int(np.datetime64(join[s], 'D').astype(np.int64))
        self.member = member
        dv = pd.DataFrame(self.C * self.V).rolling(60, min_periods=36).mean().to_numpy()
        ok = member & np.isfinite(dv) & np.isfinite(self.C) & (self.C >= PRICE_MIN)
        elig = np.zeros((nd, ns), bool)
        for d in range(nd):
            v = np.where(ok[d], dv[d], -np.inf)
            n_ok = int(ok[d].sum())
            if n_ok == 0:
                continue
            if n_ok <= top:
                elig[d] = ok[d]
            else:
                thr = np.partition(v, -top)[-top]
                elig[d] = ok[d] & (v >= thr)
        self.elig = elig
        with np.errstate(invalid='ignore', divide='ignore'):
            self.ret = np.full((nd, ns), np.nan)
            self.ret[1:] = self.C[1:] / self.C[:-1] - 1.0
        # rendimiento del universo a partes iguales: los valores operables al cierre anterior (sin costes)
        use = elig[:-1] & np.isfinite(self.ret[1:])
        cnt = use.sum(1)
        self.mret = np.zeros(nd)
        self.mret[1:] = np.where(cnt > 0, np.where(use, self.ret[1:], 0.0).sum(1) / np.maximum(cnt, 1), 0.0)
        self.year = np.array([int(x[:4]) for x in self.date])


def _roll(x: np.ndarray, w: int, fn: str, minp: Optional[int] = None) -> np.ndarray:
    import pandas as pd
    r = pd.DataFrame(x).rolling(w, min_periods=minp or max(2, int(0.8 * w)))
    return getattr(r, fn)().to_numpy()


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


def last_of(days: np.ndarray, freq: str) -> np.ndarray:
    """Último día de negociación de cada mes ('m') o de cada semana ('w')."""
    key = days.astype('datetime64[D]').astype('datetime64[M]').astype(np.int64) if freq == 'm' else (days + 3) // 7
    out = np.zeros(len(days), bool)
    out[:-1] = key[1:] != key[:-1]
    return out


# ── carteras: objetivos de peso y órdenes ─────────────────────────────────
def xs_strategy(P: Panel, score: np.ndarray, n: int, reb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cada día de `reb`, a partes iguales los `n` valores operables de mayor puntuación (si hay menos de `n` con puntuación, todo a caja); se reequilibra todo ese día."""
    tgt = np.zeros((P.nd, P.ns))
    trade = np.zeros((P.nd, P.ns), bool)
    for d in np.flatnonzero(reb):
        s = np.where(P.elig[d] & np.isfinite(score[d]), score[d], -np.inf)
        if int(np.isfinite(s).sum()) >= n:
            tgt[d, np.argpartition(-s, n - 1)[:n]] = 1.0 / n
        trade[d] = True
    return tgt, trade


def ew_strategy(P: Panel, on: np.ndarray, reb: np.ndarray, pick: Optional[np.ndarray] = None, n: int = N_XS) -> Tuple[np.ndarray, np.ndarray]:
    """Universo operable a partes iguales mientras `on` (o, con `pick`, los `n` de mayor puntuación en el momento de entrar), reequilibrando en `reb` y al cambiar de estado."""
    tgt = np.zeros((P.nd, P.ns))
    trade = np.zeros((P.nd, P.ns), bool)
    prev = False
    cur = np.zeros(P.ns)
    for d in range(P.nd):
        flip = bool(on[d]) != prev
        if on[d] and (flip or (pick is None and reb[d])):
            m = P.elig[d]
            if pick is None:
                cur = np.where(m, 1.0 / max(1, int(m.sum())), 0.0)
            else:
                s = np.where(m & np.isfinite(pick[d]), pick[d], -np.inf)
                cur = np.zeros(P.ns)
                if int(np.isfinite(s).sum()) >= n:
                    cur[np.argpartition(-s, n - 1)[:n]] = 1.0 / n
            tgt[d], trade[d] = cur, True
        elif flip and not on[d]:
            trade[d] = True
            cur = np.zeros(P.ns)
        prev = bool(on[d])
    return tgt, trade


def event_strategy(P: Panel, sig: np.ndarray, score: np.ndarray, hold: int, nmax: int = SLOTS) -> Tuple[np.ndarray, np.ndarray]:
    """Cada día entran, hasta llenar los `nmax` huecos y por puntuación descendente, los valores operables con señal que no estén ya dentro; cada uno pesa 1/nmax y sale `hold` sesiones después."""
    tgt = np.zeros((P.nd, P.ns))
    trade = np.zeros((P.nd, P.ns), bool)
    pos: Dict[int, int] = {}
    for d in range(P.nd):
        out_today = [j for j, xd in pos.items() if xd == d]
        for j in out_today:
            tgt[d, j] = 0.0
            trade[d, j] = True
            del pos[j]
        free = nmax - len(pos)
        if free > 0:
            cand = np.flatnonzero(sig[d] & P.elig[d] & np.isfinite(score[d]))
            cand = [j for j in cand[np.argsort(-score[d, cand], kind='stable')] if j not in pos and j not in out_today][:free]
            for j in cand:
                tgt[d, j] = 1.0 / nmax
                trade[d, j] = True
                pos[int(j)] = d + hold
    return tgt, trade


def run_weights(P: Panel, tgt: np.ndarray, trade: np.ndarray, cost: float = COST_SIDE) -> Dict:
    """La cartera día a día. La orden decidida al cierre de t−1 se ejecuta en la apertura de t: lo que se queda gana de cierre a cierre, lo que entra de apertura a cierre, lo que sale de cierre a
    apertura. Los valores sin orden ese día se dejan correr (sus pesos derivan). Sin apalancamiento: lo que entra no puede superar la caja que queda tras las salidas. `cost` por lado sobre lo operado."""
    nd, ns = tgt.shape
    ret = np.zeros(nd)
    expo = np.zeros(nd)
    turn = np.zeros(nd)
    w = np.zeros(ns)
    c, o = P.C, P.O
    for t in range(1, nd):
        want = np.where(trade[t - 1], tgt[t - 1], w)
        keep = np.minimum(w, want)
        add = np.maximum(want - w, 0.0)
        cut = np.maximum(w - want, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            r_keep = np.nan_to_num(c[t] / c[t - 1] - 1.0)
            r_add = np.nan_to_num(c[t] / o[t] - 1.0)
            r_cut = np.nan_to_num(o[t] / c[t - 1] - 1.0)
        avail = max(0.0, 1.0 - float(w.sum()) + float(cut @ (1.0 + r_cut)) - cost * float(cut.sum()))     # caja tras las ventas de la apertura
        if add.sum() * (1.0 + cost) > avail:
            add = add * (avail / (add.sum() * (1.0 + cost)))
        traded = float(add.sum() + cut.sum())
        r = float(keep @ r_keep + add @ r_add + cut @ r_cut) - cost * traded
        value = keep * (1.0 + r_keep) + add * (1.0 + r_add)
        w = value / (1.0 + r) if (1.0 + r) > 0 else np.zeros(ns)
        ret[t], turn[t], expo[t] = r, traded, float(w.sum())
    return {'ret': ret, 'expo': expo, 'turn': turn}


# ── las carteras ──────────────────────────────────────────────────────────
def strategies(P: Panel) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Todas las carteras con sus parámetros fijados de antemano. Cada puntuación usa solo datos hasta el cierre del día."""
    with np.errstate(invalid='ignore', divide='ignore'):
        C, V = P.C, P.V
        mon, wk = last_of(P.days, 'm'), last_of(P.days, 'w')
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        out['mercado'] = ew_strategy(P, np.ones(P.nd, bool), mon)
        out['mom_12_1'] = xs_strategy(P, _lag(C, 21) / _lag(C, 252) - 1.0, N_XS, mon)
        out['mom_6_1'] = xs_strategy(P, _lag(C, 21) / _lag(C, 126) - 1.0, N_XS, mon)
        out['rev_1m'] = xs_strategy(P, -(C / _lag(C, 21) - 1.0), N_XS, mon)
        out['rev_1s'] = xs_strategy(P, -(C / _lag(C, 5) - 1.0), N_XS, wk)
        out['cerca_max'] = xs_strategy(P, C / _roll(C, 252, 'max', 200), N_XS, mon)
        out['baja_vol'] = xs_strategy(P, -_roll(np.where(np.isfinite(P.ret), P.ret, np.nan), 60, 'std'), N_XS, mon)
        idx = np.cumprod(1.0 + P.mret)
        ma = _roll(idx[:, None], 200, 'mean', 160)[:, 0]
        out['tendencia'] = ew_strategy(P, np.isfinite(ma) & (idx > ma), mon)
        ma200 = _roll(C, 200, 'mean', 160)
        okb = P.elig & np.isfinite(ma200)
        breadth = np.where(okb.sum(1) > 0, (okb & (C > ma200)).sum(1) / np.maximum(okb.sum(1), 1), np.nan)
        on = np.zeros(P.nd, bool)
        state = False
        for d in range(P.nd):
            if np.isfinite(breadth[d]):
                state = (breadth[d] < 0.30) if not state else not (breadth[d] > 0.45)
            on[d] = state
        out['panico_todo'] = ew_strategy(P, on, mon & False)
        out['panico_30'] = ew_strategy(P, on, mon & False, pick=-(C / _lag(C, 20) - 1.0), n=N_XS)
        av20 = _lag(_roll(V, 20, 'mean', 15), 1)
        vr = V / av20
        gap = P.O / _lag(C, 1) - 1.0
        out['hueco_alza'] = event_strategy(P, (gap >= 0.04) & (vr >= 2.0) & (C > P.O), gap, 20)
        out['hueco_baja'] = event_strategy(P, (gap <= -0.05) & (vr >= 2.0), -gap, 10)
        hi = _lag(_roll(C, 252, 'max', 200), 1)
        out['ruptura_max'] = event_strategy(P, (C > hi) & (vr >= 1.5), vr, 20)
    return out


# ── estadística ───────────────────────────────────────────────────────────
def _r(x, d=2):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def block_t(x: np.ndarray, block: int = BLOCK) -> float:
    """t de la media de `x` con errores agrupados en bloques de `block` sesiones consecutivas."""
    n = len(x) // block * block
    if n < 5 * block:
        return float('nan')
    b = x[:n].reshape(-1, block).sum(1)
    sd = b.std(ddof=1)
    return float(b.mean() / (sd / np.sqrt(len(b)))) if sd > 0 else float('nan')


def alpha_beta(rs: np.ndarray, rm: np.ndarray) -> Dict:
    """Regresión diaria de la cartera sobre el universo a partes iguales: beta, alfa anualizado (puntos %) y su t por bloques; y el exceso simple (beta = 1)."""
    ok = np.isfinite(rs) & np.isfinite(rm)
    rs, rm = rs[ok], rm[ok]
    if len(rs) < 100 or rm.std() == 0:
        return {'beta': None, 'alfa': None, 't_alfa': None, 'exceso': None, 't_exceso': None}
    beta = float(np.cov(rs, rm)[0, 1] / rm.var(ddof=1))
    res = rs - beta * rm
    ex = rs - rm
    return {'beta': _r(beta, 2), 'alfa': _r(res.mean() * 252 * 100, 1), 't_alfa': _r(block_t(res), 2), 'exceso': _r(ex.mean() * 252 * 100, 1), 't_exceso': _r(block_t(ex), 2)}


def stats(ret: np.ndarray, year: np.ndarray, expo: Optional[np.ndarray] = None) -> Dict:
    eq = np.cumprod(1.0 + ret)
    m = cartera.metrics(eq / eq[0] if eq[0] else eq, year)
    if expo is not None and len(expo):
        m['exposicion_media'] = _r(float(np.mean(expo)) * 100, 1)
    return m


def lever(ret: np.ndarray, L: float, fin: float = LEV_COST) -> np.ndarray:
    """Rendimientos diarios con factor L: por encima de 1 se paga `fin` anual por lo prestado; por debajo de 1, el resto en caja sin interés."""
    return L * ret - max(L - 1.0, 0.0) * fin / 252.0


def at_drawdown(ret: np.ndarray, target: float = DD_TARGET) -> Dict:
    """Rentabilidad anual que se obtiene escalando la cartera hasta que su caída máxima sea `target` (biseccion sobre el factor L; por encima de 1, con coste del dinero prestado)."""
    def dd(L):
        eq = np.cumprod(1.0 + np.maximum(lever(ret, L), -0.999))
        return float(-(eq / np.maximum.accumulate(eq) - 1.0).min())
    lo, hi = 0.05, 8.0
    if dd(hi) < target:
        L = hi
    elif dd(lo) > target:
        return {'L': None, 'anual': None}
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if dd(mid) < target else (lo, mid)
        L = (lo + hi) / 2
    eq = np.cumprod(1.0 + np.maximum(lever(ret, L), -0.999))
    yrs = len(ret) / 252.0
    return {'L': _r(L, 2), 'anual': _r((eq[-1] ** (1.0 / yrs) - 1.0) * 100, 1)}


def blend(rets: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    """Mezcla a igual riesgo: cada cartera pesa lo inverso de su volatilidad diaria, con pesos que suman 1 (sin apalancamiento)."""
    live = [k for k in names if rets[k].std() > 1e-6]                         # una cartera que nunca opera no entra (pesaría infinito)
    inv = np.array([1.0 / rets[k].std() for k in live])
    w = inv / inv.sum()
    return sum(w[i] * rets[k] for i, k in enumerate(live)) if live else np.zeros(len(next(iter(rets.values()))))


# ── tendencia en activos de alta rentabilidad ──────────────────────────────
class AssetPanel:
    """Los activos (ETF, cripto, ETF apalancados) alineados con el calendario de la bolsa de acciones: lo que el motor de carteras necesita (aperturas y cierres)."""

    def __init__(self, P: Panel, bars: Dict[str, Dict[str, np.ndarray]]):
        self.syms = sorted(bars)
        self.days, self.date, self.nd, self.ns, self.year = P.days, P.date, P.nd, len(self.syms), P.year
        self.O, self.C = np.full((self.nd, self.ns), np.nan), np.full((self.nd, self.ns), np.nan)
        for j, s_ in enumerate(self.syms):
            t = np.asarray(bars[s_]['t']) // 86400
            di = np.searchsorted(self.days, t)
            ok = (di < self.nd) & (self.days[np.minimum(di, self.nd - 1)] == t)
            self.O[di[ok], j] = np.asarray(bars[s_]['o'])[ok]
            self.C[di[ok], j] = np.asarray(bars[s_]['c'])[ok]


def trend_weights(A: AssetPanel, j: int, sig: Optional[int] = None, length: int = TREND_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """Todo en el activo `j` mientras el cierre de `sig` (por defecto el mismo activo) esté por encima de su media de `length` sesiones; la orden se da al cierre y se ejecuta en la apertura siguiente."""
    import pandas as pd
    c = pd.Series(A.C[:, j if sig is None else sig]).ffill()
    ma = c.rolling(length, min_periods=int(0.8 * length)).mean()
    on = (c > ma).to_numpy() & ma.notna().to_numpy()
    tgt = np.zeros((A.nd, A.ns))
    trade = np.zeros((A.nd, A.ns), bool)
    prev = False
    for d in range(A.nd):
        if bool(on[d]) != prev and np.isfinite(A.O[min(d + 1, A.nd - 1), j]):
            trade[d, j] = True
            tgt[d, j] = 1.0 if on[d] else 0.0
            prev = bool(on[d])
    return tgt, trade


def asset_sleeves(P: Panel, bars: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Comprar y mantener cada activo (bh_) y comprarlo solo con tendencia alcista (tend_), y una cartera de tendencias a igual riesgo con los activos sin apalancar."""
    A = AssetPanel(P, bars)
    idx = {s_: j for j, s_ in enumerate(A.syms)}
    rets: Dict[str, np.ndarray] = {}
    expos: Dict[str, np.ndarray] = {}
    for s_, j in idx.items():
        cost = ASSET_COST.get(s_, COST_SIDE)
        first = int(np.flatnonzero(np.isfinite(A.O[:, j]) & np.isfinite(A.C[:, j]))[0]) if np.isfinite(A.C[:, j]).any() else A.nd
        tg, tr = np.zeros((A.nd, A.ns)), np.zeros((A.nd, A.ns), bool)
        if first < A.nd - 1:
            tg[first, j], tr[first, j] = 1.0, True
        r = run_weights(A, tg, tr, cost)
        rets['bh_' + s_], expos['bh_' + s_] = r['ret'], r['expo']
        sig = idx.get(TREND_SIGNAL.get(s_, ''), None)
        tg, tr = trend_weights(A, j, sig)
        r = run_weights(A, tg, tr, cost)
        rets['tend_' + s_], expos['tend_' + s_] = r['ret'], r['expo']
    base = [k_ for k_ in ('SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD', 'ETH-USD') if 'tend_' + k_ in rets]
    if len(base) >= 3:
        rets['tend_cartera'] = blend(rets, ['tend_' + k_ for k_ in base])
        expos['tend_cartera'] = np.mean([expos['tend_' + k_] for k_ in base], axis=0)
    return rets, expos


TREND_CORE = ('SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD', 'ETH-USD')
PERIODOS = {'2017-2019': ('2017-01-01', '2019-12-31'), '2020-2022': ('2020-01-01', '2022-12-31'), '2023-2026': ('2023-01-01', '2026-12-31'), '2020-2026': ('2020-01-01', '2026-12-31')}


def asset_variants(P: Panel, bars: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Robustez de la cartera de tendencias: otras medias (100, 150, 250 sesiones), sin cripto y solo con cripto. Mismas reglas, a igual riesgo, sin ajustar nada."""
    A = AssetPanel(P, bars)
    idx = {s_: j for j, s_ in enumerate(A.syms)}
    cache: Dict[Tuple[str, int], np.ndarray] = {}

    def one(sym: str, length: int) -> np.ndarray:
        if (sym, length) not in cache:
            tg, tr = trend_weights(A, idx[sym], None, length)
            cache[(sym, length)] = run_weights(A, tg, tr, ASSET_COST.get(sym, COST_SIDE))['ret']
        return cache[(sym, length)]
    out: Dict[str, np.ndarray] = {}
    have = [k_ for k_ in TREND_CORE if k_ in idx]
    for length in (100, 150, 250):
        out[f'tend_cartera_ma{length}'] = blend({k_: one(k_, length) for k_ in have}, have)
    nocry = [k_ for k_ in have if k_ not in ('BTC-USD', 'ETH-USD')]
    cry = [k_ for k_ in have if k_ in ('BTC-USD', 'ETH-USD')]
    if len(nocry) >= 2:
        out['tend_cartera_sin_cripto'] = blend({k_: one(k_, TREND_LEN) for k_ in nocry}, nocry)
    if cry:
        out['tend_cripto'] = blend({k_: one(k_, TREND_LEN) for k_ in cry}, cry)
    return out


def periods(ret: np.ndarray, P: Panel) -> Dict[str, Dict]:
    """Rentabilidad anual, caída y Sharpe de una cartera en cada subperiodo (la cuenta empieza en 1 al inicio de cada uno)."""
    out = {}
    for nm, (a, b) in PERIODOS.items():
        ix = np.flatnonzero((P.date >= a) & (P.date <= b))
        if len(ix) < 150 or not (np.abs(ret[ix]) > 0).any():
            continue
        m = stats(ret[ix], P.year[ix])
        out[nm] = {'anual': m['anual'], 'caida_max': m['caida_max'], 'sharpe': m['sharpe']}
    return out


# ── la regla de rebote como cartera diaria ────────────────────────────────
RULE_SLEEVES = (('regla_todas', 'imp_bajo200', 'esc', True), ('regla_plan', 'imp_bajo200', 'esc', False), ('regla_bajistas', 'imp_bajistas', 'esc', True), ('regla_rsi2', 'imp_rsi2', 'esc', True),
                ('regla_run5', 'imp_bajo200', 'run5', True))
DESC.update({
    'regla_todas': 'regla de rebote (impulso alto y bajo su media de 200) con la escalera, todas las señales',
    'regla_plan': 'la misma regla con el plan de 5 huecos (orden por volatilidad)',
    'regla_bajistas': 'variante de entrada: impulso alto y 2 o más señales bajistas; escalera, todas las señales',
    'regla_rsi2': 'variante de entrada: impulso alto y RSI(2) ≤ 10; escalera, todas las señales',
    'regla_run5': 'la regla base con la salida «run5» (la que más da por operación del menú; elegida mirando el menú de 13 salidas); todas las señales',
    'relleno_regla_todas': 'regla de rebote, todas las señales, con la caja no invertida puesta en el universo a partes iguales',
    'relleno_regla_plan': 'plan de 5 huecos, con la caja no invertida puesta en el universo a partes iguales',
    'bh_': 'comprar y mantener el activo (SPY, QQQ: índices; TLT: bonos largos; GLD: oro; BTC-USD y ETH-USD: cripto; TQQQ, UPRO, SOXL: ETF apalancados ×3)',
    'tend_': 'solo con tendencia: el activo mientras su cierre supere su media de 200 sesiones (TQQQ con la media de QQQ y UPRO con la de SPY), si no, en caja',
    'tend_cartera': 'cartera de tendencias a igual riesgo: SPY, QQQ, TLT, GLD, BTC-USD y ETH-USD (cada uno con su propia tendencia)',
    'vt_regla_todas': 'regla de rebote (todas las señales) con control de volatilidad: apalancamiento = 15 % / volatilidad de las últimas 20 sesiones, entre 0 y 1,5',
    'vt_mercado': 'universo a partes iguales con el mismo control de volatilidad',
    'vt_relleno_regla_todas': 'regla de rebote con la caja en el universo y el mismo control de volatilidad',
})


def rule_returns(bars: Dict[str, Dict[str, np.ndarray]], join: Optional[Dict[str, str]], dates: np.ndarray, workers: int = 1, k: int = ml.K, verbose: bool = False) -> Dict[str, np.ndarray]:
    """Rendimientos diarios y exposición, alineados con `dates`, de la regla de rebote y sus variantes con el simulador de la cartera simulada (orden por volatilidad). Todas las señales = sin límite de posiciones."""
    D, _, _ = search.build(bars, workers, k, verbose, search.MIN_DAY, search.MIN_COMBO, join, search.COST, library=False)
    jobs = [(s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if s in D.syms]
    W: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s, w in ex.map(cartera._sym_work, jobs, chunksize=8):
                W[s] = w
    else:
        for jb in jobs:
            s, w = cartera._sym_work(jb)
            W[s] = w
    Wd = cartera.World(D, W)
    masks = {r['nombre']: Wd.rule_mask(r['regla']) for r in search.ENTRADAS}
    ds = np.array([str(x) for x in D.date])
    d0 = int(np.searchsorted(ds, START))
    rng = np.random.default_rng(0)
    out = {}
    for nm, rule, ex_, todas in RULE_SLEEVES:
        s = cartera.simulate(Wd, masks[rule], ex_, 'vol', rng, d0=d0, **({'max_pos': 10 ** 6, 'corr_min': 2.0} if todas else {}))
        eq = s['eq']
        r = np.concatenate([[0.0], eq[1:] / eq[:-1] - 1.0])
        sub = ds[s['d0']:s['dend'] + 1]
        pos = np.searchsorted(dates, sub)
        okp = (pos < len(dates)) & (dates[np.minimum(pos, len(dates) - 1)] == sub)
        out[nm] = np.zeros(len(dates))
        out[nm][pos[okp]] = r[okp]
        out[nm + '_expo'] = np.zeros(len(dates))
        out[nm + '_expo'][pos[okp]] = s['expo'][okp]
    return out


def voltarget(ret: np.ndarray, target: float = 0.15, win: int = 20, cap: float = 1.5) -> np.ndarray:
    """Control de volatilidad: cada día el factor es `target` dividido por la volatilidad anualizada de las `win` sesiones anteriores (sin mirar el día), entre 0 y `cap`; por encima de 1 se paga el dinero prestado."""
    import pandas as pd
    sd = pd.Series(ret).rolling(win, min_periods=win).std().shift(1).to_numpy() * np.sqrt(252)
    lev = np.where(np.isfinite(sd) & (sd > 0), np.minimum(cap, target / np.maximum(sd, 1e-9)), 1.0)
    return lev * ret - np.maximum(lev - 1.0, 0.0) * LEV_COST / 252.0


# ── conjunto ──────────────────────────────────────────────────────────────
def analyze(bars: Dict[str, Dict[str, np.ndarray]], join: Optional[Dict[str, str]] = None, workers: int = 1, with_rule: bool = True, verbose: bool = False, k: int = ml.K,
            assets: Optional[Dict[str, Dict[str, np.ndarray]]] = None) -> Dict:
    t0 = time.time()
    P = Panel(bars, join)
    if verbose:
        print(f'  panel: {P.nd} días · {P.ns} valores · {int(P.elig.sum(1).mean())} operables por día · {time.time() - t0:.0f}s', flush=True)
    strat = strategies(P)
    runs = {}
    for nm, (tgt, trade) in strat.items():
        runs[nm] = run_weights(P, tgt, trade)
        if verbose:
            print(f'  {nm:12s} {time.time() - t0:.0f}s', flush=True)
    rets = {nm: r['ret'] for nm, r in runs.items()}
    expos = {nm: r['expo'] for nm, r in runs.items()}
    asset_names: List[str] = []
    if assets:
        ar, ae = asset_sleeves(P, assets)
        for nm in ar:
            rets[nm], expos[nm] = ar[nm], ae[nm]
            asset_names.append(nm)
    rule_names: List[str] = []
    if with_rule:
        rr = rule_returns(bars, join, P.date, workers, k, verbose)
        for nm, *_ in RULE_SLEEVES:
            rets[nm], expos[nm] = rr[nm], rr[nm + '_expo']
            rule_names.append(nm)
        for base in ('regla_todas', 'regla_plan'):
            prev = np.concatenate([[0.0], expos[base][:-1]])
            rets['relleno_' + base] = rets[base] + (1.0 - prev) * rets['mercado']
        rets['vt_regla_todas'] = voltarget(rets['regla_todas'])
        rets['vt_mercado'] = voltarget(rets['mercado'])
        rets['vt_relleno_regla_todas'] = voltarget(rets['relleno_regla_todas'])
    variants = asset_variants(P, assets) if assets else {}
    first = {nm: int(np.flatnonzero(np.abs(expos[nm]) > 0)[0]) if (np.abs(expos[nm]) > 0).any() else P.nd for nm in expos}
    d_start = int(np.searchsorted(P.date, START))
    sl = slice(d_start, P.nd)
    yr, rm = P.year[sl], P.mret[sl]
    out: Dict = {'meta': {'generated': int(time.time()), 'universe': P.ns, 'from': str(P.date[0]), 'to': str(P.date[-1]), 'operables_por_dia': int(P.elig.sum(1).mean()), 'coste_lado': COST_SIDE * 100,
                          'top': TOP_N, 'n_xs': N_XS, 'huecos': SLOTS, 'ventana': START, 'miembros': 'en la fecha' if join else 'actuales', 'lev_coste': LEV_COST, 'caida_objetivo': DD_TARGET * 100},
           'desc': {k_: v for k_, v in DESC.items() if k_ in rets or k_ in ('bh_', 'tend_') and asset_names}, 'carteras': {}, 'neutrales': {}}
    for nm in rets:
        r = rets[nm][sl]
        row = {'ventana': stats(r, yr, expos[nm][sl] if nm in expos else None), **alpha_beta(r, rm), 'a_25': at_drawdown(r)}
        if nm in runs:
            row['rotacion_anual'] = _r(float(runs[nm]['turn'][sl].sum()) / (len(r) / 252.0) * 100, 0)
        if nm in first and first[nm] < d_start:
            f0 = first[nm]
            row['desde_inicio'] = {**stats(rets[nm][f0:], P.year[f0:], expos[nm][f0:]), 'desde': str(P.date[f0])}
        out['carteras'][nm] = row
        b = row['beta']
        if b is not None and nm != 'mercado' and not nm.startswith(('vt_', 'relleno_', 'bh_')) and abs(b) > 0.02:
            neu = r - b * rm                                                                 # alfa puro: la cartera menos su beta en el universo (sin coste de la cobertura)
            out['neutrales'][nm] = {'ventana': stats(neu, yr), **{k_: v for k_, v in alpha_beta(neu, rm).items() if k_ in ('alfa', 't_alfa')}, 'a_25': at_drawdown(neu), 'beta': b}
    if variants:
        keyrows = ['mercado', 'bh_SPY', 'bh_QQQ', 'tend_QQQ', 'tend_cartera', 'bh_BTC-USD', 'tend_BTC-USD', 'tend_TQQQ']
        series = {nm: rets[nm] for nm in keyrows if nm in rets}
        series.update(variants)
        out['robustez_tendencia'] = {'variantes': {nm: {'ventana': stats(v[sl], yr), **{k_: x for k_, x in alpha_beta(v[sl], rm).items() if k_ in ('beta', 'alfa', 't_alfa')}, 'a_25': at_drawdown(v[sl])} for nm, v in variants.items()},
                                     'periodos': {nm: periods(v, P) for nm, v in series.items()}}
    if with_rule:
        names = [n for n in strat if n != 'mercado']
        rl = {nm: rets[nm][sl] for nm in names + rule_names + ['mercado']}
        with np.errstate(invalid='ignore', divide='ignore'):
            out['correlacion_regla'] = {nm: _r(float(np.corrcoef(rl[nm], rl['regla_todas'])[0, 1]), 2) for nm in names + rule_names[1:] + ['mercado']}
        mixes = {}
        for nm in names + rule_names[1:] + ['mercado']:
            b2 = blend(rl, ['regla_todas', nm])
            mixes[f'regla_todas+{nm}'] = {'ventana': stats(b2, yr), **alpha_beta(b2, rm), 'a_25': at_drawdown(b2)}
        if 'tend_cartera' in rets:
            rl['tend_cartera'] = rets['tend_cartera'][sl]
            out['correlacion_regla']['tend_cartera'] = _r(float(np.corrcoef(rl['tend_cartera'], rl['regla_todas'])[0, 1]), 2)
            b3 = blend(rl, ['regla_todas', 'tend_cartera'])
            mixes['regla_todas+tend_cartera'] = {'ventana': stats(b3, yr), **alpha_beta(b3, rm), 'a_25': at_drawdown(b3)}
        pos_ = [nm for nm in names if (out['carteras'][nm].get('t_alfa') or 0) > 0 and (out['carteras'][nm]['alfa'] or 0) > 0]
        for label, members in (('todas_las_familias', ['regla_todas'] + names), ('reglas_juntas', rule_names), ('regla+positivas', ['regla_todas'] + pos_)):
            b = blend(rl, members)
            mixes[label] = {'miembros': members, 'ventana': stats(b, yr), **alpha_beta(b, rm), 'a_25': at_drawdown(b)}
        out['mezclas'] = mixes
    if verbose:
        print(f'  listo · {time.time() - t0:.0f}s', flush=True)
    return out


def report(ev: Dict) -> str:
    m = ev['meta']
    L = [f"OTRAS FAMILIAS DE ESTRATEGIAS · {m['universe']} acciones · {m['operables_por_dia']} operables por día (las más negociadas) · {m['from']} → {m['to']} · coste {2 * m['coste_lado']:.2f} % ida y vuelta · "
         f"miembros {m['miembros']}", f"Ventana común con la regla de rebote: desde {m['ventana']}. Decisión al cierre, ejecución en la apertura siguiente, sin apalancamiento, caja sin interés. Beta, alfa y exceso se miden contra el universo a partes iguales SIN costes.",
         f"«a {m['caida_objetivo']:.0f} %»: rentabilidad anual al escalar la cartera hasta una caída máxima de {m['caida_objetivo']:.0f} % (por encima de 1× se paga {100 * m['lev_coste']:.0f} % anual por lo prestado). "
         f"Con ~30 carteras miradas a la vez, una t de alfa por debajo de {FAMILY_T} no distingue mérito de suerte de elegir la mejor.", '']
    L.append('CARTERAS (ventana común) · beta y alfa frente al universo a partes iguales')
    L.append(f"   {'cartera':22s} {'%/año':>7s} {'caída':>7s} {'Sharpe':>6s} {'a/c':>5s} {'inv.':>5s} {'beta':>5s} {'alfa':>6s} {'t alfa':>7s} {'exceso':>7s} {'t exc.':>6s} {'rot.':>6s} {'a 25 %':>14s}")
    for nm, r in ev['carteras'].items():
        v = r['ventana']
        a25 = r['a_25']
        a25s = f"{a25['anual']:+.1f} ×{a25['L']}" if a25['anual'] is not None else 'n/d'
        L.append(f"   {nm:22s} {v['anual']:+7.1f} {v['caida_max']:+7.1f} {v['sharpe']:6.2f} {_f(v['mar'], '5.2f')} {_f(v.get('exposicion_media'), '5.0f')} {_f(r['beta'], '5.2f')} {_f(r['alfa'], '+6.1f')} {_f(r['t_alfa'], '+7.2f')} "
                 f"{_f(r['exceso'], '+7.1f')} {_f(r['t_exceso'], '+6.2f')} {_f(r.get('rotacion_anual'), '6.0f')} {a25s:>14s}" + (' ◄' if (r['t_alfa'] or 0) >= FAMILY_T else ''))
    L += ['', 'AÑO A AÑO (ventana común, %)']
    for nm, r in ev['carteras'].items():
        L.append(f"   {nm:22s} " + ' '.join(f"{str(a)[2:]}:{x:+.0f}" for a, x in r['ventana']['por_año'].items()))
    L += ['', 'DESDE EL PRINCIPIO DE SUS DATOS (antes de 2020; la regla de rebote no existe antes de 2020)']
    for nm, r in ev['carteras'].items():
        d = r.get('desde_inicio')
        if d:
            L.append(f"   {nm:22s} desde {d['desde']}: {d['anual']:+.1f} %/año · caída {d['caida_max']:+.1f} % · Sharpe {d['sharpe']}")
    if ev.get('correlacion_regla'):
        L += ['', 'CORRELACIÓN DIARIA CON LA REGLA DE REBOTE (todas las señales)', '   ' + ' · '.join(f"{k} {_f(v, '+.2f')}" for k, v in ev['correlacion_regla'].items())]
        L += ['', 'MEZCLAS A IGUAL RIESGO (cada una pesa lo inverso de su volatilidad; los pesos suman 1; ventana común)']
        base = ev['carteras']['regla_todas']
        L.append(f"   {'regla sola':36s} {base['ventana']['anual']:+6.1f} %/año · caída {base['ventana']['caida_max']:+6.1f} % · Sharpe {base['ventana']['sharpe']} · a/c {_f(base['ventana']['mar'], '.2f')} · a 25 %: {_f(base['a_25']['anual'], '+.1f')}")
        for nm, r in ev['mezclas'].items():
            v = r['ventana']
            L.append(f"   {nm:36s} {v['anual']:+6.1f} %/año · caída {v['caida_max']:+6.1f} % · Sharpe {v['sharpe']} · a/c {_f(v['mar'], '.2f')} · beta {_f(r['beta'], '.2f')} · a 25 %: {_f(r['a_25']['anual'], '+.1f')}"
                     + (f" · {', '.join(r['miembros'])}" if r.get('miembros') else ''))
    rb = ev.get('robustez_tendencia')
    if rb:
        L += ['', 'ROBUSTEZ DE LA TENDENCIA (cartera de tendencias a igual riesgo con otras medias, sin cripto y solo cripto; ventana común)']
        L.append(f"   {'variante':26s} {'%/año':>7s} {'caída':>7s} {'Sharpe':>6s} {'beta':>5s} {'alfa':>6s} {'t alfa':>7s} {'a 25 %':>14s}")
        for nm, r in rb['variantes'].items():
            v, a25 = r['ventana'], r['a_25']
            a25s = f"{a25['anual']:+.1f} ×{a25['L']}" if a25['anual'] is not None else 'n/d'
            L.append(f"   {nm:26s} {v['anual']:+7.1f} {v['caida_max']:+7.1f} {v['sharpe']:6.2f} {_f(r['beta'], '5.2f')} {_f(r['alfa'], '+6.1f')} {_f(r['t_alfa'], '+7.2f')} {a25s:>14s}")
        L += ['', 'POR SUBPERIODOS (%/año · caída · Sharpe; la cuenta empieza en 1 al inicio de cada subperiodo)']
        L.append(f"   {'cartera':26s} " + ' '.join(f"{p_:>28s}" for p_ in PERIODOS))
        for nm, d_ in rb['periodos'].items():
            L.append(f"   {nm:26s} " + ' '.join((f"{d_[p_]['anual']:+7.1f} {d_[p_]['caida_max']:+7.1f} {d_[p_]['sharpe']:6.2f}   " if p_ in d_ else f"{'n/d':>28s}") for p_ in PERIODOS))
    if ev.get('neutrales'):
        L += ['', f"ALFA PURO: cada cartera menos su beta por el universo a partes iguales (la cobertura se supone gratis), escalada hasta una caída de {m['caida_objetivo']:.0f} % con el dinero prestado a {100 * m['lev_coste']:.0f} % anual"]
        L.append(f"   {'cartera':22s} {'beta':>5s} {'alfa':>6s} {'t alfa':>7s} {'Sharpe':>6s} {'caída':>7s} {'a 25 %':>14s}")
        for nm, r in ev['neutrales'].items():
            a25 = r['a_25']
            a25s = f"{a25['anual']:+.1f} ×{a25['L']}" if a25['anual'] is not None else 'n/d'
            L.append(f"   {nm:22s} {_f(r['beta'], '5.2f')} {_f(r['alfa'], '+6.1f')} {_f(r['t_alfa'], '+7.2f')} {r['ventana']['sharpe']:6.2f} {r['ventana']['caida_max']:+7.1f} {a25s:>14s}")
    L += ['', 'QUÉ ES CADA CARTERA'] + [f"   {k:22s} {v}" for k, v in ev['desc'].items()]
    return '\n'.join(L)


def _f(x, fmt: str = '') -> str:
    return 'n/d' if x is None else format(x, fmt)


def run(symbols_: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, members: str = 'asof') -> Dict:
    from . import data, universe
    bars, failed = data.fetch_many(symbols_, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols_):
        raise SystemExit(f'solo {len(bars)} de {len(symbols_)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    join = None
    if members == 'asof':
        try:
            join = universe.sp500_join_dates()
        except Exception as exc:
            print(f'  no se pudo leer la tabla de miembros del S&P 500 ({type(exc).__name__}: {exc}): se usan los miembros actuales', flush=True)
    assets = None
    try:
        assets, _ = data.fetch_many(list(ASSETS), rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
        print(f'  activos descargados: {sorted(assets)}', flush=True)
    except Exception as exc:
        print(f'  no se pudieron descargar los activos ({type(exc).__name__}: {exc}): se omite la tendencia en activos', flush=True)
    ev = analyze(bars, join=join, workers=workers, verbose=True, assets=assets or None)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'factores.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'factores_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
