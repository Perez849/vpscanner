"""Cartera simulada y menú de salidas de la regla de impulso (parte 2): ¿da para vivir?

La confirmación mide «por operación». Quien opera tiene una cuenta, un máximo de posiciones y una regla de tamaño, y lo que le importa es la rentabilidad anual, la caída máxima y
cuántas operaciones hace. Aquí se simula día a día la CARTERA DEL PLAN DE HOY, exactamente como la calcula el cribador: MAX_POS huecos; las posiciones abiertas ocupan hueco y solo se
rellenan los huecos libres con las acciones de la regla de mayor FIABILIDAD ese día (el modelo de vp/fiab.py, entrenado solo con lo cerrado antes: walk-forward por años), solo volatilidad
alta, sin valores correlacionados con los abiertos, RISK_PCT % de la cuenta de riesgo en el stop inicial de cada una, sin apalancamiento; y se compara con:
  · COMPRAR TODAS LAS SEÑALES de la regla (sin límite de posiciones ni de correlación, mismo tamaño, con lo que haya en caja): ¿elegir 5 por fiabilidad es mejor que comprarlo todo?
  · la misma cartera de 5 eligiendo AL AZAR entre las acciones del universo (misma volatilidad alta, el mismo número de entradas cada día, la misma salida): lo que pone la regla;
  · la misma cartera con las acciones de la regla en orden al azar y en orden de volatilidad: lo que pone la fiabilidad;
  · comprar y mantener el universo a partes iguales: el mercado.
El periodo es el de las predicciones fuera de muestra del modelo (desde el cuarto año de datos); antes, el modelo no existe.
Todo con los precios reales posteriores y el mismo simulador que la web y el seguimiento (search.sim_all ≡ plan.outcome_ladder). Es DESCRIPTIVO: la regla y la escalera estaban fijadas de
antemano; el menú compara salidas sobre la misma entrada, y como se mira después de ver los datos, la mejor del menú no es «la buena», es la referencia con la que se juzga la que se usa.
Resultado: data/cartera.json y data/cartera_report.txt.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ml, search, ta
from .ml import CTX

COST = search.COST
RISK_PCT = 1.0             # riesgo por operación (% de la cuenta) en el stop inicial
MAX_POS = 5                # posiciones abiertas a la vez, como mucho
CORR_MIN = 0.6             # no se abre un valor con correlación de rendimientos (60 sesiones) ≥ 0,6 con uno ya abierto
CORR_WIN = 60
VOL_MIN = 0.6              # solo volatilidad alta (percentil de ATR/precio del día ≥ 0,6): donde se midió la ventaja
SIZE_STOP_ATR = 2.5        # el tamaño se calcula con un stop a 2,5 ATR aunque la salida no lo lleve (cierre a N sesiones): mismo riesgo nominal para todas
CAP_POS = 1.0 / MAX_POS    # una posición nunca pasa del 20 % de la cuenta
REPS = 20                  # carteras al azar por comparación
MARGIN = 62                # las últimas 62 sesiones no abren operaciones: así todas las de 60 sesiones están cerradas

# Salidas del menú: todas con stop inicial a 2,5 ATR salvo que se diga. La primera es la del operador; el resto, variantes razonables fijadas aquí, antes de mirar el resultado.
MENU = (
    {'name': 'esc', 'txt': 'tu escalera: stop 2,5 ATR · al cerrar a +4 % el stop sube a +4 % y se queda · al cerrar a +10 % sube a +10 % y persigue a 2 ATR', 'stop_atr': 2.5, 'ladder': ((4.0, None), (10.0, 2.0)), 'maxb': 60},
    {'name': 'colchon', 'txt': 'escalera con colchón: al cerrar a +6 % asegura +4 %; al cerrar a +12 % asegura +10 % y persigue a 2 ATR', 'stop_atr': 2.5, 'ladder': ((6.0, 4.0, None), (12.0, 10.0, 2.0)), 'maxb': 60},
    {'name': 'corre', 'txt': 'escalera que deja correr más: +4 % plano · +10 % plano · al cerrar a +20 % persigue a 2 ATR', 'stop_atr': 2.5, 'ladder': ((4.0, None), (10.0, None), (20.0, 2.0)), 'maxb': 60},
    {'name': 'ancha', 'txt': 'escalera con persecución ancha: +4 % plano · al cerrar a +10 % persigue a 3 ATR', 'stop_atr': 2.5, 'ladder': ((4.0, None), (10.0, 3.0)), 'maxb': 60},
    {'name': 'alta', 'txt': 'escalera alta: al cerrar a +6 % el stop sube a +6 % · al cerrar a +15 % sube a +15 % y persigue a 2 ATR', 'stop_atr': 2.5, 'ladder': ((6.0, None), (15.0, 2.0)), 'maxb': 60},
    {'name': 'op4b', 'txt': 'asegura +4 % y desde ahí persigue a 3 ATR (sin segundo nivel)', 'stop_atr': 2.5, 'lock': 4.0, 'trail': 3.0, 'maxb': 60},
    {'name': 'op6b', 'txt': 'asegura +6 % y desde ahí persigue a 4 ATR', 'stop_atr': 2.5, 'lock': 6.0, 'trail': 4.0, 'maxb': 60},
    {'name': 'trail3', 'txt': 'stop 2,5 ATR que persigue al mejor cierre a 3 ATR desde el principio, sin asegurar nada', 'stop_atr': 2.5, 'trail': 3.0, 'maxb': 60},
    {'name': 'run5', 'txt': 'stop 5 ATR que persigue a 5 ATR, sin asegurar nada', 'stop_atr': 5.0, 'trail': 5.0, 'maxb': 60},
    {'name': 'stop20', 'txt': 'stop 2,5 ATR fijo y cierre a 20 sesiones', 'stop_atr': 2.5, 'maxb': 20},
    {'name': 'h10', 'txt': 'cierre a 10 sesiones, sin stop', 'hold': 10},
    {'name': 'h20', 'txt': 'cierre a 20 sesiones, sin stop', 'hold': 20},
    {'name': 'h40', 'txt': 'cierre a 40 sesiones, sin stop', 'hold': 40},
)
MENU_TXT = {e['name']: e['txt'] for e in MENU}
MN = [e['name'] for e in MENU]


def _dur(e: Dict) -> int:
    return e['hold'] if 'hold' in e else e['maxb']


def _sym_work(args: Tuple[str, Dict[str, np.ndarray]]) -> Tuple[str, Dict]:
    """Por valor: resultado neto, código y duración de entrar en largo en la apertura siguiente a cada barra, con cada salida del menú."""
    sym, b = args
    F = ta.prepare(b)
    o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
    ex = {}
    for e in MENU:
        if 'hold' in e:
            net, code, nb = search.hold_all(o, c, 1, e['hold'], COST, with_bars=True)
        else:
            net, code, nb = search.sim_all(o, h, l, c, atr, 1, e.get('stop_atr'), e.get('lock'), e.get('trail'), e['maxb'], COST, None, e.get('ladder'), with_bars=True)
        ex[e['name']] = (net.astype(np.float32), code, nb)
    return sym, {'t': (np.asarray(F['t']) // 86400).astype(np.int64), 'o': o.astype(np.float32), 'c': c.astype(np.float32), 'ex': ex}


class World:
    """Lo que necesita la simulación diaria: cierres y rendimientos por (día, valor), filas del universo por día, condiciones, y la barra → día de cada valor."""

    def __init__(self, D: ml.Data, W: Dict[str, Dict]):
        self.D, self.W = D, W
        nd, ns = D.nd, len(D.syms)
        self.nd, self.ns = nd, ns
        self.dayv = D.date.astype('datetime64[D]').astype(np.int64)
        self.close = np.full((nd, ns), np.nan, np.float32)
        self.bd: List[np.ndarray] = []                                       # por valor: día (código) de cada barra, -1 si ese día no está en el panel
        for j, s in enumerate(D.syms):
            t = W[s]['t']
            di = np.searchsorted(self.dayv, t)
            ok = (di < nd) & (self.dayv[np.minimum(di, nd - 1)] == t)
            self.close[di[ok], j] = W[s]['c'][ok]
            self.bd.append(np.where(ok, di, -1).astype(np.int64))
        with np.errstate(invalid='ignore', divide='ignore'):
            self.rets = np.full((nd, ns), np.nan, np.float32)
            self.rets[1:] = self.close[1:] / self.close[:-1] - 1.0
        memb = np.zeros((nd, ns), bool)
        memb[D.dcode, D.sym] = True
        r = np.where(memb & np.isfinite(self.rets), self.rets, 0.0)
        cnt = (memb & np.isfinite(self.rets)).sum(1)
        self.mkt = np.where(cnt > 0, r.sum(1) / np.maximum(cnt, 1), 0.0)       # comprar y mantener el universo a partes iguales (sin costes)
        order = np.argsort(D.dcode, kind='stable')
        starts = np.searchsorted(D.dcode[order], np.arange(nd + 1))
        self.rows_by_day = [order[starts[d]:starts[d + 1]] for d in range(nd)]
        self.cond = search.conditions(D)
        self.vol = D.X_ctx[:, CTX.index('atrp')]
        self.mom = D.X_ctx[:, CTX.index('mom12_1')]
        self.atrp = D.ctx[:, CTX.index('atrp')]                               # ATR / precio en %
        self.year = D.year
        self.pred = np.full(len(D.sym), np.nan, np.float32)                   # fiabilidad fuera de muestra por fila (la pone analyze); NaN donde no hay modelo
        self.blk = D.dcode // search.BLOCK
        self.nb = int(self.blk.max()) + 1
        # neto de cada fila con cada salida (para la estadística por operación y la referencia del día)
        self.net_rows = {e: np.full(len(D.sym), np.nan, np.float32) for e in MN}
        self.code_rows = {e: np.full(len(D.sym), -1, np.int8) for e in MN}
        self.bars_rows = {e: np.zeros(len(D.sym), np.int16) for e in MN}
        for j, s in enumerate(D.syms):
            sl = slice(int(D.symoff[j]), int(D.symoff[j + 1]) if j + 1 < ns else len(D.sym))
            bars = D.bar[sl]
            for e in MN:
                net, code, nb = W[s]['ex'][e]
                self.net_rows[e][sl] = net[bars]
                self.code_rows[e][sl] = code[bars]
                self.bars_rows[e][sl] = nb[bars]

    def rule_mask(self, regla: List[str], vol_min: Optional[float] = VOL_MIN) -> np.ndarray:
        m = np.ones(len(self.D.sym), bool)
        for k in regla:
            m &= self.cond[k]
        if vol_min is not None:
            m &= self.vol >= vol_min
        return m

    def corr(self, d: int, j: int, k: int, win: int = CORR_WIN) -> float:
        a, b = self.rets[max(0, d - win + 1):d + 1, j], self.rets[max(0, d - win + 1):d + 1, k]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < win * 2 // 3:
            return 0.0
        a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else 0.0


def simulate(Wd: World, mask: np.ndarray, exit_name: str, order: str = 'vol', rng: Optional[np.random.Generator] = None, risk: float = RISK_PCT, max_pos: int = MAX_POS,
             corr_min: float = CORR_MIN, schedule: Optional[np.ndarray] = None, d0: Optional[int] = None, d1: Optional[int] = None) -> Dict:
    """Un recorrido de la cartera. `mask`: filas candidatas (regla y volatilidad). `order`: 'fiab' = por fiabilidad fuera de muestra descendente (como el plan de hoy; donde no hay modelo, por
    volatilidad), 'vol' = por volatilidad del día descendente (y luego impulso), 'azar' = al azar.
    `schedule`: entradas máximas por día (para que la cartera al azar entre los mismos días y el mismo número de veces que la de la regla). Entrada en la apertura siguiente al día de la señal,
    tamaño = riesgo / distancia al stop (acotado al 20 % de la cuenta y al efectivo disponible), valoración diaria a cierre, cierre con el resultado real de la salida."""
    D = Wd.D
    nd = Wd.nd
    d0 = 0 if d0 is None else d0
    d1 = nd - 1 - MARGIN if d1 is None else d1
    dend = min(nd - 1, d1 + MARGIN)
    rng = rng or np.random.default_rng(0)
    net_rows, bars_rows = Wd.net_rows[exit_name], Wd.bars_rows[exit_name]
    cash = 1.0
    open_: Dict[int, Dict] = {}
    eq = np.ones(dend - d0 + 1)
    expo = np.zeros(dend - d0 + 1)
    entries = np.zeros(nd, np.int32)
    trades: List[Tuple[int, int, int, float, float, int]] = []                 # (día señal, día salida, valor, neto %, fracción de la cuenta, sesiones)
    for d in range(d0, dend + 1):
        for j in [j for j, p in open_.items() if p['xd'] <= d]:
            p = open_.pop(j)
            cash += p['basis'] * (1.0 + p['net'] / 100.0)
            trades.append((p['d'], p['xd'], j, p['net'], p['frac'], p['nb']))
        val = 0.0
        for j, p in open_.items():
            px = Wd.close[d, j]
            if np.isfinite(px):
                p['last'] = float(px)
            val += p['basis'] * p['last'] / p['entry']
        equity = cash + val
        eq[d - d0] = equity
        expo[d - d0] = val / equity if equity > 0 else 0.0
        if d > d1:
            continue
        slots = max_pos - len(open_)
        n_new = slots if schedule is None else min(slots, int(schedule[d]))
        if n_new <= 0:
            continue
        rows = Wd.rows_by_day[d]
        cand = rows[mask[rows]]
        if not len(cand):
            continue
        if order == 'fiab':
            key = np.where(np.isfinite(Wd.pred[cand]), Wd.pred[cand], -1.0)
            cand = cand[np.lexsort((-Wd.mom[cand], -Wd.vol[cand], -key))]
        elif order == 'vol':
            cand = cand[np.lexsort((-Wd.mom[cand], -Wd.vol[cand]))]
        else:
            cand = rng.permutation(cand)
        taken = 0
        for r in cand:
            if taken >= n_new:
                break
            j = int(D.sym[r])
            if j in open_:
                continue
            net = float(net_rows[r])
            if not np.isfinite(net):
                continue
            b = int(D.bar[r])
            nb = int(bars_rows[r])
            bd = Wd.bd[j]
            if b + nb >= len(bd) or bd[b + nb] < 0 or b + 1 >= len(Wd.W[D.syms[j]]['o']):
                continue
            xd = int(bd[b + nb])
            if any(Wd.corr(d, j, k) >= corr_min for k in open_):
                continue
            stop_pct = SIZE_STOP_ATR * float(Wd.atrp[r])
            if not stop_pct > 0:
                continue
            frac = min(risk / stop_pct, CAP_POS)
            basis = min(frac * equity, cash)
            if basis <= 1e-9:
                continue
            cash -= basis
            entry = float(Wd.W[D.syms[j]]['o'][b + 1])
            open_[j] = {'d': d, 'xd': xd, 'net': net, 'basis': basis, 'frac': basis / equity, 'entry': entry, 'last': entry, 'nb': nb}
            taken += 1
        entries[d] = taken
    for j, p in open_.items():                                                # lo que quede abierto al final se cierra con su resultado real (ya conocido)
        cash += p['basis'] * (1.0 + p['net'] / 100.0)
        trades.append((p['d'], p['xd'], j, p['net'], p['frac'], p['nb']))
    eq[-1] = cash
    return {'eq': eq, 'expo': expo, 'entries': entries, 'trades': trades, 'd0': d0, 'dend': dend}


def metrics(eq: np.ndarray, years: np.ndarray, trades: Optional[List] = None, expo: Optional[np.ndarray] = None) -> Dict:
    """Rentabilidad anual compuesta, caída máxima, volatilidad, ratio de Sharpe (sin tipo libre de riesgo), rentabilidad por año natural y, si hay operaciones, su resumen."""
    eq = np.asarray(eq, float)
    n = len(eq)
    yrs = n / 252.0
    tot = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 and eq[-1] > 0 else -1.0
    dd = eq / np.maximum.accumulate(eq) - 1.0
    r = np.diff(np.log(np.maximum(eq, 1e-12)))
    vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 0 else 0.0
    por_año = {}
    for y in np.unique(years):
        ix = np.flatnonzero(years == y)
        first = eq[ix[0] - 1] if ix[0] > 0 else eq[ix[0]]
        por_año[int(y)] = round(float((eq[ix[-1]] / first - 1.0) * 100), 2)
    out = {'total': round(float(tot * 100), 2), 'anual': round(float(cagr * 100), 2), 'caida_max': round(float(dd.min() * 100), 2), 'volatilidad': round(vol * 100, 2), 'sharpe': round(sharpe, 2),
           'por_año': por_año, 'años_positivos': int(sum(v > 0 for v in por_año.values())), 'años': len(por_año), 'peor_año': round(min(por_año.values()), 2) if por_año else None,
           'mar': round(float(cagr / -dd.min()), 2) if dd.min() < 0 else None}
    if trades is not None:
        nets = np.array([t[3] for t in trades]) if trades else np.zeros(0)
        out.update({'operaciones': len(trades), 'por_año_ops': round(len(trades) / yrs, 1) if yrs > 0 else None, 'ganan': round(float((nets > 0).mean() * 100), 1) if len(nets) else None,
                    'neto_medio': round(float(nets.mean()), 3) if len(nets) else None, 'sesiones_media': round(float(np.mean([t[5] for t in trades])), 1) if trades else None,
                    'exposicion_media': round(float(np.mean(expo) * 100), 1) if expo is not None and len(expo) else None,
                    'tamaño_medio': round(float(np.mean([t[4] for t in trades]) * 100), 1) if trades else None})
    return out


def _summary(vals: np.ndarray, obs: float, higher_is_better: bool = True) -> Dict:
    vals = np.asarray(vals, float)
    p = float((vals >= obs).mean()) if higher_is_better else float((vals <= obs).mean())
    return {'mediana': round(float(np.median(vals)), 2), 'p05': round(float(np.quantile(vals, 0.05)), 2), 'p95': round(float(np.quantile(vals, 0.95)), 2), 'p': round(p, 3), 'n': int(len(vals))}


def op_stats(Wd: World, mask: np.ndarray, exit_name: str) -> Dict:
    """Por operación, en las filas de la regla: neto, referencia del mismo día (todas las filas del universo con esa salida), exceso con t agrupada, ganan (y azar), niveles, duración."""
    D = Wd.D
    net = Wd.net_rows[exit_name]
    ok = np.isfinite(net)
    cd = np.bincount(D.dcode[ok], minlength=Wd.nd)
    sd = np.bincount(D.dcode[ok], weights=net[ok], minlength=Wd.nd)
    sw = np.bincount(D.dcode[ok], weights=(net[ok] > 0), minlength=Wd.nd)
    mean_d = np.where(cd >= search.MIN_DAY, sd / np.maximum(cd, 1), np.nan)
    win_d = np.where(cd >= search.MIN_DAY, sw / np.maximum(cd, 1), np.nan)
    ix = np.flatnonzero(mask & ok)
    rel = net[ix] - mean_d[D.dcode[ix]]
    good = np.isfinite(rel)
    ix, rel = ix[good], rel[good]
    if len(ix) < 30:
        return {'n': int(len(ix))}
    dur = _dur(next(e for e in MENU if e['name'] == exit_name))
    g = search.block_groups(Wd.nb, max(1, int(np.ceil(dur / search.BLOCK))))
    S = np.bincount(Wd.blk[ix], weights=rel, minlength=Wd.nb)
    N = np.bincount(Wd.blk[ix], minlength=Wd.nb).astype(float)
    _, mm, tt = search.t_blocks(S, N, g)
    code = Wd.code_rows[exit_name][ix]
    nets = net[ix]
    wins, loss = nets[nets > 0], nets[nets <= 0]
    bars = Wd.bars_rows[exit_name][ix].astype(float)
    return {'n': int(len(ix)), 'neto': round(float(nets.mean()), 3), 'azar_dia': round(float(mean_d[D.dcode[ix]].mean()), 3), 'exceso': round(float(mm), 3), 't': round(float(tt), 2),
            'ganan': round(float((nets > 0).mean() * 100), 1), 'ganan_azar': round(float(win_d[D.dcode[ix]].mean() * 100), 1),
            'asegura1': round(float(((code >= 1) & (code <= 2)).mean() * 100), 1), 'asegura2': round(float((code == 2).mean() * 100), 1), 'stop_inicial': round(float((code == 0).mean() * 100), 1),
            'gan_media': round(float(wins.mean()), 2) if len(wins) else None, 'perd_media': round(float(loss.mean()), 2) if len(loss) else None,
            'factor': round(float(wins.sum() / max(-loss.sum(), 1e-9)), 2) if len(loss) else None, 'sesiones': round(float(bars.mean()), 1),
            'neto_por_sesion': round(float(nets.mean() / max(bars.mean(), 1.0)), 4)}


def analyze(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, reps: int = REPS, verbose: bool = False, join: Optional[Dict[str, str]] = None, rules=search.ENTRADAS,
            menu: Tuple[str, ...] = tuple(MN), base_rule: str = 'imp_bajo200', base_exit: str = 'esc', seed: int = 7, test_years: Optional[List[int]] = None, model: bool = True) -> Dict:
    t0 = time.time()
    D, _, dropped = search.build(bars, workers, k, verbose, search.MIN_DAY, search.MIN_COMBO, join, COST, library=False)
    jobs = [(s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if s in D.syms]
    W: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s, w in ex.map(_sym_work, jobs, chunksize=8):
                W[s] = w
    else:
        for jb in jobs:
            s, w = _sym_work(jb)
            W[s] = w
    Wd = World(D, W)
    if verbose:
        print(f'  mundo: {Wd.nd} días · {Wd.ns} valores · {len(D.sym):,} filas · {time.time() - t0:.0f}s', flush=True)
    years_d = np.zeros(Wd.nd, int)
    years_d[D.dcode] = D.year
    dates = [str(x) for x in D.date]
    rng = np.random.default_rng(seed)
    uni_mask = Wd.vol >= VOL_MIN                                                # control: cualquier acción del universo con volatilidad alta
    # la fiabilidad del plan de hoy, fuera de muestra: el mismo modelo de vp/fiab.py, entrenado cada año solo con lo cerrado 95 días antes
    years = sorted(int(v) for v in np.unique(D.year))
    if test_years is None:
        test_years = [v for v in years if v >= years[0] + 3 and int((D.year == v).sum()) > 20000]
    fiab_meta: Dict = {'test_years': test_years}
    if model and test_years:
        try:
            from . import fiab
            netb = Wd.net_rows[base_exit]
            okb = np.isfinite(netb)
            Wd.pred = fiab.walk_forward(D, fiab.features(D), (netb > 0).astype(np.int8), okb, test_years, verbose=verbose)
            fiab_meta['filas_con_modelo'] = int(np.isfinite(Wd.pred).sum())
        except Exception as exc:                                             # sin scikit-learn: el orden es por volatilidad y se dice
            fiab_meta['error'] = f'{type(exc).__name__}: {exc}'
    has_model = bool(np.isfinite(Wd.pred).any())
    d0 = int(np.flatnonzero(years_d == test_years[0])[0]) if has_model else 0   # ventana de la cartera: desde que hay modelo (antes no existe el plan de hoy)
    fiab_meta['orden'] = 'fiabilidad fuera de muestra' if has_model else 'volatilidad (sin modelo)'
    out: Dict = {'meta': {'generated': int(time.time()), 'universe': Wd.ns, 'filas': int(len(D.sym)), 'from': dates[0], 'to': dates[-1], 'k': k, 'coste': COST, 'riesgo_pct': RISK_PCT, 'max_pos': MAX_POS,
                          'corr_min': CORR_MIN, 'vol_min': VOL_MIN, 'cap_pos': CAP_POS, 'reps': reps, 'menu': {e: MENU_TXT[e] for e in menu}, 'barras_quitadas': dropped, 'miembros': 'en la fecha' if join else 'actuales',
                          'reglas': {r['nombre']: {'etiqueta': r['etiqueta'], 'regla': r['regla']} for r in rules}, 'base': {'regla': base_rule, 'salida': base_exit},
                          'ultimo_dia_con_entradas': dates[Wd.nd - 1 - MARGIN], 'ventana_desde': dates[d0], 'fiab': fiab_meta}, 'menu': {}, 'base': {}}
    for r in rules:
        mask = Wd.rule_mask(r['regla'])
        out['menu'][r['nombre']] = {}
        for e in menu:
            st = op_stats(Wd, mask, e)
            sim = simulate(Wd, mask, e, 'fiab', rng, d0=d0)
            yrs = years_d[sim['d0']:sim['dend'] + 1]
            row = {'operacion': st, 'cartera': metrics(sim['eq'], yrs, sim['trades'], sim['expo'])}
            if r['nombre'] == base_rule or e == base_exit:                      # controles: cartera al azar del universo con el mismo calendario de entradas
                ctrl = [simulate(Wd, uni_mask, e, 'azar', rng, schedule=sim['entries'], d0=d0) for _ in range(reps)]
                cm = [metrics(c['eq'], yrs) for c in ctrl]
                row['azar_universo'] = {'anual': _summary([m['anual'] for m in cm], row['cartera']['anual']), 'caida_max': _summary([m['caida_max'] for m in cm], row['cartera']['caida_max']),
                                        'operaciones_media': round(float(np.mean([len(c['trades']) for c in ctrl])), 1)}
                todas = simulate(Wd, mask, e, 'fiab', rng, max_pos=10 ** 6, corr_min=2.0, d0=d0)   # comprar todas las señales de la regla (con lo que haya en caja)
                row['todas'] = metrics(todas['eq'], yrs, todas['trades'], todas['expo'])
                if r['nombre'] == base_rule and e == base_exit:
                    ord_ = [simulate(Wd, mask, e, 'azar', rng, d0=d0) for _ in range(reps)]
                    om = [metrics(c['eq'], yrs) for c in ord_]
                    vol_ = simulate(Wd, mask, e, 'vol', rng, d0=d0)
                    mk = np.cumprod(1.0 + np.concatenate([[0.0], Wd.mkt[sim['d0'] + 1:sim['dend'] + 1]]))
                    mkm = metrics(mk, yrs)
                    med = np.median(np.vstack([c['eq'] for c in ctrl]), axis=0)
                    ds = dates[sim['d0']:sim['dend'] + 1]
                    months = [i for i in range(len(ds)) if i == 0 or ds[i][:7] != ds[i - 1][:7]] + [len(ds) - 1]
                    ops_y = lambda tr_, y: [t for t in tr_ if years_d[t[0]] == y]
                    out['base'] = {'regla': r['nombre'], 'salida': e, 'desde': ds[0], 'hasta': ds[-1], 'años': round(len(ds) / 252.0, 1),
                                   'lineas': {'plan': row['cartera'], 'todas': row['todas'], 'plan_vol': metrics(vol_['eq'], yrs, vol_['trades'], vol_['expo']), 'mercado': mkm},
                                   'azar_universo': row['azar_universo'], 'azar_orden': {'anual': _summary([m['anual'] for m in om], row['cartera']['anual']), 'caida_max': _summary([m['caida_max'] for m in om], row['cartera']['caida_max'])},
                                   'curva': [[ds[i], round(float(sim['eq'][i]), 4), round(float(todas['eq'][i]), 4), round(float(mk[i]), 4), round(float(med[i]), 4)] for i in months],
                                   'por_año': [{'año': int(y), 'plan': row['cartera']['por_año'][int(y)], 'todas': row['todas']['por_año'][int(y)], 'mercado': mkm['por_año'][int(y)],
                                                'azar_mediana': round(float(np.median([m['por_año'][int(y)] for m in cm])), 2), 'operaciones': len(ops_y(sim['trades'], y)),
                                                'ganan': round(float(np.mean([t[3] > 0 for t in ops_y(sim['trades'], y)]) * 100), 1) if ops_y(sim['trades'], y) else None,
                                                'operaciones_todas': len(ops_y(todas['trades'], y))} for y in np.unique(yrs)],
                                   'variantes': {}}
                    for nm, kw in (('sin_volatilidad', {'mask': Wd.rule_mask(r['regla'], None)}), ('riesgo_2', {'risk': 2.0}), ('riesgo_0_5', {'risk': 0.5}), ('max_3', {'max_pos': 3}), ('max_10', {'max_pos': 10}),
                                   ('sin_grupos', {'corr_min': 2.0})):
                        s2 = simulate(Wd, kw.pop('mask', mask), e, 'fiab', rng, d0=d0, **kw)
                        out['base']['variantes'][nm] = metrics(s2['eq'], yrs, s2['trades'], s2['expo'])
                    if d0 > 0:                                                  # toda la historia, sin modelo (orden por volatilidad): contexto
                        full = {nm: simulate(Wd, mask, e, 'vol', rng, **kw) for nm, kw in (('plan_vol', {}), ('todas', {'max_pos': 10 ** 6, 'corr_min': 2.0}))}
                        yf = years_d[full['plan_vol']['d0']:full['plan_vol']['dend'] + 1]
                        mkf = np.cumprod(1.0 + np.concatenate([[0.0], Wd.mkt[full['plan_vol']['d0'] + 1:full['plan_vol']['dend'] + 1]]))
                        out['base']['desde_inicio'] = {'desde': dates[0], **{nm: metrics(sm['eq'], yf, sm['trades'], sm['expo']) for nm, sm in full.items()}, 'mercado': metrics(mkf, yf)}
            out['menu'][r['nombre']][e] = row
            if verbose:
                print(f"  {r['nombre']:12s} {e:8s} ops {st.get('n', 0):6d} exceso {st.get('exceso', float('nan')):+.3f} (t {st.get('t', float('nan')):+.1f}) · plan {row['cartera']['anual']:+.1f} %/año, caída {row['cartera']['caida_max']:+.1f} %"
                      + (f" · todas {row['todas']['anual']:+.1f} % · azar {row['azar_universo']['anual']['mediana']:+.1f} % (p {row['azar_universo']['anual']['p']})" if 'azar_universo' in row else '') + f' · {time.time() - t0:.0f}s', flush=True)
    return out


def report(ev: Dict) -> str:
    m = ev['meta']
    L = [f"CARTERA SIMULADA Y MENÚ DE SALIDAS · {m['universe']} acciones · {m['filas']:,} filas (las {m['k']} más negociadas) · {m['from']} → {m['to']} · coste {m['coste']} %",
         f"Disciplina del plan de hoy: máximo {m['max_pos']} posiciones, una por grupo (correlación < {m['corr_min']}), solo volatilidad alta (percentil ≥ {m['vol_min']}), riesgo {m['riesgo_pct']} % de la cuenta en cada stop inicial (a {SIZE_STOP_ATR} ATR), "
         f"posición ≤ {m['cap_pos'] * 100:.0f} % de la cuenta, sin apalancamiento. Entrada en la apertura siguiente; valoración diaria a cierre. Las posiciones abiertas ocupan hueco; los huecos libres se rellenan por fiabilidad fuera de muestra. Control: {m['reps']} carteras eligiendo al azar entre las acciones del universo con volatilidad alta, el mismo número de entradas cada día. Ventana con modelo desde {m.get('ventana_desde')}.",
         f"Universo: {'miembros actuales desde su incorporación al S&P 500' if m['miembros'] == 'en la fecha' else 'miembros actuales'}. Últimas entradas el {m['ultimo_dia_con_entradas']}.", '']
    b = ev.get('base') or {}
    if b:
        ln, au, ao = b['lineas'], b['azar_universo'], b['azar_orden']
        c, td, pv, mk = ln['plan'], ln['todas'], ln['plan_vol'], ln['mercado']
        L += [f"CASO BASE · {m['reglas'][b['regla']]['etiqueta']} · {MENU_TXT[b['salida']]} · {b['desde']} → {b['hasta']} ({b['años']} años; orden del plan: {m['fiab'].get('orden')})",
              f"   cartera del plan (5 huecos, por fiabilidad): {c['anual']:+.1f} %/año · caída máxima {c['caida_max']:+.1f} % · Sharpe {c['sharpe']} · {c['operaciones']} operaciones ({c['por_año_ops']}/año) · ganan {c['ganan']} % · exposición media {c['exposicion_media']} % · años positivos {c['años_positivos']}/{c['años']} (peor {c['peor_año']:+.1f} %)",
              f"   todas las señales de la regla (sin límite): {td['anual']:+.1f} %/año · caída {td['caida_max']:+.1f} % · Sharpe {td['sharpe']} · {td['operaciones']} operaciones · ganan {td['ganan']} % · exposición media {td['exposicion_media']} %",
              f"   misma cartera de 5 en orden de volatilidad: {pv['anual']:+.1f} %/año, caída {pv['caida_max']:+.1f} % · en orden al azar: {ao['anual']['mediana']:+.1f} %/año (p {ao['anual']['p']})",
              f"   azar del universo (mismas fechas y número de entradas): {au['anual']['mediana']:+.1f} %/año (p05 {au['anual']['p05']:+.1f}, p95 {au['anual']['p95']:+.1f}) · caída mediana {au['caida_max']['mediana']:+.1f} % · p(azar ≥ plan) = {au['anual']['p']}",
              f"   mercado a partes iguales: {mk['anual']:+.1f} %/año, caída {mk['caida_max']:+.1f} %",
              '   por año: ' + ' '.join(f"{str(r['año'])[2:]}:{r['plan']:+.1f}/{r['todas']:+.1f}/{r['mercado']:+.1f}/{r['azar_mediana']:+.1f}" for r in b['por_año']) + '  (plan/todas/mercado/azar)',
              '   variantes del plan: ' + ' · '.join(f"{k}: {v['anual']:+.1f} %/año, caída {v['caida_max']:+.1f} %, {v['operaciones']} ops" for k, v in b['variantes'].items())]
        if b.get('desde_inicio'):
            di = b['desde_inicio']
            L.append(f"   desde {di['desde']} (sin modelo, orden por volatilidad): plan {di['plan_vol']['anual']:+.1f} %/año (caída {di['plan_vol']['caida_max']:+.1f} %) · todas {di['todas']['anual']:+.1f} % (caída {di['todas']['caida_max']:+.1f} %) · mercado {di['mercado']['anual']:+.1f} % (caída {di['mercado']['caida_max']:+.1f} %)")
        L.append('')
    for rn, exits in ev['menu'].items():
        L.append(f"MENÚ DE SALIDAS · {m['reglas'][rn]['etiqueta']} (volatilidad alta)")
        L.append('   salida'.ljust(12) + 'n'.rjust(7) + 'neto'.rjust(8) + 'azar'.rjust(8) + 'exceso'.rjust(8) + 't'.rjust(6) + 'ganan/azar'.rjust(12) + 'aseg1/2/stop'.rjust(14) + 'ses'.rjust(6) + '%/ses'.rjust(8) + ' | plan %/año  caída  ops  todas  azar(p)')
        for e, row in exits.items():
            s, c = row['operacion'], row['cartera']
            if 'exceso' not in s:
                continue
            az = row.get('azar_universo')
            L.append(('   ' + e).ljust(12) + f"{s['n']:7d}{s['neto']:+7.2f}%{s['azar_dia']:+7.2f}%{s['exceso']:+7.2f}%{s['t']:+6.1f}" + f"{s['ganan']:5.0f}/{s['ganan_azar']:3.0f} %".rjust(12)
                     + f"{s['asegura1']:3.0f}/{s['asegura2']:3.0f}/{s['stop_inicial']:3.0f}".rjust(14) + f"{s['sesiones']:6.1f}{s['neto_por_sesion']:+8.3f}" + f" | {c['anual']:+7.1f}  {c['caida_max']:+6.1f}  {c['operaciones']:4d}"
                     + (f"  {row['todas']['anual']:+5.1f}  {az['anual']['mediana']:+.1f} ({az['anual']['p']})" if az else ''))
        L.append('')
    return '\n'.join(L)


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, reps: int = REPS, members: str = 'asof') -> Dict:
    from . import data, universe
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    join = None
    if members == 'asof':
        try:
            join = universe.sp500_join_dates()
        except Exception as exc:
            print(f'  no se pudo leer la tabla de miembros del S&P 500 ({type(exc).__name__}: {exc}): se usan los miembros actuales', flush=True)
    ev = analyze(bars, workers=workers, reps=reps, verbose=True, join=join)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'cartera.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'cartera_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
