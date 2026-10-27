"""
Búsqueda de la regla que más gana por encima del azar, sin engañarse.

Pregunta: ¿existe alguna combinación de condiciones (patrones y contexto) y de salida del operador (stop inicial, asegurar un porcentaje y dejar correr) cuya ganancia
supere de forma estable a entrar al azar el mismo día?

  · Panel: cada fila es una acción y una fecha (las K más negociadas ese día). Para cada fila y para largo y corto se simula la entrada en la apertura siguiente con
    6 salidas fijadas de antemano: 4 del operador (stop inicial de 1,5 o 2,5 ATR; al cerrar a +4 % o +6 % el stop sube a ese nivel y persigue al mejor cierre a 3 o 4 ATR),
    un stop que persigue a 5 ATR sin asegurar y el cierre a 10 sesiones. Resultado neto de costes. «Exceso» = neto − media de todas las filas de ese día (mismo sentido y salida).
  · Biblioteca de reglas fijada de antemano: 32 condiciones binarias (24 de contexto: tendencia, impulso, sobreventa, extremos anuales, volatilidad, volumen, huecos, mercado;
    8 de patrón: velas, figuras y rupturas alcistas o bajistas recientes y su confluencia), solas o de dos en dos, en largo o en corto. Todas las combinaciones con filas suficientes.
  · Selección hacia delante: para cada año de prueba Y se elige, con los datos anteriores a Y (menos 60 sesiones para que no se cuele ningún resultado aún sin cerrar), la combinación
    de regla y salida con mayor t de exceso, y se evalúa en Y sin tocarla. El resultado «fuera de muestra» es el del procedimiento completo de buscar y elegir, no el de una regla escogida
    mirando todos los años.
  · t con errores estándar agrupados en bloques de 10 sesiones (los resultados de días vecinos se solapan y las señales vienen en rachas).
  · Nulo del procedimiento: se repite TODO (construcción de las tablas, selección año a año, evaluación) desplazando los resultados de cada valor en el tiempo (el mismo desplazamiento
    para todos, como un calendario distinto): las condiciones conservan su estructura y ya no pueden predecir nada. Lo que el procedimiento «encuentre» así es lo que da el azar al buscar.
  · Se estudian por separado las reglas con patrón y las de solo contexto: si el patrón añade algo, las primeras han de ganar a las segundas.
"""
from __future__ import annotations
import json
import os
import sys
import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ml, ta
from .ml import CTX, COST

BLOCK = 10
LONG_BLOCKS = 6            # las salidas que duran hasta 60 sesiones agrupan sus errores en bloques de 6 × 10 = 60 sesiones (sus resultados se solapan hasta 60); el cierre a 10 usa bloques de 10
PURGE = 60                 # sesiones: lo máximo que dura una operación; los años de entrenamiento terminan 60 sesiones antes del año de prueba
MIN_COMBO = 5000           # filas mínimas para que una combinación entre en la biblioteca
MIN_TRAIN_N = 2000         # operaciones mínimas en el entrenamiento para poder ser elegida
MIN_COVER = 0.6            # fracción mínima de los periodos (grupos de error) del entrenamiento en los que la regla tiene operaciones: una regla de un solo régimen (p. ej. mercado bajista) no es elegible
MIN_GROUPS = 8             # periodos mínimos del entrenamiento con operaciones: con menos, el t agrupado no es fiable
MIN_DAY = 30               # valores mínimos del universo ese día para que exista referencia del día
DEV_END = 2022
MIN_SHIFT = 300            # desplazamiento mínimo del calendario en los nulos (sesiones): las variables de 12 meses tienen memoria de ~250 sesiones y un desplazamiento menor no rompe su relación con el resultado
# Salidas fijadas de antemano. stop_atr = stop inicial en ATR (o stop_pct, en % del precio); lock = al cerrar a ese % a favor el stop sube a ese nivel y persigue a `trail` ATR;
# ladder = la escalera del operador: (nivel %, ATR a los que persigue desde él o None = stop plano en el nivel); run5 = persigue desde el principio; h10 = cierre a 10 sesiones sin stop.
LADDER = ((4.0, None), (10.0, 2.0))          # +4 % → stop a +4 % plano (se deja correr); +10 % → stop a +10 % que persigue a 2 ATR
EXITS = (
    {'name': 'op4a', 'stop_atr': 1.5, 'lock': 4.0, 'trail': 3.0, 'maxb': 60},
    {'name': 'op4b', 'stop_atr': 2.5, 'lock': 4.0, 'trail': 3.0, 'maxb': 60},
    {'name': 'op6a', 'stop_atr': 1.5, 'lock': 6.0, 'trail': 4.0, 'maxb': 60},
    {'name': 'op6b', 'stop_atr': 2.5, 'lock': 6.0, 'trail': 4.0, 'maxb': 60},
    {'name': 'run5', 'stop_atr': 5.0, 'lock': None, 'trail': 5.0, 'maxb': 60},
    {'name': 'h10', 'stop_atr': None, 'lock': None, 'trail': None, 'maxb': 10},
    {'name': 'esc', 'stop_atr': 2.5, 'ladder': LADDER, 'maxb': 60},
    {'name': 'esca', 'stop_atr': 1.5, 'ladder': LADDER, 'maxb': 60},
    {'name': 'esc6', 'stop_pct': 6.0, 'ladder': LADDER, 'maxb': 60},
    {'name': 'esc10', 'stop_pct': 10.0, 'ladder': LADDER, 'maxb': 60},
)
EXIT_TXT = {'op4a': 'stop 1,5 ATR · asegura +4 % · persigue 3 ATR', 'op4b': 'stop 2,5 ATR · asegura +4 % · persigue 3 ATR', 'op6a': 'stop 1,5 ATR · asegura +6 % · persigue 4 ATR',
            'op6b': 'stop 2,5 ATR · asegura +6 % · persigue 4 ATR', 'run5': 'stop que persigue 5 ATR sin asegurar', 'h10': 'cierre a 10 sesiones sin stop',
            'esc': 'escalera: stop 2,5 ATR · asegura +4 % y deja correr · asegura +10 % y persigue 2 ATR', 'esca': 'escalera con stop inicial 1,5 ATR',
            'esc6': 'escalera con stop inicial al 6 % del precio', 'esc10': 'escalera con stop inicial al 10 % del precio'}
EXN = [e['name'] for e in EXITS]
CODE_TXT = {0: 'stop inicial', 1: 'aseguró +4 %', 2: 'aseguró +10 %', 3: 'cierre por tiempo sin asegurar'}
NE = len(EXITS)
DIRS = ('largo', 'corto')

DESC = {
    'sobre_200': 'precio sobre su media de 200', 'bajo_200': 'precio bajo su media de 200',
    'impulso_alto': 'impulso a 120 sesiones en el 20 % mejor del día', 'impulso_bajo': 'impulso a 120 sesiones en el 20 % peor',
    'impulso12_alto': 'impulso 12-1 meses en el 20 % mejor', 'impulso12_bajo': 'impulso 12-1 meses en el 20 % peor',
    'cae5': 'caída a 5 sesiones entre el 10 % mayor del día', 'sube5': 'subida a 5 sesiones entre el 10 % mayor del día',
    'rsi2_bajo': 'RSI(2) ≤ 10', 'rsi2_alto': 'RSI(2) ≥ 90', 'rsi14_bajo': 'RSI(14) en el 30 % más bajo', 'rsi14_alto': 'RSI(14) en el 30 % más alto',
    'max52': 'a ≤ 10 % del máximo de 52 semanas (posición ≥ 0,9)', 'min52': 'cerca del mínimo de 52 semanas (posición ≤ 0,1)',
    'vol_baja': 'volatilidad (ATR/precio) en el 40 % más bajo', 'vol_alta': 'volatilidad en el 40 % más alto', 'volumen': 'volumen ≥ 1,5 veces su media de 20',
    'hueco_abajo': 'hueco de apertura ≤ −2 %', 'hueco_arriba': 'hueco de apertura ≥ +2 %', 'mercado_alza': 'más del 55 % de las acciones sobre su media de 200',
    'mercado_baja': 'menos del 45 % de las acciones sobre su media de 200', 'bajo_media20': 'bajo su media de 20 (20 % más alejado)', 'sobre_media20': 'sobre su media de 20 (20 % más alejado)',
    'liquido': 'entre el 30 % más líquido del día',
    'velas_alcistas': 'formación de vela alcista hoy', 'velas_bajistas': 'formación de vela bajista hoy', 'figuras_alcistas': 'figura alcista en las últimas 5 sesiones',
    'figuras_bajistas': 'figura bajista en las últimas 5 sesiones', 'rupturas_alcistas': 'ruptura alcista en las últimas 5 sesiones', 'rupturas_bajistas': 'ruptura bajista en las últimas 5 sesiones',
    'confluencia_alcista': '2 o más patrones alcistas a la vez', 'confluencia_bajista': '2 o más patrones bajistas a la vez',
}
PATTERN_CONDS = ('velas_alcistas', 'velas_bajistas', 'figuras_alcistas', 'figuras_bajistas', 'rupturas_alcistas', 'rupturas_bajistas', 'confluencia_alcista', 'confluencia_bajista')
SUBSETS = ('todo', 'contexto', 'patrones')


# ── salidas ───────────────────────────────────────────────────────────────
def _levels(lvls) -> List[Tuple[float, float, Optional[float]]]:
    """Niveles de la escalera como (disparo %, nivel que se asegura %, ATR a los que persigue o None): (4, None) = al cerrar a +4 % el stop sube a +4 %; (6, 4, None) = al cerrar a +6 % sube a +4 % (colchón)."""
    return [(float(p[0]), float(p[1]), p[2]) if len(p) == 3 else (float(p[0]), float(p[0]), p[1]) for p in lvls]


def sim_all(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray, d: int, stop_atr: Optional[float] = None, lock_pct: Optional[float] = None, trail: Optional[float] = None,
            maxb: int = 60, cost: float = COST, stop_pct: Optional[float] = None, ladder=None, with_bars: bool = False):
    """Resultado neto (%) y código de salida de entrar en la apertura siguiente a CADA barra. Stop inicial a `stop_atr` ATR (o a `stop_pct` % del precio). Con `lock_pct`: al cerrar a ese %
    a favor el stop sube a ese nivel y persigue al mejor cierre a `trail` ATR; sin lock, persigue desde el principio (y sin `trail`, el stop inicial se queda fijo hasta el cierre por tiempo).
    Con `ladder` (la escalera del operador): cada nivel es (%, ATR a los que persigue desde él o None = stop plano), o (disparo %, nivel %, ATR) cuando el nivel que se asegura es
    distinto del cierre que lo dispara; con ((4, None), (10, 2)) el stop sube a +4 % y se queda, y al cerrar a +10 % sube a +10 % y persigue a 2 ATR. El stop gana si se toca el mismo día;
    un hueco más allá sale a la apertura. Es la misma regla que plan.outcome_lock / plan.outcome_ladder, vectorizada por entradas.
    Código: 0 = saltó el stop inicial, 1 = aseguró el primer nivel, 2 = aseguró el segundo, 3 = cierre por tiempo sin asegurar. NaN / -1 donde la operación no ha terminado o no hay ATR.
    Con `with_bars` devuelve además las sesiones que duró cada operación (0 donde no terminó)."""
    n = len(c)
    m = n - 1
    net = np.full(n, np.nan)
    code = np.full(n, -1, dtype=np.int8)
    nb = np.zeros(n, dtype=np.int16)
    if m < maxb + 2:
        return (net, code, nb) if with_bars else (net, code)
    entry, a = o[1:], atr[:m]
    done = ~(np.isfinite(a) & np.isfinite(entry) & (a > 0))
    st = entry - d * (stop_atr * a if stop_pct is None else stop_pct / 100.0 * entry)
    lvls = _levels(ladder if ladder is not None else (((lock_pct, trail),) if lock_pct is not None else ()))
    tg_px = [entry * (1.0 + d * tg / 100.0) for tg, _, _ in lvls]
    lk_px = [entry * (1.0 + d * lk / 100.0) for _, lk, _ in lvls]
    level = np.zeros(m, np.int8)
    best = entry.copy()
    xp = np.full(m, np.nan)
    xb = np.zeros(m, np.int16)
    idx = np.arange(m)
    for k in range(maxb):
        j = idx + 1 + k
        live = ~done & (j < n)
        if not live.any():
            break
        jj = np.minimum(j, n - 1)
        oj, hj, lj, cj = o[jj], h[jj], l[jj], c[jj]
        if d > 0:
            hit, gap = lj <= st, oj <= st
        else:
            hit, gap = hj >= st, oj >= st
        sold = live & hit
        xp[sold] = np.where(gap, oj, st)[sold]
        xb[sold] = k + 1
        done |= sold
        upd = live & ~hit
        best = np.where(upd, np.maximum(best, cj) if d > 0 else np.minimum(best, cj), best)
        if lvls:
            for q in range(len(lvls) - 1, -1, -1):                              # el nivel más alto que el cierre haya pasado (puede saltar dos de golpe)
                reached = (cj >= tg_px[q]) if d > 0 else (cj <= tg_px[q])
                level = np.where(upd & reached & (level < q + 1), q + 1, level)
            for q, (_, _, tr) in enumerate(lvls):
                at = upd & (level == q + 1)
                if tr is None:
                    st = np.where(at, lk_px[q], st)
                else:
                    new = np.maximum(lk_px[q], best - tr * a) if d > 0 else np.minimum(lk_px[q], best + tr * a)
                    st = np.where(at, new, st)
        elif trail is not None:
            new = np.maximum(st, best - trail * a) if d > 0 else np.minimum(st, best + trail * a)
            st = np.where(upd, new, st)
    j = idx + maxb                                                         # última sesión de la ventana: cierre por tiempo
    tm = ~done & (j <= n - 1)
    xp[tm] = c[np.minimum(j, n - 1)][tm]
    xb[tm] = maxb
    fin = np.isfinite(xp)
    net[:m] = d * (xp / entry - 1.0) * 100.0 - cost
    cd = np.where(level >= 1, level, np.where(tm, 3, 0)).astype(np.int8)
    code[:m] = np.where(fin, cd, -1)
    nb[:m] = np.where(fin, xb, 0)
    return (net, code, nb) if with_bars else (net, code)


def hold_all(o: np.ndarray, c: np.ndarray, d: int, hz: int, cost: float = COST, with_bars: bool = False):
    n = len(c)
    net = np.full(n, np.nan)
    code = np.full(n, -1, dtype=np.int8)
    nb = np.zeros(n, dtype=np.int16)
    if n > hz + 1:
        mm = n - hz
        net[:mm] = d * (c[hz:hz + mm] / o[1:1 + mm] - 1.0) * 100.0 - cost
        code[:mm] = 3
        nb[:mm] = hz
    return (net, code, nb) if with_bars else (net, code)


def ops_all(F: Dict[str, np.ndarray], cost: float = COST) -> Tuple[np.ndarray, np.ndarray]:
    """(barras, 2, salidas): resultado neto de entrar a cada barra en largo (0) y en corto (1) con cada salida de EXITS, y el código de salida (int8)."""
    o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
    out = np.full((len(c), 2, NE), np.nan, dtype=np.float32)
    code = np.full((len(c), 2, NE), -1, dtype=np.int8)
    for di, d in enumerate((1, -1)):
        for ei, e in enumerate(EXITS):
            if e.get('stop_atr') is None and e.get('stop_pct') is None:
                out[:, di, ei], code[:, di, ei] = hold_all(o, c, d, e['maxb'], cost)
            else:
                out[:, di, ei], code[:, di, ei] = sim_all(o, h, l, c, atr, d, e.get('stop_atr'), e.get('lock'), e.get('trail'), e['maxb'], cost, e.get('stop_pct'), e.get('ladder'))
    return out, code


def apply_join(work: Dict[str, Dict], join: Dict[str, str]) -> int:
    """Sesgo de supervivencia: un valor solo entra en el universo desde su fecha de incorporación al índice (el dinero negociado de las barras anteriores pasa a NaN y no cuenta para el
    ranking ni para la referencia del día). Devuelve las barras-valor quitadas."""
    dropped = 0
    for s_, w in work.items():
        j = join.get(s_)
        if j:
            mask = w['t'] < int(np.datetime64(j, 'D').astype(np.int64))
            dropped += int((mask & np.isfinite(w['dv'])).sum())
            w['dv'][mask] = np.nan
    return dropped


def _work(args: Tuple) -> Tuple[str, Dict]:
    sym, b, *rest = args                                                     # (valor, barras[, coste de ida y vuelta en %])
    _, w = ml._work((sym, b))
    w['ops'], w['code'] = ops_all(ta.prepare(b), rest[0] if rest else COST)
    return sym, w


# ── panel y reglas ────────────────────────────────────────────────────────
def conditions(D: ml.Data) -> Dict[str, np.ndarray]:
    """Condiciones binarias por fila, definidas de antemano con variables causales (datos hasta el cierre de la fecha). Los percentiles son del día."""
    pct = lambda name: D.X_ctx[:, CTX.index(name)]
    raw = lambda name: D.ctx[:, CTX.index(name)]
    pn = {k: j for j, k in enumerate(D.pnames)}
    pat = lambda name: D.pat[:, pn[name]].astype(np.int16)
    bre = D.breadth_d[D.dcode]
    C: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore'):
        C['sobre_200'] = raw('dist200') > 0
        C['bajo_200'] = raw('dist200') < 0
        C['impulso_alto'] = pct('r120') >= 0.8
        C['impulso_bajo'] = pct('r120') <= 0.2
        C['impulso12_alto'] = pct('mom12_1') >= 0.8
        C['impulso12_bajo'] = pct('mom12_1') <= 0.2
        C['cae5'] = pct('r5') <= 0.1
        C['sube5'] = pct('r5') >= 0.9
        C['rsi2_bajo'] = raw('rsi2') <= 10
        C['rsi2_alto'] = raw('rsi2') >= 90
        C['rsi14_bajo'] = pct('rsi14') <= 0.3
        C['rsi14_alto'] = pct('rsi14') >= 0.7
        C['max52'] = raw('pos52') >= 0.9
        C['min52'] = raw('pos52') <= 0.1
        C['vol_baja'] = pct('atrp') <= 0.4
        C['vol_alta'] = pct('atrp') >= 0.6
        C['volumen'] = D.volraw >= 1.5
        C['hueco_abajo'] = raw('gap') <= -0.02
        C['hueco_arriba'] = raw('gap') >= 0.02
        C['mercado_alza'] = bre >= 0.55
        C['mercado_baja'] = bre <= 0.45
        C['bajo_media20'] = pct('dist20') <= 0.2
        C['sobre_media20'] = pct('dist20') >= 0.8
        C['liquido'] = pct('dvol') >= 0.7
    C['velas_alcistas'] = pat('n_velas_up') > 0
    C['velas_bajistas'] = pat('n_velas_dn') > 0
    C['figuras_alcistas'] = pat('n_fig_up') > 0
    C['figuras_bajistas'] = pat('n_fig_dn') > 0
    C['rupturas_alcistas'] = pat('n_rup_up') > 0
    C['rupturas_bajistas'] = pat('n_rup_dn') > 0
    C['confluencia_alcista'] = (pat('n_velas_up') + pat('n_fig_up') + pat('n_rup_up')) >= 2
    C['confluencia_bajista'] = (pat('n_velas_dn') + pat('n_fig_dn') + pat('n_rup_dn')) >= 2
    assert set(C) == set(DESC), set(C) ^ set(DESC)
    return {k: np.asarray(v, dtype=bool) for k, v in C.items()}


class Panel:
    """Filas del universo con el resultado neto de cada salida, el exceso sobre el día y la biblioteca de combinaciones (condición sola, par de condiciones o ninguna = todas las filas)."""

    def __init__(self, D: ml.Data, work: Dict[str, Dict], min_day: int = MIN_DAY, min_combo: int = MIN_COMBO, library: bool = True):
        self.D = D
        nrow = len(D.sym)
        cnt = np.bincount(D.sym, minlength=len(D.syms))
        off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        ops = np.empty((nrow, 2, NE), np.float32)
        code = np.empty((nrow, 2, NE), np.int8)
        for j, s in enumerate(D.syms):
            sl = slice(off[j], off[j] + cnt[j])
            ops[sl] = work[s]['ops'][D.bar[sl]]
            code[sl] = work[s]['code'][D.bar[sl]]
        self.net = ops                                                       # absoluto: (filas, sentido, salida)
        self.code = code                                                     # 0 stop inicial · 1 aseguró el primer nivel · 2 el segundo · 3 cierre por tiempo sin asegurar
        self.nd, self.nsym = D.nd, len(D.syms)
        self.blk = D.dcode // BLOCK
        self.nb = int(self.blk.max()) + 1
        # exceso sobre el día; matriz (día, valor, sentido, salida) para poder desplazar el calendario en el nulo
        R = np.full((self.nd, self.nsym, 2, NE), np.nan, dtype=np.float32)
        W = np.full((self.nd, self.nsym, 2, NE), np.nan, dtype=np.float32)   # fiabilidad: ganó (1/0) menos la fracción de filas del día que ganaron con esa salida
        ok = np.isfinite(ops)
        for di in range(2):
            for ei in range(NE):
                v = ok[:, di, ei]
                cd = np.bincount(D.dcode[v], minlength=self.nd)
                sd = np.bincount(D.dcode[v], weights=ops[v, di, ei], minlength=self.nd)
                mean = np.where(cd >= min_day, sd / np.maximum(cd, 1), np.nan)
                R[D.dcode[v], D.sym[v], di, ei] = ops[v, di, ei] - mean[D.dcode[v]]
                win = (ops[v, di, ei] > 0).astype(np.float64)
                sw = np.bincount(D.dcode[v], weights=win, minlength=self.nd)
                wmean = np.where(cd >= min_day, sw / np.maximum(cd, 1), np.nan)
                W[D.dcode[v], D.sym[v], di, ei] = win - wmean[D.dcode[v]]
        self.R, self.W = R, W
        self.rel = R[D.dcode, D.sym]                                         # (filas, 2, salidas), NaN si ese día no hay referencia
        self.relw = W[D.dcode, D.sym]
        cond = conditions(D)
        self.cond_names = list(cond)
        names = [()] + ([(k,) for k in cond] + [pr for pr in combinations(cond, 2)] if library else [])
        self.combos: List[Tuple[str, ...]] = []
        self.idx: List[np.ndarray] = []
        for nm in names:
            m = np.ones(nrow, bool)
            for k in nm:
                m &= cond[k]
            ix = np.flatnonzero(m).astype(np.int32)
            if len(nm) == 0 or len(ix) >= min_combo:
                self.combos.append(nm)
                self.idx.append(ix)
        self.G = [block_groups(self.nb, 1 if e['maxb'] <= BLOCK else LONG_BLOCKS) for e in EXITS]       # grupos de cada salida
        self.Gpool = block_groups(self.nb, LONG_BLOCKS)
        self.npat = np.array([sum(k in PATTERN_CONDS for k in nm) for nm in self.combos])
        self.sub = {'todo': np.array([len(nm) > 0 for nm in self.combos]), 'contexto': np.array([len(nm) > 0 for nm in self.combos]) & (self.npat == 0),
                    'patrones': self.npat > 0}
        # año de cada día, año de cada bloque (el de su primer día) y primer día de cada año
        self.dyear = np.zeros(self.nd, int)
        self.dyear[D.dcode] = D.year
        self.byear = np.zeros(self.nb, int)
        for b in range(self.nb):
            seg = self.dyear[b * BLOCK:(b + 1) * BLOCK]
            seg = seg[seg > 0]
            self.byear[b] = seg[0] if len(seg) else 0
        self.ystart = {int(y): int(np.flatnonzero(self.dyear == y)[0]) for y in np.unique(self.dyear) if y > 0}

    def tables(self, rel: np.ndarray, absolute: bool = False, relw: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Por combinación, sentido, salida y bloque: suma del exceso (S) y número de operaciones (N); con `relw`, suma de la ganancia relativa (SX: fiabilidad frente al día);
        con `absolute`, suma del neto (SA), de las ganadoras (SW) y de las que aseguraron el primer (SL1) y el segundo nivel (SL2)."""
        C = len(self.combos)
        shape = (C, 2, NE, self.nb)
        S, N = np.zeros(shape), np.zeros(shape)
        SX = np.zeros(shape) if relw is not None else None
        SA, SW, SL1, SL2 = (np.zeros(shape) for _ in range(4)) if absolute else (None, None, None, None)
        for ci, ix in enumerate(self.idx):
            b = self.blk[ix]
            x = rel[ix]
            ok = np.isfinite(x)
            xv = np.where(ok, x, 0.0)
            if relw is not None:
                wv = np.where(ok, relw[ix], 0.0)
            if absolute:
                a = self.net[ix]
                cdx = self.code[ix]
                oka = np.isfinite(a)
                av = np.where(oka, a, 0.0)
            for di in range(2):
                for ei in range(NE):
                    S[ci, di, ei] = np.bincount(b, weights=xv[:, di, ei], minlength=self.nb)
                    N[ci, di, ei] = np.bincount(b, weights=ok[:, di, ei], minlength=self.nb)
                    if relw is not None:
                        SX[ci, di, ei] = np.bincount(b, weights=wv[:, di, ei], minlength=self.nb)
                    if absolute:
                        SA[ci, di, ei] = np.bincount(b, weights=av[:, di, ei], minlength=self.nb)
                        SW[ci, di, ei] = np.bincount(b, weights=(a[:, di, ei] > 0) & oka[:, di, ei], minlength=self.nb)
                        SL1[ci, di, ei] = np.bincount(b, weights=(cdx[:, di, ei] >= 1) & (cdx[:, di, ei] <= 2) & oka[:, di, ei], minlength=self.nb)
                        SL2[ci, di, ei] = np.bincount(b, weights=(cdx[:, di, ei] == 2) & oka[:, di, ei], minlength=self.nb)
        out = {'S': S, 'N': N}
        if relw is not None:
            out['SX'] = SX
        if absolute:
            out.update({'SA': SA, 'SW': SW, 'SL1': SL1, 'SL2': SL2})
        return out


# ── estadística ───────────────────────────────────────────────────────────
def block_groups(nb: int, g: int) -> np.ndarray:
    """Matriz (bloques base × grupos) que junta `g` bloques de 10 sesiones en un grupo para los errores estándar agrupados."""
    G = np.zeros((nb, (nb + g - 1) // g))
    G[np.arange(nb), np.arange(nb) // g] = 1.0
    return G


def t_blocks(S: np.ndarray, N: np.ndarray, G: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n, media y t con errores estándar agrupados, a partir de las sumas por bloque (último eje); `G` junta bloques en grupos más largos."""
    n = N.sum(-1)
    if G is not None:
        S, N = S @ G, N @ G
    with np.errstate(invalid='ignore', divide='ignore'):
        m = S.sum(-1) / n
        se = np.sqrt(((S - m[..., None] * N) ** 2).sum(-1)) / n
        t = np.where(se > 0, m / se, np.nan)
    return n, m, t


def stats_by_exit(P: Panel, T: Dict[str, np.ndarray], tr: np.ndarray, key: str = 'S') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """n, media, t de la tabla `key` (S = exceso; SX = ganancia relativa, la fiabilidad) y cobertura de todas las combinaciones (regla, sentido, salida) en los bloques `tr`,
    con los grupos de error de cada salida. Cobertura = fracción de los periodos (grupos) del universo en esos bloques en los que la regla tiene alguna operación."""
    C = T['S'].shape[0]
    n, m, t, cov = (np.zeros((C, 2, NE)) for _ in range(4))
    for e in range(NE):
        Ne = T['N'][:, :, e, :][..., tr]
        G = P.G[e][tr]
        n[:, :, e], m[:, :, e], t[:, :, e] = t_blocks(T[key][:, :, e, :][..., tr], Ne, G)
        tot = max(int(((T['N'][0, 0, e, :][tr] @ G) > 0).sum()), 1)              # periodos con alguna fila del universo (la combinación 0 = todas las filas)
        has = ((Ne @ G) > 0).sum(-1)
        cov[:, :, e] = has / tot
        t[:, :, e] = np.where(has >= MIN_GROUPS, t[:, :, e], np.nan)
    return n, m, t, cov


MODES = ('rent', 'fiab')          # rent = se elige la regla de mayor t de exceso de rentabilidad; fiab = la de mayor t de ganancia relativa (gana más veces que el azar del día), con exceso > 0


def pick(P: Panel, T: Dict[str, np.ndarray], sub: np.ndarray, tr: np.ndarray, min_n: int = MIN_TRAIN_N, mode: str = 'rent') -> Optional[Tuple[int, int, int, float, float, float]]:
    """Combinación (regla, sentido, salida) elegible de mayor t (de exceso, o de fiabilidad con exceso positivo) en los bloques `tr`; solo entre las reglas de `sub`."""
    n, m, t, cov = stats_by_exit(P, T, tr)
    if mode == 'fiab':
        _, mw, tw, _ = stats_by_exit(P, T, tr, 'SX')
        score = np.where((n >= min_n) & (m > 0) & (mw > 0) & (cov >= MIN_COVER) & np.isfinite(tw), tw, -np.inf)
    else:
        score = np.where((n >= min_n) & (m > 0) & (cov >= MIN_COVER) & np.isfinite(t), t, -np.inf)
    score[~sub] = -np.inf
    k = int(np.argmax(score))
    if not np.isfinite(score.flat[k]):
        return None
    c, d, e = np.unravel_index(k, score.shape)
    return int(c), int(d), int(e), float(n[c, d, e]), float(m[c, d, e]), float(score[c, d, e])


def walk_forward(P: Panel, T: Dict[str, np.ndarray], sub: np.ndarray, test_years: List[int], absolute: bool = False, mode: str = 'rent') -> Dict:
    """Para cada año de prueba se elige con los datos anteriores (menos PURGE sesiones) y se evalúa en el año; se acumulan los bloques del año."""
    nb = P.nb
    So, No, SXo = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    SAo, SWo, SAb, SWb, Nb, L1o, L2o = (np.zeros(nb) for _ in range(7))
    rows = []
    for y in test_years:
        d0 = P.ystart.get(y)
        if d0 is None:
            continue
        tr = ((np.arange(nb) + 1) * BLOCK <= d0 - PURGE)
        te = P.byear == y
        if tr.sum() < 20 or not te.any():
            continue
        p = pick(P, T, sub, tr, mode=mode)
        if p is None:
            continue
        c, d, e, ntr, mtr, ttr = p
        So[te] += T['S'][c, d, e, te]
        No[te] += T['N'][c, d, e, te]
        row = {'año': int(y), 'regla': list(P.combos[c]), 'sentido': DIRS[d], 'salida': EXN[e], 'train_n': int(ntr), 'train_exceso': round(mtr, 3), 'train_t': round(ttr, 2)}
        n_, m_, t_ = t_blocks(T['S'][c, d, e, te], T['N'][c, d, e, te], P.G[e][te])
        row.update({'n': int(n_), 'exceso': round(float(m_), 3) if n_ else None})
        if 'SX' in T:
            SXo[te] += T['SX'][c, d, e, te]
            row['fiab'] = round(float(100 * T['SX'][c, d, e, te].sum() / max(n_, 1)), 1)        # puntos de ganancia por encima del azar del día
        if absolute:
            SAo[te] += T['SA'][c, d, e, te]
            SWo[te] += T['SW'][c, d, e, te]
            L1o[te] += T['SL1'][c, d, e, te]
            L2o[te] += T['SL2'][c, d, e, te]
            SAb[te] += T['SA'][0, d, e, te]
            SWb[te] += T['SW'][0, d, e, te]
            Nb[te] += T['N'][0, d, e, te]
            row['neto'] = round(float(T['SA'][c, d, e, te].sum() / max(n_, 1)), 3)
            row['ganan'] = round(float(100 * T['SW'][c, d, e, te].sum() / max(n_, 1)), 1)
            row['asegura4'] = round(float(100 * T['SL1'][c, d, e, te].sum() / max(n_, 1)), 1)
            row['asegura10'] = round(float(100 * T['SL2'][c, d, e, te].sum() / max(n_, 1)), 1)
            row['azar_neto'] = round(float(T['SA'][0, d, e, te].sum() / max(T['N'][0, d, e, te].sum(), 1)), 3)
            row['azar_ganan'] = round(float(100 * T['SW'][0, d, e, te].sum() / max(T['N'][0, d, e, te].sum(), 1)), 1)
        rows.append(row)
    n, m, t = t_blocks(So[None, :], No[None, :], P.Gpool)
    res = {'modo': mode, 'años': rows, 'n': int(n[0]), 'exceso': float(m[0]) if n[0] else None, 't': float(t[0]) if n[0] else None}
    if 'SX' in T and n[0]:
        _, mw, tw = t_blocks(SXo[None, :], No[None, :], P.Gpool)
        res['fiab'], res['fiab_t'] = float(100 * mw[0]), float(tw[0])
    if absolute and n[0]:
        res['neto'] = float(SAo.sum() / n[0])
        res['ganan'] = float(100 * SWo.sum() / n[0])
        res['asegura4'] = float(100 * L1o.sum() / n[0])
        res['asegura10'] = float(100 * L2o.sum() / n[0])
        res['neto_azar_mismas_fechas'] = float(res['neto'] - res['exceso'])
        res['ganan_todas_las_filas'] = float(100 * SWb.sum() / max(Nb.sum(), 1))
    return res


def static_split(P: Panel, T: Dict[str, np.ndarray], sub: np.ndarray, dev_end: int, top: int = 15) -> Dict:
    """Se eligen las mejores combinaciones con los años ≤ dev_end y se miran una sola vez en los años posteriores (menos el purgado)."""
    d_end = P.ystart.get(dev_end + 1)
    if d_end is None:
        return {}
    tr = ((np.arange(P.nb) + 1) * BLOCK <= d_end - PURGE)
    te = (np.arange(P.nb) * BLOCK >= d_end)
    n, m, t, cov = stats_by_exit(P, T, tr)
    t = np.where((n >= MIN_TRAIN_N) & (m > 0) & (cov >= MIN_COVER) & np.isfinite(t), t, -np.inf)
    t[~sub] = -np.inf
    order = np.argsort(-t.ravel())[:top]
    out = []
    for k in order:
        if not np.isfinite(t.flat[k]):
            continue
        c, d, e = np.unravel_index(k, t.shape)
        nf, mf, tf = t_blocks(T['S'][c, d, e, te][None, :], T['N'][c, d, e, te][None, :], P.G[e][te])
        out.append({'regla': list(P.combos[c]), 'sentido': DIRS[d], 'salida': EXN[e], 'train_n': int(n[c, d, e]), 'train_exceso': round(float(m[c, d, e]), 3), 'train_t': round(float(t[c, d, e]), 2),
                    'final_n': int(nf[0]), 'final_exceso': round(float(mf[0]), 3) if nf[0] else None, 'final_t': round(float(tf[0]), 2) if nf[0] and np.isfinite(tf[0]) else None})
    return {'dev_end': dev_end, 'mejores': out}


def in_sample_best(P: Panel, T: Dict[str, np.ndarray], sub: np.ndarray, top: int = 10) -> List[Dict]:
    """Las mejores combinaciones mirando TODOS los años (descriptivo: está elegido con los mismos datos con los que se mide)."""
    n, m, t, cov = stats_by_exit(P, T, np.ones(P.nb, bool))
    t = np.where((n >= MIN_TRAIN_N) & (m > 0) & (cov >= MIN_COVER) & np.isfinite(t), t, -np.inf)
    t[~sub] = -np.inf
    out = []
    years = sorted(int(y) for y in np.unique(P.byear) if y > 0)
    for k in np.argsort(-t.ravel())[:top]:
        if not np.isfinite(t.flat[k]):
            continue
        c, d, e = np.unravel_index(k, t.shape)
        per = {}
        for y in years:
            mk = P.byear == y
            ny, my, _ = t_blocks(T['S'][c, d, e, mk][None, :], T['N'][c, d, e, mk][None, :], P.G[e][mk])
            if ny[0] >= 100:
                per[y] = round(float(my[0]), 3)
        out.append({'regla': list(P.combos[c]), 'sentido': DIRS[d], 'salida': EXN[e], 'n': int(n[c, d, e]), 'exceso': round(float(m[c, d, e]), 3), 't': round(float(t[c, d, e]), 2),
                    'neto': round(float(T['SA'][c, d, e].sum() / n[c, d, e]), 3) if 'SA' in T else None, 'ganan': round(float(100 * T['SW'][c, d, e].sum() / n[c, d, e]), 1) if 'SW' in T else None,
                    'por_año': per})
    return out


def null_runs(P: Panel, test_years: List[int], reps: int, seed: int = 13, verbose: bool = False) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Repite todo el procedimiento (los dos modos de selección) con los resultados desplazados en el calendario (mismo desplazamiento para todos los valores)."""
    rng = np.random.default_rng(seed)
    out = {md: {s: {'exceso': [], 't': [], 'fiab': [], 'fiab_t': []} for s in SUBSETS} for md in MODES}
    t0 = time.time()
    for q in range(reps):
        sh = int(rng.integers(MIN_SHIFT, max(P.nd - MIN_SHIFT, MIN_SHIFT + 1)))
        ix = ((P.D.dcode + sh) % P.nd, P.D.sym)
        T = P.tables(P.R[ix], relw=P.W[ix])
        for md in MODES:
            for s in SUBSETS:
                r = walk_forward(P, T, P.sub[s], test_years, mode=md)
                for k in ('exceso', 't', 'fiab', 'fiab_t'):
                    out[md][s][k].append(r[k] if r.get(k) is not None else np.nan)
        if verbose and (q + 1) % 5 == 0:
            print(f'    nulo {q + 1}/{reps} · {time.time() - t0:.0f}s', flush=True)
    return {md: {s: {k: np.array(v) for k, v in d.items()} for s, d in o.items()} for md, o in out.items()}


# ── conjunto ──────────────────────────────────────────────────────────────
def build(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, verbose: bool = False, min_day: int = MIN_DAY, min_combo: int = MIN_COMBO, join: Optional[Dict[str, str]] = None,
          cost: float = COST, library: bool = True, keep_last: bool = False) -> Tuple[ml.Data, Panel, int]:
    """Descarga ya hecha → trabajo por valor (indicadores, patrones, salidas) → datos del universo → panel. Devuelve (datos, panel, barras-valor quitadas por la incorporación al índice).
    Con `keep_last` entran también las últimas sesiones (sin rendimiento a 10 sesiones todavía): el cribador necesita la fila de hoy."""
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    jobs = [(s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}, cost) for s, b in bars.items() if len(b['c']) > 400]
    work: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n_, (s, w) in enumerate(ex.map(_work, jobs, chunksize=4)):
                work[s] = w
                if verbose and (n_ + 1) % 100 == 0:
                    print(f'    {n_ + 1}/{len(jobs)} valores · {time.time() - t0:.0f}s', flush=True)
    else:
        for jb in jobs:
            s, w = _work(jb)
            work[s] = w
    dropped = apply_join(work, join) if join else 0
    if keep_last:
        for w in work.values():
            w['ret'] = np.nan_to_num(w['ret'])                                  # el panel exige rendimiento finito; aquí no se usa
    D = ml.Data(work, k=k)
    P = Panel(D, work, min_day=min_day, min_combo=min_combo, library=library)
    if verbose:
        print(f'  panel: {len(D.sym):,} filas · {len(P.combos)} combinaciones de reglas · {time.time() - t0:.0f}s', flush=True)
    return D, P, dropped


def analyze(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, reps: int = 30, test_years: Optional[List[int]] = None, dev_end: int = DEV_END, verbose: bool = False,
            min_day: int = MIN_DAY, min_combo: int = MIN_COMBO, join: Optional[Dict[str, str]] = None, cost: float = COST) -> Dict:
    t0 = time.time()
    D, P, dropped = build(bars, workers, k, verbose, min_day, min_combo, join, cost)
    work_syms = D.syms
    years = sorted(int(y) for y in np.unique(D.year))
    if test_years is None:
        test_years = [y for y in years if y >= years[0] + 3 and int((D.year == y).sum()) > 20000]
    T = P.tables(P.rel, absolute=True, relw=P.relw)
    wf = {md: {s: walk_forward(P, T, P.sub[s], test_years, absolute=True, mode=md) for s in SUBSETS} for md in MODES}
    st = {s: static_split(P, T, P.sub[s], dev_end) for s in SUBSETS}
    ins = {s: in_sample_best(P, T, P.sub[s]) for s in SUBSETS}
    if verbose:
        print(f'  selección hacia delante hecha · {time.time() - t0:.0f}s', flush=True)
    null = null_runs(P, test_years, reps, verbose=verbose) if reps else {}
    nl = {md: {} for md in MODES}
    for md in null:
        for s in SUBSETS:
            real = wf[md][s]
            q = {'reps': 0}
            for k in ('exceso', 't', 'fiab', 'fiab_t'):
                v = null[md][s][k]
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                q['reps'] = int(len(v))
                q[f'{k}_p50'], q[f'{k}_p95'], q[f'{k}_max'] = round(float(np.median(v)), 4), round(float(np.quantile(v, 0.95)), 4), round(float(v.max()), 4)
                q[f'p_{k}'] = round(float((v >= real[k]).mean()), 3) if real.get(k) is not None else None
            nl[md][s] = q
    # referencia: entrar al azar con cada salida (neto absoluto) en todas las filas
    base = {}
    for di in range(2):
        for ei in range(NE):
            n = T['N'][0, di, ei].sum()
            base[f'{DIRS[di]} {EXN[ei]}'] = {'n': int(n), 'neto': round(float(T['SA'][0, di, ei].sum() / n), 3), 'ganan': round(float(100 * T['SW'][0, di, ei].sum() / n), 1),
                                              'asegura4': round(float(100 * T['SL1'][0, di, ei].sum() / n), 1), 'asegura10': round(float(100 * T['SL2'][0, di, ei].sum() / n), 1)}
    out = {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(D.sym)), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k, 'combinaciones': len(P.combos) - 1,
                    'pruebas': (len(P.combos) - 1) * 2 * NE, 'condiciones': len(P.cond_names), 'test_years': test_years, 'dev_end': dev_end, 'reps': reps, 'salidas': {e['name']: EXIT_TXT[e['name']] for e in EXITS}, 'escalera': [list(x) for x in LADDER],
                    'miembros': 'en la fecha' if join else 'actuales', 'valores_con_fecha': int(sum(1 for s_ in work_syms if join and join.get(s_))), 'barras_quitadas': dropped, 'coste': cost},
           'walk_forward': wf['rent'], 'nulo': nl['rent'], 'walk_forward_fiab': wf['fiab'], 'nulo_fiab': nl['fiab'], 'dev_final': st, 'en_muestra': ins, 'azar_absoluto': base}
    if verbose:
        print(f'  búsqueda: {len(P.combos) - 1} combinaciones × 2 sentidos × {NE} salidas · {time.time() - t0:.0f}s', flush=True)
    return out


def _rule(nm: List[str], sentido: str) -> str:
    return (' Y '.join(DESC[k] for k in nm) if nm else 'cualquier fila') + f' → {sentido}'


def report(ev: Dict) -> str:
    m = ev['meta']
    L = [f"BÚSQUEDA DE LA REGLA QUE MÁS GANA SOBRE EL AZAR · {m['universe']} acciones · {m['filas']:,} filas (las {m['k']} más negociadas) · {m['from']} → {m['to']}",
         f"{m['combinaciones']} combinaciones de condiciones × 2 sentidos × {len(m['salidas'])} salidas = {m['pruebas']:,} reglas. Exceso = neto de costes menos la media de todas las filas de ese día (mismo sentido y salida). "
         f"Selección hacia delante: cada año de prueba se elige con lo anterior (menos {PURGE} sesiones) y se mide en el año; solo son elegibles reglas con ≥ {MIN_TRAIN_N:,} operaciones y operaciones en al menos el {int(MIN_COVER * 100)} % de los periodos del entrenamiento. "
         f"t con errores agrupados en bloques de {BLOCK} sesiones (cierre a 10) o de {BLOCK * LONG_BLOCKS} (salidas de hasta 60). Nulo: todo el procedimiento con los resultados desplazados en el calendario (al menos {MIN_SHIFT} sesiones; {m['reps']} repeticiones).",
         f"Universo: {'miembros ACTUALES de los índices' if m.get('miembros') == 'actuales' else 'miembros actuales, pero cada valor del S&P 500 solo desde su fecha de incorporación (' + str(m.get('valores_con_fecha')) + ' valores con fecha; ' + format(m.get('barras_quitadas', 0), ',') + ' barras-valor quitadas)'} (sesgo de supervivencia: los que salieron del índice no están).",
         'Salidas: ' + ' · '.join(f'{k} = {v}' for k, v in m['salidas'].items()), '']
    for md, wfk, nlk, mtxt in (('rent', 'walk_forward', 'nulo', 'elegida por RENTABILIDAD (mayor t de exceso)'), ('fiab', 'walk_forward_fiab', 'nulo_fiab', 'elegida por FIABILIDAD (mayor t de «gana más veces que el azar del día», con exceso > 0)')):
        if wfk not in ev:
            continue
        for s, titulo in (('todo', 'TODAS LAS REGLAS'), ('contexto', 'SOLO CONTEXTO (sin patrones)'), ('patrones', 'CON PATRÓN (al menos una condición de patrón)')):
            w, nl = ev[wfk][s], ev[nlk].get(s)
            L.append(f'=== {titulo} · {mtxt} · selección hacia delante, años {m["test_years"][0]}–{m["test_years"][-1]} ===')
            if w['exceso'] is None:
                L += ['  sin resultados', '']
                continue
            L.append(f"  FUERA DE MUESTRA: {w['n']:,} operaciones · exceso {w['exceso']:+.3f} % por operación (t {w['t']:+.1f}) · neto {w['neto']:+.3f} % (azar en las mismas fechas {w['neto_azar_mismas_fechas']:+.3f} %) · ganan {w['ganan']:.1f} % (todas las filas {w['ganan_todas_las_filas']:.1f} %)"
                     + (f" · fiabilidad {w['fiab']:+.1f} puntos sobre el azar del día (t {w['fiab_t']:+.1f})" if 'fiab' in w else '') + (f" · aseguró +4 % el {w['asegura4']:.1f} % y +10 % el {w['asegura10']:.1f} %" if 'asegura4' in w else ''))
            if nl:
                L.append(f"  Nulo del procedimiento: exceso mediano {nl.get('exceso_p50', float('nan')):+.3f} %, p95 {nl.get('exceso_p95', float('nan')):+.3f} %; t p95 {nl.get('t_p95', float('nan')):+.1f} · p del exceso {nl.get('p_exceso')} y de la t {nl.get('p_t')}"
                         + (f" · fiabilidad p95 {nl['fiab_p95']:+.1f} puntos, p {nl.get('p_fiab')} (t p95 {nl.get('fiab_t_p95', float('nan')):+.1f}, p {nl.get('p_fiab_t')})" if 'fiab_p95' in nl else ''))
            L.append(f"  {'año':>5s} {'n':>8s} {'exceso':>8s} {'neto':>8s} {'azar':>8s} {'ganan':>6s} {'+4%':>5s} {'+10%':>5s}  regla elegida con el pasado (t de entrenamiento)")
            for r in w['años']:
                L.append(f"  {r['año']:5d} {r['n']:8,} {r['exceso'] if r['exceso'] is not None else float('nan'):+8.3f} {r.get('neto', float('nan')):+8.3f} {r.get('azar_neto', float('nan')):+8.3f} {r.get('ganan', float('nan')):6.1f} {r.get('asegura4', float('nan')):5.1f} {r.get('asegura10', float('nan')):5.1f}  "
                         f"{_rule(r['regla'], r['sentido'])} · {r['salida']} (t {r['train_t']:.1f}, exceso {r['train_exceso']:+.2f} %)")
            L.append('')
    L.append(f"=== REGLAS ELEGIDAS CON LOS AÑOS ≤ {ev['meta']['dev_end']} Y MIRADAS UNA SOLA VEZ DESPUÉS ===")
    for s in SUBSETS:
        d = ev['dev_final'].get(s, {})
        if not d.get('mejores'):
            continue
        L.append(f'  [{s}]')
        L.append(f"  {'entren. n':>10s} {'exceso':>8s} {'t':>6s} | {'final n':>8s} {'exceso':>8s} {'t':>6s}  regla")
        for r in d['mejores']:
            fe = '   n/d' if r['final_exceso'] is None else f"{r['final_exceso']:+8.3f}"
            ft = '   n/d' if r['final_t'] is None else f"{r['final_t']:+6.1f}"
            L.append(f"  {r['train_n']:10,} {r['train_exceso']:+8.3f} {r['train_t']:+6.1f} | {r['final_n']:8,} {fe} {ft}  {_rule(r['regla'], r['sentido'])} · {r['salida']}")
        pos = [r for r in d['mejores'] if r['final_exceso'] is not None]
        if pos:
            L.append(f"  → de {len(pos)} reglas, {sum(r['final_exceso'] > 0 for r in pos)} mantienen exceso positivo en el final; exceso final medio {np.mean([r['final_exceso'] for r in pos]):+.3f} % (entrenamiento medio {np.mean([r['train_exceso'] for r in pos]):+.3f} %)")
    L += ['', '=== LAS MEJORES MIRANDO TODOS LOS AÑOS (descriptivo: están elegidas con los mismos datos con los que se miden; la cifra fiable es la de fuera de muestra) ===']
    for s in SUBSETS:
        L.append(f'  [{s}]')
        for r in ev['en_muestra'].get(s, [])[:6]:
            ys = ' '.join(f"{str(y)[2:]}:{v:+.1f}" for y, v in r['por_año'].items())
            L.append(f"  n {r['n']:8,} exceso {r['exceso']:+.3f} % (t {r['t']:+.1f}) neto {r['neto']:+.3f} % ganan {r['ganan']:.1f} %  {_rule(r['regla'], r['sentido'])} · {r['salida']}   años: {ys}")
    L += ['', '=== ENTRAR AL AZAR (todas las filas) CON CADA SALIDA: neto medio por operación y % que termina con ganancia ===']
    for k, v in ev['azar_absoluto'].items():
        L.append(f"  {k:12s} n {v['n']:9,}  neto {v['neto']:+.3f} %  ganan {v['ganan']:.1f} %" + (f"  asegura +4 % {v['asegura4']:.1f} %  +10 % {v['asegura10']:.1f} %" if 'asegura4' in v else ''))
    return '\n'.join(L)


# ── confirmación en otras acciones ────────────────────────────────────────
# Reglas candidatas fijadas con la búsqueda sobre las 300 acciones más negociadas del S&P 500 ∪ Nasdaq-100 (2017–2026), ANTES de mirarlas en acciones distintas. Son las que aguantaron
# la selección hacia delante y la mayoría de los años: valores de impulso alto a 12 meses (sin el último mes) que han caído por debajo de su media de 200 sesiones o que acumulan señales bajistas.
CANDIDATAS = (
    {'nombre': 'C1', 'regla': ['bajo_200', 'impulso12_alto'], 'sentido': 'largo', 'salida': 'op4b'},
    {'nombre': 'C2', 'regla': ['bajo_200', 'impulso12_alto'], 'sentido': 'largo', 'salida': 'op6b'},
    {'nombre': 'C3', 'regla': ['bajo_200', 'impulso12_alto'], 'sentido': 'largo', 'salida': 'h10'},
    {'nombre': 'C4', 'regla': ['impulso12_alto', 'confluencia_bajista'], 'sentido': 'largo', 'salida': 'h10'},
    {'nombre': 'C5', 'regla': ['impulso12_alto', 'rsi2_bajo'], 'sentido': 'largo', 'salida': 'h10'},
    # la escalera del operador (+4 % → stop a +4 % plano; +10 % → stop a +10 % que persigue a 2 ATR), fijada por el operador antes de mirar datos
    {'nombre': 'C6', 'regla': ['bajo_200', 'impulso12_alto'], 'sentido': 'largo', 'salida': 'esc'},
    {'nombre': 'C7', 'regla': ['bajo_200', 'impulso12_alto'], 'sentido': 'largo', 'salida': 'esc6'},
    {'nombre': 'C8', 'regla': ['impulso12_alto', 'confluencia_bajista'], 'sentido': 'largo', 'salida': 'esc'},
)


# Las mismas entradas, sin la salida: lo que aplica el cribador diario (parte 2). La salida es siempre la escalera del operador (`esc`); los nombres son los que ve el operador.
ENTRADAS = (
    {'nombre': 'imp_bajo200', 'etiqueta': 'Impulso alto y bajo su media de 200', 'corta': 'bajo 200', 'regla': ['bajo_200', 'impulso12_alto'], 'sentido': 'largo', 'salida': 'esc'},
    {'nombre': 'imp_bajistas', 'etiqueta': 'Impulso alto y 2 o más señales bajistas', 'corta': '2+ bajistas', 'regla': ['impulso12_alto', 'confluencia_bajista'], 'sentido': 'largo', 'salida': 'esc'},
    {'nombre': 'imp_rsi2', 'etiqueta': 'Impulso alto y RSI(2) ≤ 10', 'corta': 'RSI(2) ≤ 10', 'regla': ['impulso12_alto', 'rsi2_bajo'], 'sentido': 'largo', 'salida': 'esc'},
)
ENTRY_TXT = {tuple(sorted(e['regla'])): e['etiqueta'] for e in ENTRADAS}
EXIT_SHORT = {'op4a': 'asegura +4 % (stop 1,5 ATR)', 'op4b': 'asegura +4 % y persigue 3 ATR', 'op6a': 'asegura +6 % (stop 1,5 ATR)', 'op6b': 'asegura +6 % y persigue 4 ATR', 'run5': 'persigue 5 ATR',
              'h10': 'cierre a 10 sesiones', 'esc': 'escalera', 'esca': 'escalera (stop 1,5 ATR)', 'esc6': 'escalera (stop 6 %)', 'esc10': 'escalera (stop 10 %)'}


def label(rule: Dict) -> str:
    """Nombre legible de una regla candidata: entrada · salida (y la volatilidad si la lleva). Los códigos C1…C8 / V1…V6 son solo identificadores de los ficheros de datos."""
    base = [k for k in rule['regla'] if k not in ('vol_baja', 'vol_alta')]
    ent = ENTRY_TXT.get(tuple(sorted(base)), _rule(base, ''))
    vol = ' · volatilidad baja' if 'vol_baja' in rule['regla'] else ' · volatilidad alta' if 'vol_alta' in rule['regla'] else ''
    return f"{ent}{vol} · {EXIT_SHORT.get(rule['salida'], rule['salida'])}"


# Robustez (DESCRIPTIVA, no fijada de antemano): la misma regla base según la volatilidad del valor (ATR / precio en el 40 % más bajo o más alto del día). Responde a si la ventaja existe
# también en valores tranquilos, donde el stop inicial es normal, o solo en los muy volátiles.
ROBUSTEZ = (
    {'nombre': 'V1', 'regla': ['bajo_200', 'impulso12_alto', 'vol_baja'], 'sentido': 'largo', 'salida': 'op4b'},
    {'nombre': 'V2', 'regla': ['bajo_200', 'impulso12_alto', 'vol_alta'], 'sentido': 'largo', 'salida': 'op4b'},
    {'nombre': 'V3', 'regla': ['bajo_200', 'impulso12_alto', 'vol_baja'], 'sentido': 'largo', 'salida': 'h10'},
    {'nombre': 'V4', 'regla': ['bajo_200', 'impulso12_alto', 'vol_alta'], 'sentido': 'largo', 'salida': 'h10'},
    {'nombre': 'V5', 'regla': ['bajo_200', 'impulso12_alto', 'vol_baja'], 'sentido': 'largo', 'salida': 'esc'},
    {'nombre': 'V6', 'regla': ['bajo_200', 'impulso12_alto', 'vol_alta'], 'sentido': 'largo', 'salida': 'esc'},
)
LABELS = {r['nombre']: label(r) for r in CANDIDATAS + ROBUSTEZ + ENTRADAS}        # código → nombre legible (la web y los informes enseñan el nombre; el código queda en los datos)


def evaluate_rules(P: Panel, rules: Tuple[Dict, ...], reps: int = 300, seed: int = 21) -> List[Dict]:
    """Exceso, t agrupado, neto, % ganadoras, años positivos y p-valores (resultados desplazados en el calendario) de reglas FIJAS: una sola hipótesis por regla, sin elegir nada."""
    D = P.D
    cond = conditions(D)
    exn = EXN
    rng = np.random.default_rng(seed)
    shifts = rng.integers(MIN_SHIFT, max(P.nd - MIN_SHIFT, MIN_SHIFT + 1), size=reps)
    out = []
    for r in rules:
        m = np.ones(len(D.sym), bool)
        for k in r['regla']:
            m &= cond[k]
        ix = np.flatnonzero(m)
        d, e = DIRS.index(r['sentido']), exn.index(r['salida'])
        b = P.blk[ix]

        def stat(x: np.ndarray) -> Tuple[float, float, float]:
            ok = np.isfinite(x)
            S = np.bincount(b[ok], weights=x[ok], minlength=P.nb)
            N = np.bincount(b[ok], minlength=P.nb).astype(float)
            n, mm, tt = t_blocks(S, N, P.G[e])
            return float(n), float(mm), float(tt)
        n, mean, t = stat(P.rel[ix, d, e])
        nul = np.array([stat(P.R[(D.dcode[ix] + sh) % P.nd, D.sym[ix], d, e]) for sh in shifts])
        _, fmean, ft = stat(P.relw[ix, d, e])                                   # fiabilidad: ganó menos la fracción del día que ganó (puntos)
        nulw = np.array([stat(P.W[(D.dcode[ix] + sh) % P.nd, D.sym[ix], d, e])[1] for sh in shifts])
        ab = P.net[ix, d, e]
        okb = np.isfinite(ab)
        allnet = P.net[:, d, e]
        cdx = P.code[ix, d, e][okb]
        wins, loss = ab[okb][ab[okb] > 0], ab[okb][ab[okb] <= 0]
        extra = {'fiab': round(float(100 * fmean), 2), 'fiab_t': round(float(ft), 2), 'p_fiab': round(float((nulw >= fmean).mean()), 3),
                 'asegura4': round(float(100 * ((cdx >= 1) & (cdx <= 2)).mean()), 1), 'asegura10': round(float(100 * (cdx == 2).mean()), 1), 'stop_inicial': round(float(100 * (cdx == 0).mean()), 1),
                 'gan_media': round(float(wins.mean()), 2) if len(wins) else None, 'perd_media': round(float(loss.mean()), 2) if len(loss) else None,
                 'factor': round(float(wins.sum() / max(-loss.sum(), 1e-9)), 2) if len(loss) else None}
        yr = {}
        for y in sorted(int(v) for v in np.unique(D.year[ix])):
            xs = P.rel[ix, d, e][(D.year[ix] == y)]
            xs = xs[np.isfinite(xs)]
            if len(xs) >= 100:
                yr[y] = round(float(xs.mean()), 3)
        out.append({**r, 'n': int(n), 'valores': int(len(np.unique(D.sym[ix]))), 'exceso': round(mean, 3), 't': round(t, 2), 'neto': round(float(ab[okb].mean()), 3), 'azar_neto': round(float(ab[okb].mean() - mean), 3),
                    'ganan': round(float(100 * (ab[okb] > 0).mean()), 1), 'ganan_todas': round(float(100 * (allnet[np.isfinite(allnet)] > 0).mean()), 1), 'por_año': yr,
                    'nulo_exceso_p95': round(float(np.quantile(nul[:, 1], 0.95)), 3), 'nulo_t_p95': round(float(np.nanquantile(nul[:, 2], 0.95)), 2),
                    'p_exceso': round(float((nul[:, 1] >= mean).mean()), 3), 'p_t': round(float((nul[np.isfinite(nul[:, 2]), 2] >= t).mean()), 3), 'reps': int(reps), **extra})
    return out


def confirm(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, cost: float = COST, reps: int = 300, join: Optional[Dict[str, str]] = None, verbose: bool = False,
            rules: Tuple[Dict, ...] = CANDIDATAS, min_day: int = MIN_DAY, robustez: Tuple[Dict, ...] = ROBUSTEZ) -> Dict:
    D, P, dropped = build(bars, workers, k, verbose, min_day, MIN_COMBO, join, cost, library=False)
    res = evaluate_rules(P, rules, reps)
    rob = evaluate_rules(P, robustez, reps) if robustez else []
    return {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(D.sym)), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k, 'coste': cost, 'reps': reps,
                     'miembros': 'en la fecha' if join else 'actuales', 'barras_quitadas': dropped, 'salidas': {e['name']: EXIT_TXT[e['name']] for e in EXITS}}, 'reglas': res, 'robustez': rob}


def confirm_report(ev: Dict, titulo: str = '') -> str:
    m = ev['meta']
    L = [f"CONFIRMACIÓN DE LAS REGLAS CANDIDATAS{' · ' + titulo if titulo else ''} · {m['universe']} acciones · {m['filas']:,} filas (las {m['k']} más negociadas) · {m['from']} → {m['to']} · coste {m['coste']} %",
         f'Reglas fijadas antes de mirar estas acciones (una hipótesis cada una, no se elige nada). Exceso = neto menos la media de todas las filas de ese día. p = fracción de repeticiones con los resultados desplazados en el calendario (al menos {MIN_SHIFT} sesiones) que igualan o superan lo observado.',
         f"Universo: {'miembros actuales' if m['miembros'] == 'actuales' else 'miembros actuales desde su incorporación al S&P 500'}. Salidas: " + ' · '.join(f'{k} = {v}' for k, v in m['salidas'].items()), '']
    for r in ev['reglas']:
        ys = ' '.join(f"{str(y)[2:]}:{v:+.1f}" for y, v in r['por_año'].items())
        pos = sum(v > 0 for v in r['por_año'].values())
        L += [f"{r['nombre']} «{label(r)}» · {_rule(r['regla'], r['sentido'])} · {r['salida']}",
              f"   n {r['n']:,} ({r['valores']} valores) · exceso {r['exceso']:+.3f} % (t {r['t']:+.1f}; nulo p95 {r['nulo_exceso_p95']:+.3f} % / t {r['nulo_t_p95']:+.1f}; p exceso {r['p_exceso']}, p t {r['p_t']}) · neto {r['neto']:+.3f} % (azar mismas fechas {r['azar_neto']:+.3f} %) · ganan {r['ganan']:.1f} % (todas las filas {r['ganan_todas']:.1f} %)",
              f"   fiabilidad {r.get('fiab', float('nan')):+.1f} puntos sobre el azar del día (t {r.get('fiab_t', float('nan')):+.1f}, p {r.get('p_fiab')}) · aseguró +4 % {r.get('asegura4', float('nan')):.1f} %, +10 % {r.get('asegura10', float('nan')):.1f} %, stop inicial {r.get('stop_inicial', float('nan')):.1f} % · ganancia media {r.get('gan_media')} % / pérdida media {r.get('perd_media')} % · factor de beneficio {r.get('factor')}",
              f"   años positivos {pos}/{len(r['por_año'])}: {ys}"]
    if ev.get('robustez'):
        L += ['', 'ROBUSTEZ POR VOLATILIDAD (descriptiva, no fijada de antemano): la misma regla base en valores tranquilos (ATR/precio en el 40 % más bajo del día) o muy volátiles (40 % más alto)']
        for r in ev['robustez']:
            ys = ' '.join(f"{str(y)[2:]}:{v:+.1f}" for y, v in r['por_año'].items())
            pos = sum(v > 0 for v in r['por_año'].values())
            L += [f"{r['nombre']} «{label(r)}» · {_rule(r['regla'], r['sentido'])} · {r['salida']}",
                  f"   n {r['n']:,} ({r['valores']} valores) · exceso {r['exceso']:+.3f} % (t {r['t']:+.1f}; p exceso {r['p_exceso']}, p t {r['p_t']}) · neto {r['neto']:+.3f} % (azar mismas fechas {r['azar_neto']:+.3f} %) · ganan {r['ganan']:.1f} % (todas las filas {r['ganan_todas']:.1f} %)",
                  f"   años positivos {pos}/{len(r['por_año'])}: {ys}"]
    return '\n'.join(L)


def confirm_run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, reps: int = 300, cost: float = COST, k: int = ml.K, tag: str = '', members: str = 'asof') -> Dict:
    from . import data, universe
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    join = None
    if members == 'asof' and not tag:                                      # la tabla de incorporaciones es la del S&P 500: solo se aplica al universo grande
        try:
            join = universe.sp500_join_dates()
        except Exception as exc:
            print(f'  no se pudo leer la tabla de miembros del S&P 500 ({type(exc).__name__}: {exc}): se usan los miembros actuales', flush=True)
    ev = confirm(bars, workers=workers, k=k, cost=cost, reps=reps, join=join, verbose=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, f'search_confirm{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = confirm_report(ev, tag.strip('_'))
    open(os.path.join(out_dir, f'search_confirm{tag}.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, reps: int = 30, members: str = 'asof') -> Dict:
    from . import data, universe
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    join = None
    if members == 'asof':
        try:
            join = universe.sp500_join_dates()
            print(f'  fechas de incorporación al S&P 500: {len(join)} valores', flush=True)
        except Exception as exc:
            print(f'  no se pudo leer la tabla de cambios del S&P 500 ({type(exc).__name__}: {exc}): se usan los miembros actuales', flush=True)
    ev = analyze(bars, workers=workers, reps=reps, verbose=True, join=join)
    os.makedirs(out_dir, exist_ok=True)
    tag = '' if members == 'asof' else '_' + members
    json.dump(ev, open(os.path.join(out_dir, f'search{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, f'search_report{tag}.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
