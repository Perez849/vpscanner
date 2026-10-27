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
# nombre, stop inicial (ATR), asegurar (% a favor, None = no asegura), persigue (ATR del mejor cierre), sesiones máximas
EXITS = (('op4a', 1.5, 4.0, 3.0, 60), ('op4b', 2.5, 4.0, 3.0, 60), ('op6a', 1.5, 6.0, 4.0, 60), ('op6b', 2.5, 6.0, 4.0, 60), ('run5', 5.0, None, 5.0, 60), ('h10', None, None, None, 10))
EXIT_TXT = {'op4a': 'stop 1,5 ATR · asegura +4 % · persigue 3 ATR', 'op4b': 'stop 2,5 ATR · asegura +4 % · persigue 3 ATR', 'op6a': 'stop 1,5 ATR · asegura +6 % · persigue 4 ATR',
            'op6b': 'stop 2,5 ATR · asegura +6 % · persigue 4 ATR', 'run5': 'stop que persigue 5 ATR sin asegurar', 'h10': 'cierre a 10 sesiones sin stop'}
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
def sim_all(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray, d: int, stop_atr: float, lock_pct: Optional[float], trail: float, maxb: int, cost: float = COST) -> np.ndarray:
    """Resultado neto (%) de entrar en la apertura siguiente a CADA barra, con la regla del operador: stop inicial a `stop_atr` ATR de la entrada; si `lock_pct` no es None, al cerrar
    una sesión a ese % a favor el stop sube a ese nivel y desde entonces persigue al mejor cierre a `trail` ATR (nunca por debajo de lo asegurado); si es None, el stop persigue
    al mejor cierre desde el principio. El stop gana si se toca el mismo día y un hueco más allá sale a la apertura. Es la misma regla que plan.outcome_lock, vectorizada por entradas.
    NaN donde la operación no ha terminado todavía (últimas barras) o no hay ATR."""
    n = len(c)
    m = n - 1
    net = np.full(n, np.nan)
    if m < maxb + 2:
        return net
    entry, a = o[1:], atr[:m]
    done = ~(np.isfinite(a) & np.isfinite(entry) & (a > 0))
    st = entry - d * stop_atr * a
    lock = entry * (1.0 + d * lock_pct / 100.0) if lock_pct is not None else None
    best = entry.copy()
    locked = np.zeros(m, bool)
    xp = np.full(m, np.nan)
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
        done |= sold
        upd = live & ~hit
        best = np.where(upd, np.maximum(best, cj) if d > 0 else np.minimum(best, cj), best)
        if lock is not None:
            locked |= upd & ((cj >= lock) if d > 0 else (cj <= lock))
            new = np.maximum(lock, best - trail * a) if d > 0 else np.minimum(lock, best + trail * a)
            st = np.where(upd & locked, new, st)
        else:
            new = np.maximum(st, best - trail * a) if d > 0 else np.minimum(st, best + trail * a)
            st = np.where(upd, new, st)
    j = idx + maxb                                                         # última sesión de la ventana: cierre por tiempo
    tm = ~done & (j <= n - 1)
    xp[tm] = c[np.minimum(j, n - 1)][tm]
    net[:m] = d * (xp / entry - 1.0) * 100.0 - cost
    return net


def hold_all(o: np.ndarray, c: np.ndarray, d: int, hz: int, cost: float = COST) -> np.ndarray:
    n = len(c)
    net = np.full(n, np.nan)
    if n > hz + 1:
        mm = n - hz
        net[:mm] = d * (c[hz:hz + mm] / o[1:1 + mm] - 1.0) * 100.0 - cost
    return net


def ops_all(F: Dict[str, np.ndarray], cost: float = COST) -> np.ndarray:
    """(barras, 2, salidas): resultado neto de entrar a cada barra en largo (0) y en corto (1) con cada salida de EXITS."""
    o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
    out = np.full((len(c), 2, NE), np.nan, dtype=np.float32)
    for di, d in enumerate((1, -1)):
        for ei, (name, stop_atr, lock, trail, maxb) in enumerate(EXITS):
            out[:, di, ei] = hold_all(o, c, d, maxb, cost) if stop_atr is None else sim_all(o, h, l, c, atr, d, stop_atr, lock, trail, maxb, cost)
    return out


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
    w['ops'] = ops_all(ta.prepare(b), rest[0] if rest else COST)
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
        for j, s in enumerate(D.syms):
            ops[off[j]:off[j] + cnt[j]] = work[s]['ops'][D.bar[off[j]:off[j] + cnt[j]]]
        self.net = ops                                                       # absoluto: (filas, sentido, salida)
        self.nd, self.nsym = D.nd, len(D.syms)
        self.blk = D.dcode // BLOCK
        self.nb = int(self.blk.max()) + 1
        # exceso sobre el día; matriz (día, valor, sentido, salida) para poder desplazar el calendario en el nulo
        R = np.full((self.nd, self.nsym, 2, NE), np.nan, dtype=np.float32)
        ok = np.isfinite(ops)
        for di in range(2):
            for ei in range(NE):
                v = ok[:, di, ei]
                sd = np.bincount(D.dcode[v], weights=ops[v, di, ei], minlength=self.nd)
                cd = np.bincount(D.dcode[v], minlength=self.nd)
                mean = np.where(cd >= min_day, sd / np.maximum(cd, 1), np.nan)
                R[D.dcode[v], D.sym[v], di, ei] = ops[v, di, ei] - mean[D.dcode[v]]
        self.R = R
        self.rel = R[D.dcode, D.sym]                                         # (filas, 2, salidas), NaN si ese día no hay referencia
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
        self.G = [block_groups(self.nb, 1 if e[4] <= BLOCK else LONG_BLOCKS) for e in EXITS]            # grupos de cada salida
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

    def tables(self, rel: np.ndarray, absolute: bool = False) -> Dict[str, np.ndarray]:
        """Por combinación, sentido, salida y bloque: suma del exceso (S) y número de operaciones (N); y con `absolute`, suma del neto (SA) y de las ganadoras (SW)."""
        C = len(self.combos)
        S = np.zeros((C, 2, NE, self.nb))
        N = np.zeros((C, 2, NE, self.nb))
        SA = np.zeros((C, 2, NE, self.nb)) if absolute else None
        SW = np.zeros((C, 2, NE, self.nb)) if absolute else None
        for ci, ix in enumerate(self.idx):
            b = self.blk[ix]
            x = rel[ix]
            ok = np.isfinite(x)
            xv = np.where(ok, x, 0.0)
            if absolute:
                a = self.net[ix]
                oka = np.isfinite(a)
                av = np.where(oka, a, 0.0)
            for di in range(2):
                for ei in range(NE):
                    S[ci, di, ei] = np.bincount(b, weights=xv[:, di, ei], minlength=self.nb)
                    N[ci, di, ei] = np.bincount(b, weights=ok[:, di, ei], minlength=self.nb)
                    if absolute:
                        SA[ci, di, ei] = np.bincount(b, weights=av[:, di, ei], minlength=self.nb)
                        SW[ci, di, ei] = np.bincount(b, weights=(a[:, di, ei] > 0) & oka[:, di, ei], minlength=self.nb)
        out = {'S': S, 'N': N}
        if absolute:
            out.update({'SA': SA, 'SW': SW})
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


def stats_by_exit(P: Panel, T: Dict[str, np.ndarray], tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """n, media, t de exceso y cobertura de todas las combinaciones (regla, sentido, salida) en los bloques `tr`, con los grupos de error de cada salida.
    Cobertura = fracción de los periodos (grupos) del universo en esos bloques en los que la regla tiene alguna operación; -1 en t si no hay suficientes periodos."""
    C = T['S'].shape[0]
    n, m, t, cov = (np.zeros((C, 2, NE)) for _ in range(4))
    for e in range(NE):
        Ne = T['N'][:, :, e, :][..., tr]
        G = P.G[e][tr]
        n[:, :, e], m[:, :, e], t[:, :, e] = t_blocks(T['S'][:, :, e, :][..., tr], Ne, G)
        tot = max(int(((T['N'][0, 0, e, :][tr] @ G) > 0).sum()), 1)              # periodos con alguna fila del universo (la combinación 0 = todas las filas)
        has = ((Ne @ G) > 0).sum(-1)
        cov[:, :, e] = has / tot
        t[:, :, e] = np.where(has >= MIN_GROUPS, t[:, :, e], np.nan)
    return n, m, t, cov


def pick(P: Panel, T: Dict[str, np.ndarray], sub: np.ndarray, tr: np.ndarray, min_n: int = MIN_TRAIN_N) -> Optional[Tuple[int, int, int, float, float, float]]:
    """Combinación (regla, sentido, salida) de mayor t de exceso con exceso positivo en los bloques `tr`; solo entre las reglas de `sub`."""
    n, m, t, cov = stats_by_exit(P, T, tr)
    t = np.where((n >= min_n) & (m > 0) & (cov >= MIN_COVER) & np.isfinite(t), t, -np.inf)
    t[~sub] = -np.inf
    k = int(np.argmax(t))
    if not np.isfinite(t.flat[k]):
        return None
    c, d, e = np.unravel_index(k, t.shape)
    return int(c), int(d), int(e), float(n[c, d, e]), float(m[c, d, e]), float(t[c, d, e])


def walk_forward(P: Panel, T: Dict[str, np.ndarray], sub: np.ndarray, test_years: List[int], absolute: bool = False) -> Dict:
    """Para cada año de prueba se elige con los datos anteriores (menos PURGE sesiones) y se evalúa en el año; se acumulan los bloques del año."""
    nb = P.nb
    So, No = np.zeros(nb), np.zeros(nb)
    SAo, SWo, SAb, SWb, Nb = np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)
    rows = []
    for y in test_years:
        d0 = P.ystart.get(y)
        if d0 is None:
            continue
        tr = ((np.arange(nb) + 1) * BLOCK <= d0 - PURGE)
        te = P.byear == y
        if tr.sum() < 20 or not te.any():
            continue
        p = pick(P, T, sub, tr)
        if p is None:
            continue
        c, d, e, ntr, mtr, ttr = p
        So[te] += T['S'][c, d, e, te]
        No[te] += T['N'][c, d, e, te]
        row = {'año': int(y), 'regla': list(P.combos[c]), 'sentido': DIRS[d], 'salida': EXITS[e][0], 'train_n': int(ntr), 'train_exceso': round(mtr, 3), 'train_t': round(ttr, 2)}
        n_, m_, t_ = t_blocks(T['S'][c, d, e, te], T['N'][c, d, e, te], P.G[e][te])
        row.update({'n': int(n_), 'exceso': round(float(m_), 3) if n_ else None})
        if absolute:
            SAo[te] += T['SA'][c, d, e, te]
            SWo[te] += T['SW'][c, d, e, te]
            SAb[te] += T['SA'][0, d, e, te]
            SWb[te] += T['SW'][0, d, e, te]
            Nb[te] += T['N'][0, d, e, te]
            row['neto'] = round(float(T['SA'][c, d, e, te].sum() / max(n_, 1)), 3)
            row['ganan'] = round(float(100 * T['SW'][c, d, e, te].sum() / max(n_, 1)), 1)
            row['azar_neto'] = round(float(T['SA'][0, d, e, te].sum() / max(T['N'][0, d, e, te].sum(), 1)), 3)
            row['azar_ganan'] = round(float(100 * T['SW'][0, d, e, te].sum() / max(T['N'][0, d, e, te].sum(), 1)), 1)
        rows.append(row)
    n, m, t = t_blocks(So[None, :], No[None, :], P.Gpool)
    res = {'años': rows, 'n': int(n[0]), 'exceso': float(m[0]) if n[0] else None, 't': float(t[0]) if n[0] else None}
    if absolute and n[0]:
        res['neto'] = float(SAo.sum() / n[0])
        res['ganan'] = float(100 * SWo.sum() / n[0])
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
        out.append({'regla': list(P.combos[c]), 'sentido': DIRS[d], 'salida': EXITS[e][0], 'train_n': int(n[c, d, e]), 'train_exceso': round(float(m[c, d, e]), 3), 'train_t': round(float(t[c, d, e]), 2),
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
        out.append({'regla': list(P.combos[c]), 'sentido': DIRS[d], 'salida': EXITS[e][0], 'n': int(n[c, d, e]), 'exceso': round(float(m[c, d, e]), 3), 't': round(float(t[c, d, e]), 2),
                    'neto': round(float(T['SA'][c, d, e].sum() / n[c, d, e]), 3) if 'SA' in T else None, 'ganan': round(float(100 * T['SW'][c, d, e].sum() / n[c, d, e]), 1) if 'SW' in T else None,
                    'por_año': per})
    return out


def null_runs(P: Panel, test_years: List[int], reps: int, seed: int = 13, verbose: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """Repite todo el procedimiento con los resultados desplazados en el calendario (mismo desplazamiento para todos los valores)."""
    rng = np.random.default_rng(seed)
    out = {s: {'exceso': [], 't': []} for s in SUBSETS}
    t0 = time.time()
    for q in range(reps):
        sh = int(rng.integers(MIN_SHIFT, max(P.nd - MIN_SHIFT, MIN_SHIFT + 1)))
        rel = P.R[(P.D.dcode + sh) % P.nd, P.D.sym]
        T = P.tables(rel)
        for s in SUBSETS:
            r = walk_forward(P, T, P.sub[s], test_years)
            out[s]['exceso'].append(r['exceso'] if r['exceso'] is not None else np.nan)
            out[s]['t'].append(r['t'] if r['t'] is not None else np.nan)
        if verbose and (q + 1) % 5 == 0:
            print(f'    nulo {q + 1}/{reps} · {time.time() - t0:.0f}s', flush=True)
    return {s: {k: np.array(v) for k, v in d.items()} for s, d in out.items()}


# ── conjunto ──────────────────────────────────────────────────────────────
def build(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, verbose: bool = False, min_day: int = MIN_DAY, min_combo: int = MIN_COMBO, join: Optional[Dict[str, str]] = None,
          cost: float = COST, library: bool = True) -> Tuple[ml.Data, Panel, int]:
    """Descarga ya hecha → trabajo por valor (indicadores, patrones, salidas) → datos del universo → panel. Devuelve (datos, panel, barras-valor quitadas por la incorporación al índice)."""
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
    T = P.tables(P.rel, absolute=True)
    wf = {s: walk_forward(P, T, P.sub[s], test_years, absolute=True) for s in SUBSETS}
    st = {s: static_split(P, T, P.sub[s], dev_end) for s in SUBSETS}
    ins = {s: in_sample_best(P, T, P.sub[s]) for s in SUBSETS}
    if verbose:
        print(f'  selección hacia delante hecha · {time.time() - t0:.0f}s', flush=True)
    null = null_runs(P, test_years, reps, verbose=verbose) if reps else {}
    nl = {}
    for s in SUBSETS:
        if s not in null:
            continue
        ex, tt = null[s]['exceso'], null[s]['t']
        ex, tt = ex[np.isfinite(ex)], tt[np.isfinite(tt)]
        real = wf[s]
        nl[s] = {'reps': int(len(ex)), 'exceso_p50': round(float(np.median(ex)), 4), 'exceso_p95': round(float(np.quantile(ex, 0.95)), 4), 'exceso_max': round(float(ex.max()), 4),
                 't_p50': round(float(np.median(tt)), 2), 't_p95': round(float(np.quantile(tt, 0.95)), 2), 't_max': round(float(tt.max()), 2),
                 'p_exceso': round(float((ex >= real['exceso']).mean()), 3) if real['exceso'] is not None else None, 'p_t': round(float((tt >= real['t']).mean()), 3) if real['t'] is not None else None}
    # referencia: entrar al azar con cada salida (neto absoluto) en todas las filas
    base = {}
    for di in range(2):
        for ei in range(NE):
            n = T['N'][0, di, ei].sum()
            base[f'{DIRS[di]} {EXITS[ei][0]}'] = {'n': int(n), 'neto': round(float(T['SA'][0, di, ei].sum() / n), 3), 'ganan': round(float(100 * T['SW'][0, di, ei].sum() / n), 1)}
    out = {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(D.sym)), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k, 'combinaciones': len(P.combos) - 1,
                    'pruebas': (len(P.combos) - 1) * 2 * NE, 'condiciones': len(P.cond_names), 'test_years': test_years, 'dev_end': dev_end, 'reps': reps, 'salidas': {e[0]: EXIT_TXT[e[0]] for e in EXITS},
                    'miembros': 'en la fecha' if join else 'actuales', 'valores_con_fecha': int(sum(1 for s_ in work_syms if join and join.get(s_))), 'barras_quitadas': dropped, 'coste': cost},
           'walk_forward': wf, 'nulo': nl, 'dev_final': st, 'en_muestra': ins, 'azar_absoluto': base}
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
    for s, titulo in (('todo', 'TODAS LAS REGLAS'), ('contexto', 'SOLO CONTEXTO (sin patrones)'), ('patrones', 'CON PATRÓN (al menos una condición de patrón)')):
        w, nl = ev['walk_forward'][s], ev['nulo'].get(s)
        L.append(f'=== {titulo} · selección hacia delante, años {m["test_years"][0]}–{m["test_years"][-1]} ===')
        if w['exceso'] is None:
            L += ['  sin resultados', '']
            continue
        L.append(f"  FUERA DE MUESTRA: {w['n']:,} operaciones · exceso {w['exceso']:+.3f} % por operación (t {w['t']:+.1f}) · neto {w['neto']:+.3f} % (azar en las mismas fechas {w['neto_azar_mismas_fechas']:+.3f} %) · ganan {w['ganan']:.1f} % (todas las filas {w['ganan_todas_las_filas']:.1f} %)")
        if nl:
            L.append(f"  Nulo del procedimiento: exceso mediano {nl['exceso_p50']:+.3f} %, p95 {nl['exceso_p95']:+.3f} %, máximo {nl['exceso_max']:+.3f} %; t p95 {nl['t_p95']:+.1f} · p-valor del exceso {nl['p_exceso']} y de la t {nl['p_t']}")
        L.append(f"  {'año':>5s} {'n':>8s} {'exceso':>8s} {'neto':>8s} {'azar':>8s} {'ganan':>6s}  regla elegida con el pasado (t de entrenamiento)")
        for r in w['años']:
            L.append(f"  {r['año']:5d} {r['n']:8,} {r['exceso'] if r['exceso'] is not None else float('nan'):+8.3f} {r.get('neto', float('nan')):+8.3f} {r.get('azar_neto', float('nan')):+8.3f} {r.get('ganan', float('nan')):6.1f}  "
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
        L.append(f"  {k:12s} n {v['n']:9,}  neto {v['neto']:+.3f} %  ganan {v['ganan']:.1f} %")
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
)


def evaluate_rules(P: Panel, rules: Tuple[Dict, ...], reps: int = 300, seed: int = 21) -> List[Dict]:
    """Exceso, t agrupado, neto, % ganadoras, años positivos y p-valores (resultados desplazados en el calendario) de reglas FIJAS: una sola hipótesis por regla, sin elegir nada."""
    D = P.D
    cond = conditions(D)
    exn = [e[0] for e in EXITS]
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
        ab = P.net[ix, d, e]
        okb = np.isfinite(ab)
        allnet = P.net[:, d, e]
        yr = {}
        for y in sorted(int(v) for v in np.unique(D.year[ix])):
            xs = P.rel[ix, d, e][(D.year[ix] == y)]
            xs = xs[np.isfinite(xs)]
            if len(xs) >= 100:
                yr[y] = round(float(xs.mean()), 3)
        out.append({**r, 'n': int(n), 'valores': int(len(np.unique(D.sym[ix]))), 'exceso': round(mean, 3), 't': round(t, 2), 'neto': round(float(ab[okb].mean()), 3), 'azar_neto': round(float(ab[okb].mean() - mean), 3),
                    'ganan': round(float(100 * (ab[okb] > 0).mean()), 1), 'ganan_todas': round(float(100 * (allnet[np.isfinite(allnet)] > 0).mean()), 1), 'por_año': yr,
                    'nulo_exceso_p95': round(float(np.quantile(nul[:, 1], 0.95)), 3), 'nulo_t_p95': round(float(np.nanquantile(nul[:, 2], 0.95)), 2),
                    'p_exceso': round(float((nul[:, 1] >= mean).mean()), 3), 'p_t': round(float((nul[np.isfinite(nul[:, 2]), 2] >= t).mean()), 3), 'reps': int(reps)})
    return out


def confirm(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, cost: float = COST, reps: int = 300, join: Optional[Dict[str, str]] = None, verbose: bool = False,
            rules: Tuple[Dict, ...] = CANDIDATAS, min_day: int = MIN_DAY) -> Dict:
    D, P, dropped = build(bars, workers, k, verbose, min_day, MIN_COMBO, join, cost, library=False)
    res = evaluate_rules(P, rules, reps)
    return {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(D.sym)), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k, 'coste': cost, 'reps': reps,
                     'miembros': 'en la fecha' if join else 'actuales', 'barras_quitadas': dropped, 'salidas': {e[0]: EXIT_TXT[e[0]] for e in EXITS}}, 'reglas': res}


def confirm_report(ev: Dict, titulo: str = '') -> str:
    m = ev['meta']
    L = [f"CONFIRMACIÓN DE LAS REGLAS CANDIDATAS{' · ' + titulo if titulo else ''} · {m['universe']} acciones · {m['filas']:,} filas (las {m['k']} más negociadas) · {m['from']} → {m['to']} · coste {m['coste']} %",
         f'Reglas fijadas antes de mirar estas acciones (una hipótesis cada una, no se elige nada). Exceso = neto menos la media de todas las filas de ese día. p = fracción de repeticiones con los resultados desplazados en el calendario (al menos {MIN_SHIFT} sesiones) que igualan o superan lo observado.',
         f"Universo: {'miembros actuales' if m['miembros'] == 'actuales' else 'miembros actuales desde su incorporación al S&P 500'}. Salidas: " + ' · '.join(f'{k} = {v}' for k, v in m['salidas'].items()), '']
    for r in ev['reglas']:
        ys = ' '.join(f"{str(y)[2:]}:{v:+.1f}" for y, v in r['por_año'].items())
        pos = sum(v > 0 for v in r['por_año'].values())
        L += [f"{r['nombre']} · {_rule(r['regla'], r['sentido'])} · {r['salida']}",
              f"   n {r['n']:,} ({r['valores']} valores) · exceso {r['exceso']:+.3f} % (t {r['t']:+.1f}; nulo p95 {r['nulo_exceso_p95']:+.3f} % / t {r['nulo_t_p95']:+.1f}; p exceso {r['p_exceso']}, p t {r['p_t']}) · neto {r['neto']:+.3f} % (azar mismas fechas {r['azar_neto']:+.3f} %) · ganan {r['ganan']:.1f} % (todas las filas {r['ganan_todas']:.1f} %)",
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
