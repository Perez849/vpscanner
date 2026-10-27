"""
Estudio de evidencia: ¿baten los patrones a entrar al azar?

  · Universo móvil: las K acciones más negociadas en cada fecha (dinero medio de 60 sesiones con datos hasta ese día), K = 50 y 300.
  · Cada señal se opera en la apertura siguiente y se cierra a 5, 10 (principal) y 20 sesiones; los patrones bajistas se miden en corto. Coste de ida y vuelta del 0,15 %.
  · Referencia: entrar al azar en el mismo universo EL MISMO DÍA («exceso» = rendimiento del patrón − media de todas las acciones del universo ese día, en el mismo sentido): así el momento
    del mercado (un patrón que salta justo antes de un rebote general) no cuenta como ventaja. También se guarda el exceso frente a la media del año (`exy`, `ty`).
  · t con errores estándar agrupados en BLOQUES de 10 sesiones (antes, por día): los rendimientos a 10 sesiones de días contiguos se solapan y las señales de un mismo valor o de un mismo
    régimen de mercado vienen en rachas. Con señales colocadas al azar ambas agrupaciones salen igual de bien calibradas (simulación: 4,5–6 % de |t| > 1,96); el bloque protege cuando las señales se pegan en el tiempo.
  · Umbral del mejor de N: se desplaza en el tiempo, al azar, las señales de cada valor (conservando su agrupación) y se anota el mayor |t| entre TODOS los patrones;
    un patrón solo «supera el azar» si su |t| pasa el percentil 95 de ese máximo. Se estudian también las señales de calidad alta (≥ 75) de cada figura.
  · Limitación: el universo son los miembros ACTUALES del S&P 500 y el Nasdaq-100 (sesgo de supervivencia que favorece a los largos).
"""
from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import engine, ta
from .catalog import CATALOG, HQ

KS = (50, 300)
HORIZONS = (5, 10, 20)
PRIMARY = 10
COST = 0.15
MEM_WINDOW = 60
MIN_EVENTS = 20
PATH = 20                  # sesiones de la trayectoria media que se dibuja tras cada patrón
BLOCK = 10                 # sesiones por bloque en los errores estándar agrupados (= el horizonte principal: los rendimientos de dentro de un bloque se solapan y los de bloques distintos, no)
DAY_SHIFT = 5 * 3600       # una barra pertenece al día en que cierra el mercado: 14:30 UTC (bolsa) y 00:00 UTC (cripto) caen el mismo día; el cierre de las divisas (22:00–23:00 UTC) cae al siguiente


def _work(args: Tuple[str, Dict[str, np.ndarray]]) -> Tuple[str, Dict]:
    """Por valor: retornos futuros por horizonte, dinero negociado y los índices de barra de cada patrón (y de su variante de calidad alta)."""
    sym, b = args
    F = ta.prepare(b)
    events, _ = engine.scan(F)
    return sym, work_from(F, events)


def work_from(F: Dict[str, np.ndarray], events: List[Dict], paths: bool = True) -> Dict:
    """Lo mismo que _work pero con los indicadores y los eventos ya calculados (el estudio del plan reutiliza el escaneo)."""
    n = len(F['c'])
    o, c = F['o'], F['c']
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    ret = {}
    for hz in HORIZONS:
        ex = np.full(n, np.nan)
        if n > hz:
            ex[:n - hz] = c[hz:]
        ret[hz] = (ex / entry - 1.0) * 100.0
    dv = np.full(n, np.nan)
    if n >= MEM_WINDOW:
        cs = np.cumsum(np.insert(c * F['v'], 0, 0.0))
        dv[MEM_WINDOW - 1:] = (cs[MEM_WINDOW:] - cs[:-MEM_WINDOW]) / MEM_WINDOW
    with np.errstate(invalid='ignore', divide='ignore'):
        r120 = (c / ta.shift(c, 120) - 1.0) * 100.0
    feat = {'r120': r120.astype(np.float32), 'rsi2': F['rsi2'].astype(np.float32), 'atrp': F['atrp'].astype(np.float32), 'a200': (c > F['sma200']).astype(np.float32)}
    idx: Dict[str, List[int]] = {}
    for e in events:
        if e['i'] < 60:
            continue
        idx.setdefault(e['name'], []).append(e['i'])
        if e['q'] is not None and e['q'] >= HQ:
            idx.setdefault(e['name'] + ':hq', []).append(e['i'])
    out = {'t': ((F['t'] + DAY_SHIFT) // 86400).astype(np.int64), 'ret': ret, 'dv': dv, 'feat': feat, 'idx': {k: np.array(v, dtype=np.int64) for k, v in idx.items()}}
    if paths:                                                   # rendimiento acumulado de 1 a PATH sesiones desde la apertura siguiente (para dibujar qué pasa después)
        pa = np.full((n, PATH), np.nan, dtype=np.float32)
        for k in range(1, PATH + 1):
            if n > k + 1:
                pa[:n - k, k - 1] = (c[k:] / entry[:n - k] - 1.0) * 100.0
        out['path'] = pa
    return out


class Panel:
    """Todas las barras de todos los valores en un solo vector, con retornos futuros, pertenencia al universo móvil y señales por patrón."""

    def __init__(self, work: Dict[str, Dict], ks: Tuple[int, ...] = KS, cost: float = COST):
        self.cost = cost
        self.syms = list(work)
        lens = np.array([len(work[s]['t']) for s in self.syms])
        self.off = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        self.nbar = lens.astype(np.int64)
        self.symv = np.repeat(np.arange(len(self.syms)), lens)
        self.dayv = np.concatenate([work[s]['t'] for s in self.syms])
        days = np.unique(self.dayv)
        self.dcode = np.searchsorted(days, self.dayv)
        self.ndays = len(days)
        self.year = (days.astype('datetime64[D]').astype('datetime64[Y]').astype(int) + 1970)[self.dcode]
        self.y0 = int(self.year.min())
        # pertenencia: rango por dinero negociado cada día
        dvm = np.full((len(days), len(self.syms)), np.nan, dtype=np.float32)
        for j, s in enumerate(self.syms):
            dvm[np.searchsorted(days, work[s]['t']), j] = work[s]['dv']
        rank = np.argsort(np.argsort(-np.nan_to_num(dvm, nan=-1.0), axis=1), axis=1)
        okd = np.isfinite(dvm)
        self.mem = {K: np.concatenate([(rank[np.searchsorted(days, work[s]['t']), j] < K) & okd[np.searchsorted(days, work[s]['t']), j] for j, s in enumerate(self.syms)]) for K in ks}
        self.min_day = {K: max(3, min(10, K // 3)) for K in ks}          # mínimo de valores del universo ese día para que exista referencia
        self.ret = {}
        for hz in HORIZONS:
            r = np.concatenate([work[s]['ret'][hz] for s in self.syms])
            self.ret[('long', hz)] = r - cost
            self.ret[('short', hz)] = -r - cost
        self.path = np.vstack([work[s]['path'] for s in self.syms]) if all('path' in work[s] for s in self.syms) else None
        self.feat = {k: np.concatenate([work[s]['feat'][k] for s in self.syms]) for k in ('r120', 'rsi2', 'atrp', 'a200')} if all('feat' in work[s] for s in self.syms) else None
        self.ev: Dict[str, np.ndarray] = {}
        for j, s in enumerate(self.syms):
            for k, ix in work[s]['idx'].items():
                self.ev.setdefault(k, []).append(ix + self.off[j])                           # type: ignore[arg-type]
        self.ev = {k: np.concatenate(v) for k, v in self.ev.items()}
        ny = int(self.year.max()) - self.y0 + 1
        self.pbase: Dict[int, np.ndarray] = {}                      # trayectoria media de las barras del universo móvil por día
        if self.path is not None:
            for K in ks:
                pb = np.full((self.ndays, PATH), np.nan)
                m = self.mem[K]
                for k in range(PATH):
                    f = m & np.isfinite(self.path[:, k])
                    sd = np.bincount(self.dcode[f], weights=self.path[f, k], minlength=self.ndays)
                    cd = np.bincount(self.dcode[f], minlength=self.ndays)
                    pb[:, k] = np.where(cd >= self.min_day[K], sd / np.maximum(cd, 1), np.nan)
                self.pbase[K] = pb
        self.base: Dict[Tuple, np.ndarray] = {}                     # media de entrar al azar por año (referencia secundaria)
        self.bday: Dict[Tuple, np.ndarray] = {}                     # media de entrar al azar por día (referencia principal; NaN si ese día hay pocos valores en el universo)
        for K in ks:
            for key, r in self.ret.items():
                ok = self.mem[K] & np.isfinite(r)
                s_ = np.bincount(self.year[ok] - self.y0, weights=r[ok], minlength=ny)
                c_ = np.bincount(self.year[ok] - self.y0, minlength=ny)
                self.base[(K,) + key] = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
                sd = np.bincount(self.dcode[ok], weights=r[ok], minlength=self.ndays)
                cd = np.bincount(self.dcode[ok], minlength=self.ndays)
                self.bday[(K,) + key] = np.where(cd >= self.min_day[K], sd / np.maximum(cd, 1), np.nan)

    def blocks(self, rows: np.ndarray, block: int = BLOCK) -> np.ndarray:
        """Bloque de `block` sesiones al que pertenece cada fila del panel (los códigos de día son consecutivos entre sesiones de mercado)."""
        return self.dcode[rows] // block

    def t_cluster(self, ex: np.ndarray, code: np.ndarray) -> float:
        """t de la media con errores estándar agrupados por el código que se pase (bloques de sesiones, ver `blocks`)."""
        n = len(ex)
        if n < 5:
            return float('nan')
        m = ex.mean()
        s = np.bincount(code, weights=ex, minlength=self.ndays)
        k = np.bincount(code, minlength=self.ndays)
        se = np.sqrt(((s - m * k) ** 2).sum()) / n
        return float(m / se) if se > 0 else float('nan')

    def stat(self, idx: np.ndarray, K: int, side: str, hz: int) -> Dict:
        r = self.ret[(side, hz)][idx]
        bd = self.bday[(K, side, hz)][self.dcode[idx]]
        ok = self.mem[K][idx] & np.isfinite(r) & np.isfinite(bd)
        if ok.sum() < 5:
            return {'n': int(ok.sum())}
        i2, x = idx[ok], r[ok]
        ex = x - bd[ok]
        exy = x - self.base[(K, side, hz)][self.year[i2] - self.y0]
        yrs = self.year[i2]
        bl = self.blocks(i2, max(BLOCK, hz))
        pos = tot = 0
        for Y in np.unique(yrs):
            m = yrs == Y
            if m.sum() >= 8:
                tot += 1
                pos += int(ex[m].mean() > 0)
        return {'n': int(ok.sum()), 'wr': float((x > 0).mean() * 100), 'mean': float(x.mean()), 'med': float(np.median(x)), 'ex': float(ex.mean()), 't': self.t_cluster(ex, bl), 'se': self.se_cluster(ex, bl),
                'exy': float(exy.mean()), 'ty': self.t_cluster(exy, bl), 'pos': pos, 'tot': tot}

    def se_cluster(self, ex: np.ndarray, code: np.ndarray) -> float:
        """Error estándar de la media con la agrupación que se pase (el denominador de t_cluster)."""
        n = len(ex)
        if n < 5:
            return float('nan')
        m = ex.mean()
        s = np.bincount(code, weights=ex, minlength=self.ndays)
        k = np.bincount(code, minlength=self.ndays)
        return float(np.sqrt(((s - m * k) ** 2).sum()) / n)

    def path_stat(self, idx: np.ndarray, K: int, side: str) -> Optional[Dict]:
        """Trayectoria media (sesiones 1..PATH, %) del lado operado menos la del universo ese mismo día, con su error estándar agrupado por bloques de PATH sesiones."""
        if self.path is None or K not in self.pbase:
            return None
        sg = 1.0 if side == 'long' else -1.0
        ok = self.mem[K][idx]
        if ok.sum() < 5:
            return None
        i2 = idx[ok]
        ex = sg * (self.path[i2] - self.pbase[K][self.dcode[i2]])
        bl = self.blocks(i2, PATH)
        mean, se = np.full(PATH, np.nan), np.full(PATH, np.nan)
        for k in range(PATH):
            f = np.isfinite(ex[:, k])
            if f.sum() >= 5:
                mean[k] = ex[f, k].mean()
                se[k] = self.se_cluster(ex[f, k], bl[f])
        return {'m': [round(float(x), 3) for x in mean], 'se': [round(float(x), 3) for x in se], 'n': int(ok.sum())}

    def null_max_t(self, K: int, reps: int, seed: int = 5) -> np.ndarray:
        """Distribución del mayor |t| entre todos los patrones cuando las señales de cada valor se desplazan en el tiempo (circularmente y al azar)."""
        rng = np.random.default_rng(seed)
        evs = []
        for k, v in self.ev.items():
            side = 'long' if CATALOG[k.split(':')[0]].dir > 0 else 'short'
            evs.append((side, v, self.symv[v]))
        out = np.empty(reps)
        for q in range(reps):
            sh = np.array([rng.integers(30, max(nb - 30, 31)) for nb in self.nbar])
            best = 0.0
            for side, v, sv in evs:
                if len(v) < MIN_EVENTS:
                    continue
                o, nb = self.off[sv], self.nbar[sv]
                j = o + ((v - o + sh[sv]) % nb)
                r = self.ret[(side, PRIMARY)][j]
                bd = self.bday[(K, side, PRIMARY)][self.dcode[j]]
                ok = self.mem[K][j] & np.isfinite(r) & np.isfinite(bd)
                if ok.sum() < MIN_EVENTS:
                    continue
                ex = r[ok] - bd[ok]
                t = abs(self.t_cluster(ex, self.blocks(j[ok])))
                if t > best:
                    best = t
            out[q] = best
        return out


def combos(P: 'Panel', ks: Tuple[int, ...]) -> Dict:
    """Qué pasa cuando varios patrones saltan a la vez en el mismo valor y la misma sesión. Cada «valor-día» se cuenta UNA vez, no una por patrón.
    Clases (solo valor-días con alguna señal del sentido analizado y ninguna contraria): solo velas, solo estructura (figuras, rupturas, directrices, rebotes) o ambas juntas;
    y los valor-días contradictorios (señales alcistas y bajistas a la vez), que se miden en largo y en corto. Los patrones de consenso no cuentan (son las propias velas)."""
    n = len(P.dayv)
    cnt = {(d, c): np.zeros(n, dtype=np.int16) for d in (1, -1) for c in ('v', 'e')}
    for k, rows in P.ev.items():
        if k.endswith(':hq') or k.startswith('consenso'):
            continue
        pat = CATALOG[k]
        if pat.dir == 0:
            continue
        cnt[(pat.dir, 'v' if pat.fam == 'velas' else 'e')][rows] += 1
    up_v, up_e, dn_v, dn_e = cnt[(1, 'v')] > 0, cnt[(1, 'e')] > 0, cnt[(-1, 'v')] > 0, cnt[(-1, 'e')] > 0
    up, dn = up_v | up_e, dn_v | dn_e
    classes = {
        'alcistas': {'solo velas': up_v & ~up_e & ~dn, 'solo estructura': up_e & ~up_v & ~dn, 'velas y estructura juntas': up_v & up_e & ~dn, 'todas (cada valor-día una vez)': up & ~dn},
        'bajistas': {'solo velas': dn_v & ~dn_e & ~up, 'solo estructura': dn_e & ~dn_v & ~up, 'velas y estructura juntas': dn_v & dn_e & ~up, 'todas (cada valor-día una vez)': dn & ~up},
    }
    out: Dict = {}
    for K in ks:
        rec: Dict = {}
        for sent, side in (('alcistas', 'long'), ('bajistas', 'short')):
            rec[sent] = {}
            for name, m in classes[sent].items():
                st = P.stat(np.flatnonzero(m), K, side, PRIMARY)
                rec[sent][name] = {a: (round(b, 3) if isinstance(b, float) else b) for a, b in st.items()}
        both = np.flatnonzero(up & dn)
        rec['contradictorias'] = {side: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in P.stat(both, K, side, PRIMARY).items()} for side in ('long', 'short')}
        tot_up, tot_dn = int(up.sum()), int(dn.sum())
        rec['valor_dias'] = {'alcistas': tot_up, 'bajistas': tot_dn, 'contradictorios': int((up & dn).sum()), 'con_varias': int(((cnt[(1, 'v')] + cnt[(1, 'e')] + cnt[(-1, 'v')] + cnt[(-1, 'e')]) > 1).sum())}
        rec['familias'] = {}
        for fam in ('figuras', 'rupturas', 'velas', 'directrices', 'rebotes'):
            for dd, side in ((1, 'long'), (-1, 'short')):
                ks_ = [k for k in P.ev if not k.endswith(':hq') and not k.startswith('consenso') and CATALOG[k].fam == fam and CATALOG[k].dir == dd]
                if not ks_:
                    continue
                rows = np.unique(np.concatenate([P.ev[k] for k in ks_]))
                st = P.stat(rows, K, side, PRIMARY)
                rec['familias'][f'{fam} {"alcistas" if dd > 0 else "bajistas"}'] = {**{a: (round(b, 3) if isinstance(b, float) else b) for a, b in st.items()}, 'patrones': len(ks_)}
        out[f'K{K}'] = rec
    return out


def control_rows(P: 'Panel', K: int) -> Dict[str, np.ndarray]:
    """Filas del panel (ordenadas) de cada señal de control; los percentiles son del mismo día entre los valores del universo móvil."""
    m = P.mem[K]
    sel: Dict[str, np.ndarray] = {}
    for name, col in (('r120', 'r120'), ('atrp', 'atrp')):
        v = P.feat[col]
        rows = np.flatnonzero(m & np.isfinite(v))
        order = np.lexsort((v[rows], P.dcode[rows]))
        rows, dd = rows[order], P.dcode[rows][order]
        cnt = np.bincount(dd, minlength=P.ndays)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        pos = np.arange(len(rows)) - start[dd]
        pct = pos / np.maximum(cnt[dd] - 1, 1)
        ok = cnt[dd] >= P.min_day[K]
        if name == 'r120':
            sel['momentum alto (10 % mejor a 120 sesiones)'] = rows[ok & (pct >= 0.9)]
            sel['momentum bajo (10 % peor a 120 sesiones)'] = rows[ok & (pct <= 0.1)]
        else:
            sel['volatilidad alta (10 % superior de ATR / precio)'] = rows[ok & (pct >= 0.9)]
    sel['sobreventa a corto (RSI 2 < 10 sobre la media de 200)'] = np.flatnonzero(m & (P.feat['rsi2'] < 10) & (P.feat['a200'] > 0.5))
    return {k: np.sort(v) for k, v in sel.items()}


def controls(P: 'Panel', ks: Tuple[int, ...]) -> Dict:
    """Controles positivos con el MISMO motor (referencia del mismo día, t por bloques, mismo umbral): anomalías documentadas de cartera que, si el método funciona, deben aparecer.
    Momentum: el 10 % de valores con mejor / peor rendimiento a 120 sesiones ese día (largos); sobreventa a corto (RSI de 2 sesiones < 10 con el precio sobre su media de 200); volatilidad alta (10 % superior)."""
    if P.feat is None:
        return {}
    return {f'K{K}': {nm: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in P.stat(ix, K, 'long', PRIMARY).items()} for nm, ix in control_rows(P, K).items()} for K in ks}


def analyze_panel(bars: Dict[str, Dict[str, np.ndarray]], ks: Tuple[int, ...] = KS, reps: int = 150, workers: int = 1, verbose: bool = False, cost: float = COST) -> Dict:
    """Barras por valor → estadísticas de evidencia (el contenido de data/evidence.json)."""
    t0 = time.time()
    jobs = [(s, {k: b[k] for k in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if len(b['c']) > 300]
    work: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, (s, w) in enumerate(ex.map(_work, jobs, chunksize=4)):
                work[s] = w
                if verbose and (k + 1) % 100 == 0:
                    print(f'    {k + 1}/{len(jobs)} valores · {time.time() - t0:.0f}s', flush=True)
    else:
        for k, jb in enumerate(jobs):
            s, w = _work(jb)
            work[s] = w
    P = Panel(work, ks, cost)
    out: Dict = {'meta': {'generated': int(time.time()), 'universe': len(P.syms), 'bars': int(len(P.dayv)), 'from': str(np.datetime64(int(P.dayv.min()), 'D')), 'to': str(np.datetime64(int(P.dayv.max()), 'D')),
                          'cost': P.cost, 'primary': PRIMARY, 'horizons': list(HORIZONS), 'ks': list(ks), 'hq': HQ, 'thr': {}, 'tested': {}, 'min_events': MIN_EVENTS, 'block': BLOCK, 'reference': 'day',
                          'note': 'Universo = miembros actuales del S&P 500 y del Nasdaq-100: sesgo de supervivencia que favorece a los largos.'},
                 'baseline': {}, 'patterns': {}, 'paths': {}}
    out['combos'] = combos(P, ks)
    out['controls'] = controls(P, ks)
    for K in ks:
        null = P.null_max_t(K, reps)
        out['meta']['thr'][f'K{K}'] = {'p95': round(float(np.quantile(null, 0.95)), 2), 'p99': round(float(np.quantile(null, 0.99)), 2), 'bars': int(P.mem[K].sum())}
        out['baseline'][f'K{K}'] = {side: {str(P.y0 + i): round(float(v), 3) for i, v in enumerate(P.base[(K, side, PRIMARY)]) if np.isfinite(v)} for side in ('long', 'short')}
    for k, v in sorted(P.ev.items()):
        base = k.split(':')[0]
        pat = CATALOG[base]
        side = 'long' if pat.dir > 0 else 'short'
        rec: Dict = {'dir': pat.dir, 'fam': pat.fam, 'hq': k.endswith(':hq')}
        for K in ks:
            s = P.stat(v, K, side, PRIMARY)
            if s.get('n', 0) >= 1:
                for hz in HORIZONS:
                    if hz != PRIMARY:
                        sh = P.stat(v, K, side, hz)
                        if 'ex' in sh:
                            s[f'h{hz}'] = {'ex': round(sh['ex'], 3), 't': round(sh['t'], 2), 'mean': round(sh['mean'], 3)}
                rec[f'K{K}'] = {a: (round(b, 3) if isinstance(b, float) else b) for a, b in s.items()}
                if not rec['hq']:
                    ps = P.path_stat(v, K, side)
                    if ps:
                        out['paths'].setdefault(k, {})[f'K{K}'] = ps
        out['patterns'][k] = rec
    for K in ks:
        rows = [(k, r[f'K{K}']) for k, r in out['patterns'].items() if f'K{K}' in r and r[f'K{K}'].get('n', 0) >= MIN_EVENTS and 'ex' in r[f'K{K}']]
        thr = out['meta']['thr'][f'K{K}']['p95']
        out['meta']['tested'][f'K{K}'] = {'n': len(rows), 'above': [k for k, s in rows if abs(s['t']) >= thr], 'above_positive': [k for k, s in rows if s['t'] >= thr and s['ex'] > 0]}
    if verbose:
        print(f'  estudio: {len(P.syms)} valores, {len(P.dayv):,} barras, {len(out["patterns"])} patrones · {time.time() - t0:.0f}s', flush=True)
    return out


def report(ev: Dict) -> str:
    """Informe en texto: los patrones de mayor |t| y el resumen por familia."""
    L: List[str] = []
    m = ev['meta']
    L.append(f"ESTUDIO DE EVIDENCIA · {m['universe']} valores · {m['bars']:,} barras · {m['from']} → {m['to']} · coste {m['cost']} % · salida a {m['primary']} sesiones")
    for K in m['ks']:
        thr = m['thr'][f'K{K}']
        t_ = m['tested'][f'K{K}']
        L.append(f"\n=== Los {K} valores más negociados en cada fecha ({thr['bars']:,} barras) · umbral del mejor de {t_['n']}: |t| ≥ {thr['p95']} (p95) / {thr['p99']} (p99, poco preciso) ===")
        rows = [(k, r[f'K{K}']) for k, r in ev['patterns'].items() if f'K{K}' in r and r[f'K{K}'].get('n', 0) >= MIN_EVENTS and 'ex' in r[f'K{K}']]
        rows.sort(key=lambda x: -abs(x[1]['t']))
        L.append('  patrón'.ljust(30) + 'n'.rjust(7) + 'ganan'.rjust(9) + 'media'.rjust(8) + 'exceso'.rjust(8) + 't'.rjust(7) + 'años+'.rjust(7) + 'vs año'.rjust(9) + 't'.rjust(6))
        for k, s in rows[:25]:
            star = '★' if abs(s['t']) >= thr['p95'] else ' '
            L.append(('  ' + ('▲ ' if CATALOG[k.split(':')[0]].dir > 0 else '▼ ') + k + star).ljust(30) + f"{s['n']:7d}{s['wr']:8.1f}%{s['mean']:+7.2f}%{s['ex']:+7.2f}%{s['t']:+7.1f}{s['pos']:4d}/{s['tot']:<2d}{s.get('exy', float('nan')):+8.2f}%{s.get('ty', float('nan')):+6.1f}")
        ses = [s['se'] for k, s in rows if 'se' in s]
        if ses:
            L.append(f"  potencia: con el error estándar medio de estos patrones (mediana {np.median(ses):.2f} %, el de los 10 más raros {np.mean(sorted(ses)[-10:]):.2f} %) el estudio detecta con un 80 % de probabilidad efectos de ≥ {(thr['p95'] + 0.84) * np.median(ses):.2f} % en un patrón típico y de ≥ {(thr['p95'] + 0.84) * np.mean(sorted(ses)[-10:]):.2f} % en los más raros")
        L.append(f"  superan el umbral del mejor de {t_['n']}: {len(t_['above'])} ({', '.join(t_['above']) or 'ninguno'}); por azar se esperarían ≈ {0.05 * t_['n']:.1f} si se miraran sueltos")
        for fam in ('figuras', 'rupturas', 'velas', 'directrices', 'rebotes'):
            for d, dn in ((1, 'alcistas'), (-1, 'bajistas')):
                sel = [s for k, s in rows if ev['patterns'][k]['fam'] == fam and ev['patterns'][k]['dir'] == d and not ev['patterns'][k]['hq']]
                if sel:
                    L.append(f"  {fam:12s} {dn:9s} {len(sel):2d} patrones · exceso medio {np.mean([s['ex'] for s in sel]):+.3f} % · mediano {np.median([s['ex'] for s in sel]):+.3f} % · positivos {np.mean([s['ex'] > 0 for s in sel]) * 100:.0f} %")
        ct = ev.get('controls', {}).get(f'K{K}')
        if ct:
            L.append('  controles positivos (anomalías documentadas medidas con el mismo método y el mismo umbral; si el método funciona deben aparecer):')
            for nm, st in ct.items():
                if 'ex' in st:
                    L.append(f"    {nm:56s} n {st['n']:8,} · exceso {st['ex']:+.3f} % ± {1.96 * st['se']:.3f} (t {st['t']:+.1f}){' ← supera el umbral' if abs(st['t']) >= thr['p95'] else ''}")
        cb = ev.get('combos', {}).get(f'K{K}')
        if cb:
            vd = cb['valor_dias']
            L.append(f"  coincidencias (cada valor-día cuenta una vez; {vd['alcistas']:,} con señal alcista, {vd['bajistas']:,} con señal bajista, {vd['contradictorios']:,} contradictorios, {vd['con_varias']:,} con varias señales):")
            for sent in ('alcistas', 'bajistas'):
                for name, st in cb[sent].items():
                    if 'ex' in st:
                        L.append(f"    {sent:9s} {name:32s} n {st['n']:7,} · ganan {st['wr']:4.1f} % · exceso {st['ex']:+.3f} % (t {st['t']:+.1f}) · vs año {st['exy']:+.3f} % (t {st['ty']:+.1f})")
            L.append('    por familia (todas las señales de la familia, cada valor-día una vez; 10 contrastes: umbral de Bonferroni |t| ≥ 2,8):')
            for nm, st in cb.get('familias', {}).items():
                if 'ex' in st:
                    L.append(f"      {nm:22s} {st['patrones']:2d} patrones · n {st['n']:7,} · ganan {st['wr']:4.1f} % · exceso {st['ex']:+.3f} % ± {1.96 * st['se']:.3f} (t {st['t']:+.1f}){' ←' if abs(st['t']) >= 2.8 else ''}")
            for side, st in cb['contradictorias'].items():
                if 'ex' in st:
                    L.append(f"    contradictorias, medidas en {'largo' if side == 'long' else 'corto'}: n {st['n']:,} · exceso {st['ex']:+.3f} % (t {st['t']:+.1f})")
    return '\n'.join(L)


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, reps: int = 150, workers: int = 2, ks: Tuple[int, ...] = KS, cost: float = COST, tag: str = '', note: str = '') -> Dict:
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):                    # lo descargado queda en la caché: se vuelve a lanzar y continúa donde lo dejó
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    ev = analyze_panel(bars, ks, reps, workers, verbose=True, cost=cost)
    if note:
        ev['meta']['note'] = note
    ev['meta']['tag'] = tag
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, f'evidence{tag}.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, f'evidence_report{tag}.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
