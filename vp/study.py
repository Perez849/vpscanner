"""
Estudio de evidencia: ¿baten los patrones a entrar al azar?

  · Universo móvil: las K acciones más negociadas en cada fecha (dinero medio de 60 sesiones con datos hasta ese día), K = 50 y 300.
  · Cada señal se opera en la apertura siguiente y se cierra a 5, 10 (principal) y 20 sesiones; los patrones bajistas se miden en corto. Coste de ida y vuelta del 0,15 %.
  · Referencia: entrar al azar en el mismo universo el mismo año («exceso» = rendimiento del patrón − media del año).
  · t con errores estándar agrupados por día (las señales de un mismo día no son independientes).
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


def _work(args: Tuple[str, Dict[str, np.ndarray]]) -> Tuple[str, Dict]:
    """Por valor: retornos futuros por horizonte, dinero negociado y los índices de barra de cada patrón (y de su variante de calidad alta)."""
    sym, b = args
    F = ta.prepare(b)
    events, _ = engine.scan(F)
    return sym, work_from(F, events)


def work_from(F: Dict[str, np.ndarray], events: List[Dict]) -> Dict:
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
    idx: Dict[str, List[int]] = {}
    for e in events:
        if e['i'] < 60:
            continue
        idx.setdefault(e['name'], []).append(e['i'])
        if e['q'] is not None and e['q'] >= HQ:
            idx.setdefault(e['name'] + ':hq', []).append(e['i'])
    return {'t': (F['t'] // 86400).astype(np.int64), 'ret': ret, 'dv': dv, 'idx': {k: np.array(v, dtype=np.int64) for k, v in idx.items()}}


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
        self.ret = {}
        for hz in HORIZONS:
            r = np.concatenate([work[s]['ret'][hz] for s in self.syms])
            self.ret[('long', hz)] = r - cost
            self.ret[('short', hz)] = -r - cost
        self.ev: Dict[str, np.ndarray] = {}
        for j, s in enumerate(self.syms):
            for k, ix in work[s]['idx'].items():
                self.ev.setdefault(k, []).append(ix + self.off[j])                           # type: ignore[arg-type]
        self.ev = {k: np.concatenate(v) for k, v in self.ev.items()}
        ny = int(self.year.max()) - self.y0 + 1
        self.base: Dict[Tuple, np.ndarray] = {}
        for K in ks:
            for key, r in self.ret.items():
                ok = self.mem[K] & np.isfinite(r)
                s_ = np.bincount(self.year[ok] - self.y0, weights=r[ok], minlength=ny)
                c_ = np.bincount(self.year[ok] - self.y0, minlength=ny)
                self.base[(K,) + key] = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)

    def t_cluster(self, ex: np.ndarray, code: np.ndarray) -> float:
        """t de la media con errores estándar agrupados por día."""
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
        ok = self.mem[K][idx] & np.isfinite(r)
        if ok.sum() < 5:
            return {'n': int(ok.sum())}
        i2, x = idx[ok], r[ok]
        ex = x - self.base[(K, side, hz)][self.year[i2] - self.y0]
        yrs = self.year[i2]
        pos = tot = 0
        for Y in np.unique(yrs):
            m = yrs == Y
            if m.sum() >= 8:
                tot += 1
                pos += int(ex[m].mean() > 0)
        return {'n': int(ok.sum()), 'wr': float((x > 0).mean() * 100), 'mean': float(x.mean()), 'med': float(np.median(x)), 'ex': float(ex.mean()),
                't': self.t_cluster(ex, self.dcode[i2]), 'pos': pos, 'tot': tot}

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
                ok = self.mem[K][j] & np.isfinite(r)
                if ok.sum() < MIN_EVENTS:
                    continue
                ex = r[ok] - self.base[(K, side, PRIMARY)][self.year[j[ok]] - self.y0]
                t = abs(self.t_cluster(ex, self.dcode[j[ok]]))
                if t > best:
                    best = t
            out[q] = best
        return out


def analyze_panel(bars: Dict[str, Dict[str, np.ndarray]], ks: Tuple[int, ...] = KS, reps: int = 150, workers: int = 1, verbose: bool = False) -> Dict:
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
    P = Panel(work, ks)
    out: Dict = {'meta': {'generated': int(time.time()), 'universe': len(P.syms), 'bars': int(len(P.dayv)), 'from': str(np.datetime64(int(P.dayv.min()), 'D')), 'to': str(np.datetime64(int(P.dayv.max()), 'D')),
                          'cost': P.cost, 'primary': PRIMARY, 'horizons': list(HORIZONS), 'ks': list(ks), 'hq': HQ, 'thr': {}, 'tested': {}, 'min_events': MIN_EVENTS,
                          'note': 'Universo = miembros actuales del S&P 500 y del Nasdaq-100: sesgo de supervivencia que favorece a los largos.'},
                 'baseline': {}, 'patterns': {}}
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
    L.append(f"ESTUDIO DE EVIDENCIA · {m['universe']} acciones · {m['bars']:,} barras · {m['from']} → {m['to']} · coste {m['cost']} % · salida a {m['primary']} sesiones")
    for K in m['ks']:
        thr = m['thr'][f'K{K}']
        t_ = m['tested'][f'K{K}']
        L.append(f"\n=== Las {K} acciones más negociadas en cada fecha ({thr['bars']:,} barras) · umbral del mejor de {t_['n']}: |t| ≥ {thr['p95']} (p95) / {thr['p99']} (p99) ===")
        rows = [(k, r[f'K{K}']) for k, r in ev['patterns'].items() if f'K{K}' in r and r[f'K{K}'].get('n', 0) >= MIN_EVENTS and 'ex' in r[f'K{K}']]
        rows.sort(key=lambda x: -abs(x[1]['t']))
        L.append('  patrón'.ljust(30) + 'n'.rjust(7) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'exceso'.rjust(8) + 't'.rjust(7) + 'años+'.rjust(7))
        for k, s in rows[:25]:
            star = '★' if abs(s['t']) >= thr['p95'] else ' '
            L.append(('  ' + ('▲ ' if CATALOG[k.split(':')[0]].dir > 0 else '▼ ') + k + star).ljust(30) + f"{s['n']:7d}{s['wr']:8.1f}%{s['mean']:+7.2f}%{s['ex']:+7.2f}%{s['t']:+7.1f}{s['pos']:4d}/{s['tot']:<2d}")
        L.append(f"  superan el umbral del mejor de {t_['n']}: {len(t_['above'])} ({', '.join(t_['above']) or 'ninguno'}); por azar se esperarían ≈ {0.05 * t_['n']:.1f} si se miraran sueltos")
        for fam in ('figuras', 'rupturas', 'velas', 'directrices', 'rebotes'):
            for d, dn in ((1, 'alcistas'), (-1, 'bajistas')):
                sel = [s for k, s in rows if ev['patterns'][k]['fam'] == fam and ev['patterns'][k]['dir'] == d and not ev['patterns'][k]['hq']]
                if sel:
                    L.append(f"  {fam:12s} {dn:9s} {len(sel):2d} patrones · exceso medio {np.mean([s['ex'] for s in sel]):+.3f} % · positivos {np.mean([s['ex'] > 0 for s in sel]) * 100:.0f} %")
    return '\n'.join(L)


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, reps: int = 150, workers: int = 2) -> Dict:
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):                    # lo descargado queda en la caché: se vuelve a lanzar y continúa donde lo dejó
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    ev = analyze_panel(bars, KS, reps, workers, verbose=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'evidence.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'evidence_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
