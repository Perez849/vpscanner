"""
Estudio del PLAN completo: ¿el resultado de operar cada señal con su stop y su objetivo supera al de entrar al azar con las mismas distancias?

El estudio de evidencia (study.py) mide salidas a tiempo fijo, sin stop ni objetivo. Este responde a la objeción contraria: «quizá no funcionan porque el stop y el objetivo son enormes».

  · Cada señal se opera como en la web: entrada en la apertura siguiente, stop y objetivo del plan, el stop gana si se tocan ambos el mismo día, cierre a las 40 sesiones. Coste de ida y vuelta del 0,15 %.
  · Referencia por señal: M entradas al azar en el mismo valor (toda su historia) con el MISMO sentido y las MISMAS distancias de stop y de objetivo medidas en ATR de su propia barra
    y con una volatilidad relativa parecida (así la volatilidad del momento no cuenta como ventaja). «Exceso» = resultado de la señal − media de sus M entradas al azar.
    No se usa «el mismo año»: la referencia de ese año contendría la tendencia que provocó la señal y sesgaría el exceso hacia abajo (comprobado con paseos aleatorios).
  · Tres geometrías: «plan» (stop estructural y objetivo clásico, lo que dibuja la web), «tact» (stop táctico de 1,5 ATR con el mismo objetivo) y «fix» (stop 1,5 ATR y objetivo 3 ATR para todos).
    Con «fix» el tamaño de stop y objetivo es idéntico en todos los patrones: lo que quede es la dirección que aporta el patrón.
  · t con errores estándar agrupados en bloques de tantas sesiones como dura la operación (40 y 60): las operaciones de días vecinos comparten casi todo su recorrido y agrupar solo por día
    no lo recoge. Universo y miembros móviles como en study.py (las 50 y las 300 más negociadas).
  · «Ganan» = parte de las operaciones que terminan con ganancia neta (después de costes), llegue o no al objetivo; el reparto objetivo / stop / cierre por tiempo es solo descriptivo.
"""
from __future__ import annotations
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import engine, plan as P, study, ta
from .catalog import CATALOG, HQ

VARIANTS = ('plan', 'tact', 'fix', 'trail3', 'trail5')
TRAIL = {'trail3': 3.0, 'trail5': 5.0}      # stop que persigue al cierre más alto (o bajo) a N ATR, sin objetivo
TRAIL_HOLD = 60
HOLD = P.HOLD
COST = study.COST
M = 8                      # entradas al azar por señal y variante
VOL_BAND = 0.25            # las entradas al azar tienen una volatilidad relativa (ATR / precio) a ± 25 % de la de la señal
FIX_STOP, FIX_TARGET = 1.5, 3.0
MIN_EVENTS = study.MIN_EVENTS
RES = {'target': 0, 'stop': 1, 'timeout': 2}


def sim(o, h, l, c, i: int, d: int, stop: float, target: float, maxb: int = HOLD) -> Optional[Tuple[int, float]]:
    """(resultado, rentabilidad % antes de costes) de entrar en la apertura de la barra i+1 con ese stop y ese objetivo; None si no se puede operar o aún no se ha resuelto.
    Misma regla que plan.outcome (el stop gana si se tocan ambos el mismo día, hueco de apertura al precio de apertura), pero con cortes de NumPy."""
    n = len(c)
    e = i + 1
    if e + maxb > n:
        return None
    entry = float(o[e])
    if (d > 0 and (entry <= stop or entry >= target)) or (d < 0 and (entry >= stop or entry <= target)):
        return None
    hh, ll = h[e:e + maxb], l[e:e + maxb]
    if d > 0:
        hs, ht = ll <= stop, hh >= target
    else:
        hs, ht = hh >= stop, ll <= target
    ks = int(np.argmax(hs)) if hs.any() else maxb
    kt = int(np.argmax(ht)) if ht.any() else maxb
    if ks < maxb and ks <= kt:
        gap = (o[e + ks] <= stop) if d > 0 else (o[e + ks] >= stop)
        px = float(o[e + ks]) if gap else stop
        return RES['stop'], d * (px / entry - 1.0) * 100.0
    if kt < maxb:
        gap = (o[e + kt] >= target) if d > 0 else (o[e + kt] <= target)
        px = float(o[e + kt]) if gap else target
        return RES['target'], d * (px / entry - 1.0) * 100.0
    return RES['timeout'], d * (float(c[e + maxb - 1]) / entry - 1.0) * 100.0


def sim_trail(o, h, l, c, i: int, d: int, mult: float, a: float, maxb: int = TRAIL_HOLD) -> Optional[Tuple[int, float]]:
    """Entrada en la apertura de i+1 y stop a `mult` ATR del mejor cierre desde la entrada (solo se acerca, nunca se aleja), sin objetivo; cierre a las `maxb` sesiones.
    El stop de cada sesión se fija con los cierres anteriores; un hueco más allá del stop sale a la apertura. Devuelve (1 = stop, 2 = cierre por tiempo, rentabilidad %)."""
    n = len(c)
    e = i + 1
    if e + maxb > n:
        return None
    entry = float(o[e])
    cl = c[e:e + maxb - 1]
    if d > 0:
        ext = np.maximum.accumulate(np.concatenate([[entry], cl]))[:maxb]
        stop = np.maximum(ext, entry) - mult * a
        hit = l[e:e + maxb] <= stop
    else:
        ext = np.minimum.accumulate(np.concatenate([[entry], cl]))[:maxb]
        stop = np.minimum(ext, entry) + mult * a
        hit = h[e:e + maxb] >= stop
    if hit.any():
        k = int(np.argmax(hit))
        gap = (o[e + k] <= stop[k]) if d > 0 else (o[e + k] >= stop[k])
        px = float(o[e + k]) if gap else float(stop[k])
        return 1, d * (px / entry - 1.0) * 100.0
    return 2, d * (float(c[e + maxb - 1]) / entry - 1.0) * 100.0


def _geoms(F: Dict[str, np.ndarray], e: Dict) -> Optional[Dict[str, Tuple[float, float]]]:
    """(stop, objetivo) de cada variante para el evento e."""
    i, name = e['i'], e['name']
    d = CATALOG[name].dir
    a = float(F['atr'][i])
    if d == 0 or not a > 0:
        return None
    entry = float(F['c'][i])
    try:
        stop, tgt = P.rule(name, e['g'], F, i, d)
        bx = P.box(F, i, d, entry, stop, tgt)
    except Exception:
        return None
    tgt_ = bx['_target']
    return {'plan': (bx['_stop'], tgt_), 'tact': (entry - d * P.TACT * a, tgt_), 'fix': (entry - d * FIX_STOP * a, entry + d * FIX_TARGET * a)}


def _plan_work(args: Tuple[str, Dict[str, np.ndarray]]) -> Tuple[str, Dict]:
    """Por valor: lo de study._work más, para cada señal y variante, su resultado y la media de M entradas al azar con las mismas distancias."""
    sym, b = args
    F = ta.prepare(b)
    events, _ = engine.scan(F)
    w = study.work_from(F, events, paths=False)
    o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
    n = len(c)
    rng = np.random.default_rng(zlib.crc32(sym.encode()))
    lat = np.log(np.where(atr > 0, atr, np.nan) / c)
    pool = np.flatnonzero((np.arange(n) >= 60) & (np.arange(n) < n - HOLD - 2) & np.isfinite(atr) & (atr > 0))
    names: List[str] = []
    out = {v: {k: [] for k in ('i', 'pid', 'hq', 'ret', 'base', 'res', 'bt', 'bs', 'bo', 'ra', 'rw', 'win', 'bwin')} for v in VARIANTS}
    for e in events:
        i = e['i']
        if i < 60 or i >= n - HOLD - 2 or e['name'] not in CATALOG:
            continue
        gm = _geoms(F, e)
        if len(pool) < 100 or gm is None:
            continue
        d = CATALOG[e['name']].dir
        if e['name'] not in names:
            names.append(e['name'])
        pid = names.index(e['name'])
        hq = bool(e['q'] is not None and e['q'] >= HQ)
        a, entry = float(atr[i]), float(c[i])
        near = np.abs(lat[pool] - lat[i])                           # entradas al azar con la misma volatilidad relativa (ATR / precio) que la señal
        cand = pool[near <= VOL_BAND] if (near <= VOL_BAND).sum() >= 30 else pool[np.argsort(near)[:30]]
        js = rng.choice(cand, M)
        for v in TRAIL:
            m_ = TRAIL[v]
            r0 = sim_trail(o, h, l, c, i, d, m_, a)
            if r0 is None:
                continue
            rs = []
            for j in js:
                rj = sim_trail(o, h, l, c, int(j), d, m_, float(atr[int(j)]))
                if rj is not None:
                    rs.append(rj)
            if len(rs) < M // 2:
                continue
            res = np.array([x[0] for x in rs])
            t = out[v]
            t['i'].append(i); t['pid'].append(pid); t['hq'].append(hq); t['ret'].append(r0[1] - COST); t['res'].append(r0[0])
            t['base'].append(float(np.mean([x[1] for x in rs])) - COST)
            t['bt'].append(0.0); t['bs'].append(float((res == 1).mean())); t['bo'].append(float((res == 2).mean()))
            t['ra'].append(m_); t['rw'].append(0.0)
            t['win'].append(float(r0[1] - COST > 0)); t['bwin'].append(float(np.mean([x[1] - COST > 0 for x in rs])))
        for v, (stop, tgt) in gm.items():
            r0 = sim(o, h, l, c, i, d, stop, tgt)
            if r0 is None:
                continue
            ra, rw = abs(entry - stop) / a, abs(tgt - entry) / a
            if not (ra > 0 and rw > 0):
                continue
            rs = []
            for j in js:
                j = int(j)
                rj = sim(o, h, l, c, j, d, float(c[j]) - d * ra * float(atr[j]), float(c[j]) + d * rw * float(atr[j]))
                if rj is not None:
                    rs.append(rj)
            if len(rs) < M // 2:
                continue
            res = np.array([x[0] for x in rs])
            t = out[v]
            t['i'].append(i); t['pid'].append(pid); t['hq'].append(hq); t['ret'].append(r0[1] - COST); t['res'].append(r0[0])
            t['base'].append(float(np.mean([x[1] for x in rs])) - COST)
            t['bt'].append(float((res == 0).mean())); t['bs'].append(float((res == 1).mean())); t['bo'].append(float((res == 2).mean()))
            t['ra'].append(ra); t['rw'].append(rw)
            t['win'].append(float(r0[1] - COST > 0)); t['bwin'].append(float(np.mean([x[1] - COST > 0 for x in rs])))
    w['names'] = names
    w['pl'] = {v: {k: np.array(x, dtype=np.float64 if k in ('ret', 'base', 'bt', 'bs', 'bo', 'ra', 'rw', 'win', 'bwin') else np.int64) for k, x in t.items()} for v, t in out.items()}
    return sym, w


def _stat(P_: study.Panel, K: int, gi: np.ndarray, t: Dict[str, np.ndarray], m: np.ndarray, block: int = HOLD) -> Dict:
    ok = m & P_.mem[K][gi]
    if ok.sum() < 5:
        return {'n': int(ok.sum())}
    ex = t['ret'][ok] - t['base'][ok]
    res = t['res'][ok]
    return {'n': int(ok.sum()), 'mean': float(t['ret'][ok].mean()), 'base': float(t['base'][ok].mean()), 'ex': float(ex.mean()), 't': P_.t_cluster(ex, P_.blocks(gi[ok], block)),
            'win': float(t['win'][ok].mean() * 100), 'bwin': float(t['bwin'][ok].mean() * 100),
            'tgt': float((res == 0).mean() * 100), 'stp': float((res == 1).mean() * 100), 'tmo': float((res == 2).mean() * 100),
            'btgt': float(t['bt'][ok].mean() * 100), 'bstp': float(t['bs'][ok].mean() * 100), 'btmo': float(t['bo'][ok].mean() * 100),
            'ra': float(np.median(t['ra'][ok])), 'rw': float(np.median(t['rw'][ok]))}


def analyze_plan(bars: Dict[str, Dict[str, np.ndarray]], ks: Tuple[int, ...] = study.KS, workers: int = 1, verbose: bool = False) -> Dict:
    t0 = time.time()
    jobs = [(s, {k: b[k] for k in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if len(b['c']) > 300]
    work: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, (s, w) in enumerate(ex.map(_plan_work, jobs, chunksize=4)):
                work[s] = w
                if verbose and (k + 1) % 100 == 0:
                    print(f'    {k + 1}/{len(jobs)} valores · {time.time() - t0:.0f}s', flush=True)
    else:
        for jb in jobs:
            s, w = _plan_work(jb)
            work[s] = w
    P_ = study.Panel(work, ks)
    out: Dict = {'meta': {'generated': int(time.time()), 'universe': len(P_.syms), 'bars': int(len(P_.dayv)), 'from': str(np.datetime64(int(P_.dayv.min()), 'D')), 'to': str(np.datetime64(int(P_.dayv.max()), 'D')),
                          'cost': COST, 'hold': HOLD, 'trail_hold': TRAIL_HOLD, 'random_entries': M, 'ks': list(ks), 'variants': list(VARIANTS), 'fix': [FIX_STOP, FIX_TARGET], 'min_events': MIN_EVENTS},
                 'patterns': {}, 'pooled': {}}
    for v in VARIANTS:
        gi, pid, hq, cols = [], [], [], {k: [] for k in ('ret', 'base', 'res', 'bt', 'bs', 'bo', 'ra', 'rw', 'win', 'bwin')}
        pname: List[str] = []
        for j, s in enumerate(P_.syms):
            w = work[s]
            t = w['pl'][v]
            if not len(t['i']):
                continue
            gi.append(t['i'] + P_.off[j])
            for k in cols:
                cols[k].append(t[k])
            hq.append(t['hq'])
            pid.append(np.array([w['names'][q] for q in t['pid']]))
        if not gi:
            continue
        gi, hq, pid = np.concatenate(gi), np.concatenate(hq).astype(bool), np.concatenate(pid)
        t = {k: np.concatenate(x) for k, x in cols.items()}
        fam = np.array([CATALOG[x].fam for x in pid])
        blk = TRAIL_HOLD if v in TRAIL else HOLD                      # bloque de los errores estándar = duración de la operación
        dr = np.array([CATALOG[x].dir for x in pid])
        for K in ks:
            for nm in np.unique(pid):
                for hqf in (False, True):
                    m = (pid == nm) & (hq if hqf else True)
                    st = _stat(P_, K, gi, t, m, blk)
                    if st.get('n', 0) >= 1:
                        out['patterns'].setdefault(nm + (':hq' if hqf else ''), {'dir': CATALOG[nm].dir, 'fam': CATALOG[nm].fam, 'hq': hqf}).setdefault(v, {})[f'K{K}'] = {a: (round(b, 3) if isinstance(b, float) else b) for a, b in st.items()}
            pooled = {}
            pooled['todos'] = _stat(P_, K, gi, t, np.ones(len(gi), bool), blk)
            for f in ('figuras', 'rupturas', 'velas', 'directrices', 'rebotes'):
                for d, dn in ((1, 'alcistas'), (-1, 'bajistas')):
                    pooled[f'{f} {dn}'] = _stat(P_, K, gi, t, (fam == f) & (dr == d), blk)
            out['pooled'].setdefault(v, {})[f'K{K}'] = {k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in s_.items()} for k, s_ in pooled.items()}
    if verbose:
        print(f'  estudio del plan: {len(P_.syms)} valores, {len(P_.dayv):,} barras · {time.time() - t0:.0f}s', flush=True)
    return out


def calibrate(n: int = 100, length: int = 1500, seed: int = 11, workers: int = 1) -> Dict:
    """El mismo procedimiento sobre paseos aleatorios (sin ventaja posible): el exceso agrupado que salga aquí es sesgo del método y hay que descontarlo."""
    rng = np.random.default_rng(seed)
    bars = {}
    for k in range(n):
        sig = 0.015
        r = rng.normal(0.0, sig, length)
        c = 100 * np.exp(np.cumsum(r))
        o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, sig / 3, length))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sig / 2, length)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sig / 2, length)))
        bars[f'RW{k}'] = {'t': 1500000000 + 86400 * np.arange(length), 'o': o, 'h': h, 'l': l, 'c': c, 'v': np.full(length, 1e6)}
    ev = analyze_plan(bars, (n,), workers)
    return {v: {a: ev['pooled'][v][f'K{n}']['todos'][a] for a in ('n', 'ex', 't')} for v in ev['pooled']}


def report(ev: Dict) -> str:
    L: List[str] = []
    m = ev['meta']
    if 'calibration' in ev:
        L.append('CALIBRACIÓN con paseos aleatorios (sin ventaja posible; el exceso que salga aquí es sesgo del método y hay que descontarlo): ' +
                 ' · '.join(f"{v}: {c['ex']:+.2f} % (t {c['t']:+.1f}, n {c['n']})" for v, c in ev['calibration'].items()))
    L.append(f"ESTUDIO DEL PLAN · {m['universe']} acciones · {m['bars']:,} barras · {m['from']} → {m['to']} · coste {m['cost']} % · cierre a {m['hold']} sesiones · {m['random_entries']} entradas al azar por señal")
    desc = {'trail3': 'STOP QUE PERSIGUE a 3 ATR del mejor cierre, sin objetivo, hasta 60 sesiones', 'trail5': 'STOP QUE PERSIGUE a 5 ATR del mejor cierre, sin objetivo, hasta 60 sesiones',
            'plan': 'PLAN: stop estructural y objetivo clásico (lo que dibuja la web)', 'tact': 'TÁCTICO: stop de 1,5 ATR y el objetivo clásico', 'fix': f"FIJO: stop {m['fix'][0]} ATR y objetivo {m['fix'][1]} ATR para todos los patrones"}
    for K in m['ks']:
        for v in m['variants']:
            if v not in ev['pooled']:
                continue
            L.append(f"\n=== {desc[v]} · las {K} más negociadas ===")
            L.append('  grupo'.ljust(26) + 'n'.rjust(8) + 'media'.rjust(8) + 'azar'.rjust(8) + 'exceso'.rjust(8) + 't'.rjust(7) + '  ganan señal/azar'.rjust(19) + ' | objetivo/stop/tiempo señal vs azar'.rjust(40) + '   stop/obj (ATR)')
            for g, s in ev['pooled'][v][f'K{K}'].items():
                if s.get('n', 0) < 20 or 'ex' not in s:
                    continue
                L.append(('  ' + g).ljust(26) + f"{s['n']:8d}{s['mean']:+7.2f}%{s['base']:+7.2f}%{s['ex']:+7.2f}%{s['t']:+7.1f}" + f"  {s['win']:3.0f}/{s['bwin']:3.0f} %".rjust(15) +
                         f"  | {s['tgt']:3.0f}/{s['stp']:3.0f}/{s['tmo']:3.0f}  vs  {s['btgt']:3.0f}/{s['bstp']:3.0f}/{s['btmo']:3.0f}".rjust(40) + f"   {s['ra']:.1f}/{s['rw']:.1f}")
            rows = [(k, r[v][f'K{K}']) for k, r in ev['patterns'].items() if not r['hq'] and v in r and f'K{K}' in r[v] and r[v][f'K{K}'].get('n', 0) >= MIN_EVENTS and 'ex' in r[v][f'K{K}']]
            rows.sort(key=lambda x: -abs(x[1]['t']))
            L.append('  los 10 patrones de mayor |t|:')
            for k, s in rows[:10]:
                L.append(('    ' + ('▲ ' if CATALOG[k].dir > 0 else '▼ ') + k).ljust(30) + f"{s['n']:7d}{s['mean']:+7.2f}%{s['base']:+7.2f}%{s['ex']:+7.2f}%{s['t']:+7.1f}   ganan {s['win']:3.0f} % (azar {s['bwin']:3.0f} %)")
            pos = [s['ex'] for _, s in rows]
            L.append(f"  patrones con n ≥ {MIN_EVENTS}: {len(rows)} · exceso medio {np.mean(pos):+.3f} % · con exceso positivo {np.mean([x > 0 for x in pos]) * 100:.0f} % · |t| ≥ 3,35: {sum(abs(s['t']) >= 3.35 for _, s in rows)} (umbral orientativo; el del mejor de N está en el estudio de evidencia)")
    return '\n'.join(L)


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2) -> Dict:
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    ev = analyze_plan(bars, study.KS, workers, verbose=True)
    ev['calibration'] = calibrate(workers=workers)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'plan_study.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'plan_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
