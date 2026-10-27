"""
Cribador diario de la regla de impulso con caída (parte 2 del laboratorio, separada del chartismo).

Aplica al ÚLTIMO cierre del universo (S&P 500 + Nasdaq-100, las 300 más negociadas ese día) las cinco reglas candidatas de `search.CANDIDATAS`, con exactamente las mismas condiciones y los mismos
percentiles del día que usa la búsqueda y su confirmación (`search.conditions`): una acción cumple una regla si hoy está en el 20 % mejor del universo por impulso a 12 meses (sin el último mes)
Y ha caído (bajo su media de 200 sesiones, RSI(2) ≤ 10 o 2 o más señales bajistas a la vez).

Para cada acción que cumple alguna regla da el plan de las salidas validadas: stop inicial a 2,5 ATR, nivel donde se asegura +4 % o +6 %, y la alternativa de cerrar a 10 sesiones. Además guarda cada día
qué acciones cumplían cada regla (`impulso_hist.json`) para poder medir más adelante, hacia delante y sin posibilidad de ajustar nada, qué ha pasado con las que salieron.
"""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from . import ml, search
from .ml import CTX

HIST_MAX = 500             # días que se conservan en el registro
STOP_ATR = 2.5             # stop inicial de las salidas op4b y op6b


def _flags(D: ml.Data, rules) -> Dict[str, np.ndarray]:
    cond = search.conditions(D)
    out = {}
    for r in rules:
        m = np.ones(len(D.sym), bool)
        for k in r['regla']:
            m &= cond[k]
        out[r['nombre']] = m
    return out


def screen(bars: Dict[str, Dict[str, np.ndarray]], k: int = ml.K, workers: int = 1, rules=search.CANDIDATAS, verbose: bool = False) -> Dict:
    """Acciones que cumplen cada regla en el último cierre. `bars`: barras diarias por valor (≥ 400 barras)."""
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    jobs = [(s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if len(b['c']) > 400]
    work: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n_, (s, w) in enumerate(ex.map(ml._work, jobs, chunksize=4)):
                work[s] = w
                if verbose and (n_ + 1) % 100 == 0:
                    print(f'    {n_ + 1}/{len(jobs)} valores · {time.time() - t0:.0f}s', flush=True)
    else:
        for jb in jobs:
            s, w = ml._work(jb)
            work[s] = w
    for w in work.values():
        w['ret'] = np.zeros_like(w['ret'])                                  # la última sesión no tiene rendimiento futuro: aquí no se usa y el panel exige que sea finito
    D = ml.Data(work, k=k)
    flags = _flags(D, rules)
    last = int(D.dcode.max())
    today, before = D.dcode == last, D.dcode == last - 1
    ctx = lambda name: D.ctx[:, CTX.index(name)]
    pct = lambda name: D.X_ctx[:, CTX.index(name)]
    pn = {n: j for j, n in enumerate(D.pnames)}
    nbear = D.pat[:, pn['n_velas_dn']].astype(int) + D.pat[:, pn['n_fig_dn']].astype(int) + D.pat[:, pn['n_rup_dn']].astype(int)
    hit_today = {r['nombre']: set(D.sym[flags[r['nombre']] & today].tolist()) for r in rules}
    hit_before = {r['nombre']: set(D.sym[flags[r['nombre']] & before].tolist()) for r in rules}
    picks = []
    rows_today = np.flatnonzero(today)
    for row in rows_today:
        j = int(D.sym[row])
        names = [r['nombre'] for r in rules if j in hit_today[r['nombre']]]
        if not names:
            continue
        s = D.syms[j]
        close = float(bars[s]['c'][int(D.bar[row])])
        atrp = float(ctx('atrp')[row])
        atr = atrp * close / 100.0
        picks.append({'sym': s, 'close': round(close, 4), 'atr': round(atr, 4), 'atrp': round(atrp, 2), 'dist200': round(float(ctx('dist200')[row]) * 100, 2), 'r120': round(float(pct('r120')[row]), 3),
                      'mom121': round(float(pct('mom12_1')[row]), 3), 'rsi2': round(float(ctx('rsi2')[row]), 1), 'pos52': round(float(ctx('pos52')[row]), 3), 'volrel': round(float(D.volraw[row]), 2),
                      'bajistas': int(nbear[row]), 'reglas': names, 'nuevas': [n for n in names if j not in hit_before[n]],
                      'plan': {'stop': round(close - STOP_ATR * atr, 4), 'stop_pct': round(-STOP_ATR * atr / close * 100, 2), 'asegura4': round(close * 1.04, 4), 'asegura6': round(close * 1.06, 4),
                               'persigue4': round(3 * atr, 4), 'persigue6': round(4 * atr, 4), 'tam1': round(100.0 / (STOP_ATR * atr / close * 100), 1)}})
    picks.sort(key=lambda p: (-len(p['reglas']), -p['mom121'], p['sym']))
    bre = float(D.breadth_d[last])
    return {'meta': {'asof': str(D.date[last]), 'generated': int(time.time()), 'valores': len(D.syms), 'k': k, 'en_universo': int(today.sum()), 'sobre_200': round(bre * 100, 1), 'coste': ml.COST,
                     'reglas': [{'nombre': r['nombre'], 'regla': r['regla'], 'sentido': r['sentido'], 'salida': r['salida'], 'texto': search._rule(r['regla'], r['sentido'])} for r in rules]},
            'resumen': {r['nombre']: {'hoy': len(hit_today[r['nombre']]), 'nuevas': len(hit_today[r['nombre']] - hit_before[r['nombre']])} for r in rules}, 'picks': picks}


def update_history(path: str, ev: Dict) -> List[Dict]:
    """Añade al registro (un elemento por día, sin repetir fecha) las acciones que cumplían cada regla ese día."""
    try:
        hist = json.load(open(path, encoding='utf-8'))
    except (OSError, ValueError):
        hist = []
    today = {'date': ev['meta']['asof'], 'reglas': {r['nombre']: sorted(p['sym'] for p in ev['picks'] if r['nombre'] in p['reglas']) for r in ev['meta']['reglas']}}
    hist = [h for h in hist if h.get('date') != today['date']] + [today]
    hist.sort(key=lambda h: h['date'])
    return hist[-HIST_MAX:]


def run(symbols: List[str], out_dir: str, rng: str = '3y', cache_dir: Optional[str] = None, workers: int = 2) -> Dict:
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=12, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene')
    ev = screen(bars, workers=workers, verbose=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'impulso.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    hist = update_history(os.path.join(out_dir, 'impulso_hist.json'), ev)
    json.dump(hist, open(os.path.join(out_dir, 'impulso_hist.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    m = ev['meta']
    print(f"CRIBADOR DE LA REGLA DE IMPULSO · {m['asof']} · {m['en_universo']} acciones en el universo (las {m['k']} más negociadas) · {m['sobre_200']} % sobre su media de 200", flush=True)
    for r in m['reglas']:
        s = ev['resumen'][r['nombre']]
        print(f"  {r['nombre']} {r['texto']} · {r['salida']}: {s['hoy']} hoy ({s['nuevas']} nuevas)", flush=True)
    for p in ev['picks'][:15]:
        print(f"    {p['sym']:6s} {p['close']:9.2f}  impulso {p['mom121'] * 100:3.0f}  vs 200 {p['dist200']:+6.1f} %  RSI2 {p['rsi2']:4.1f}  {','.join(p['reglas'])}", flush=True)
    sys.stdout.flush()
    return ev
