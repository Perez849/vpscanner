"""La regla de impulso de la web aplicada a ETF apalancados (TQQQ, SQQQ, SOXL, SOXS), conservando el universo de acciones.

Se construye el mismo mundo de la cartera simulada con las acciones del universo MÁS los ETF (los percentiles de impulso y volatilidad del día se calculan con todos, como pasaría si se añadieran
a la web) y se mide: cuántas señales dan las tres entradas, cómo les va operación a operación con la escalera y con el resto del menú de salidas, y qué hace la cartera simulada con la disciplina del
plan de hoy con y sin ellos, solo con ellos y con cada uno por separado. Frente a comprar y mantener cada ETF y frente a entrar al azar en los mismos ETF con el mismo calendario.
DESCRIPTIVO: cuatro valores, muy correlacionados dos a dos (los inversos son el espejo de los otros), así que las muestras son pequeñas.

  python -m vp etfs [--out data]"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from . import cartera, ml, search

ETFS = ('TQQQ', 'SQQQ', 'SOXL', 'SOXS')
START = '2020-01-02'
BASE_RULE = 'imp_bajo200'
BASE_EXIT = 'esc'
RISK_EXITS = ('esc', 'op4b', 'h20')                # salidas con las que se prueba a dar más riesgo a los ETF
RISK_MULT = (1, 2, 3, 5)                           # veces el riesgo del 1 % por operación
DESC = {'TQQQ': 'Nasdaq-100 ×3', 'SQQQ': 'Nasdaq-100 ×3 inverso', 'SOXL': 'semiconductores ×3', 'SOXS': 'semiconductores ×3 inverso'}


def _r(x, d=2):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def _f(x, fmt: str = '') -> str:
    return 'n/d' if x is None else format(x, fmt)


def build_world(bars: Dict[str, Dict[str, np.ndarray]], join: Optional[Dict[str, str]], workers: int = 1, k: int = ml.K, verbose: bool = False):
    D, _, dropped = search.build(bars, workers, k, verbose, search.MIN_DAY, search.MIN_COMBO, join, search.COST, library=False)
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
    return D, cartera.World(D, W), dropped


def line(Wd, mask: np.ndarray, ex_: str, years_d: np.ndarray, d0: int, rng, todas: bool, order: str = 'vol') -> Dict:
    s = cartera.simulate(Wd, mask, ex_, order, rng, d0=d0, **({'max_pos': 10 ** 6, 'corr_min': 2.0} if todas else {}))
    return {**cartera.metrics(s['eq'], years_d[s['d0']:s['dend'] + 1], s['trades'], s['expo']), '_entries': s['entries']}


def hold(Wd, j: int, years_d: np.ndarray, d0: int) -> Dict:
    """Comprar y mantener un valor desde el cierre del día d0 hasta el final (los huecos sin dato arrastran el último cierre)."""
    px = Wd.close[:, j].astype(np.float64).copy()
    for d in range(1, len(px)):
        if not np.isfinite(px[d]):
            px[d] = px[d - 1]
    seg = px[d0:]
    if not np.isfinite(seg).all() or seg[0] <= 0:
        return {}
    return cartera.metrics(seg / seg[0], years_d[d0:])


def analyze(bars: Dict[str, Dict[str, np.ndarray]], join: Optional[Dict[str, str]] = None, workers: int = 1, reps: int = 200, verbose: bool = False, k: int = ml.K, etfs=ETFS, seed: int = 3) -> Dict:
    t0 = time.time()
    D, Wd, dropped = build_world(bars, join, workers, k, verbose)
    have = [s for s in etfs if s in D.syms]
    ids = {s: list(D.syms).index(s) for s in have}
    is_etf = np.isin(D.sym, list(ids.values()))
    years_d = np.zeros(Wd.nd, int)
    years_d[D.dcode] = D.year
    dates = [str(x) for x in D.date]
    d0s = {'desde_2020': int(np.searchsorted(dates, START)), 'desde_2017': 0}
    rng = np.random.default_rng(seed)
    uni_etf = (Wd.vol >= cartera.VOL_MIN) & is_etf
    out: Dict = {'meta': {'generated': int(time.time()), 'etfs': have, 'universe': Wd.ns, 'acciones': Wd.ns - len(have), 'from': dates[0], 'to': dates[-1], 'k': k, 'barras_quitadas': dropped,
                          'miembros': 'en la fecha' if join else 'actuales', 'ventana': START, 'regla_base': BASE_RULE, 'salida_base': BASE_EXIT, 'reps': reps,
                          'desde_dato': {s: str(D.date[int(D.dcode[D.sym == j].min())]) if (D.sym == j).any() else None for s, j in ids.items()}},
           'rows': {}, 'por_etf': {}, 'comprar_y_mantener': {}, 'cartera': {}, 'salidas': {}, 'azar': {}}
    rules = {r['nombre']: r for r in search.ENTRADAS}
    # 1. señales de cada entrada en los ETF y en las acciones
    for rn, r in rules.items():
        m = Wd.rule_mask(r['regla'])
        out['rows'][rn] = {'etiqueta': r['etiqueta'], 'corta': r['corta'], 'etfs': cartera.op_stats(Wd, m & is_etf, BASE_EXIT), 'acciones': cartera.op_stats(Wd, m & ~is_etf, BASE_EXIT), 'por_etf': {}}
        for s, j in ids.items():
            ix = np.flatnonzero(m & (D.sym == j) & np.isfinite(Wd.net_rows[BASE_EXIT]))
            nets = Wd.net_rows[BASE_EXIT][ix]
            yrs = D.year[ix]
            out['rows'][rn]['por_etf'][s] = {'n': int(len(ix)), 'neto': _r(float(nets.mean()), 2) if len(ix) else None, 'ganan': _r(float((nets > 0).mean() * 100), 1) if len(ix) else None,
                                              'por_año': {int(y): int((yrs == y).sum()) for y in np.unique(yrs)}}
    # 2. el menú de salidas con la regla base, solo ETF
    m0 = Wd.rule_mask(rules[BASE_RULE]['regla'])
    for e in cartera.MN:
        out['salidas'][e] = {'etfs': cartera.op_stats(Wd, m0 & is_etf, e), 'acciones': cartera.op_stats(Wd, m0 & ~is_etf, e)}
    # 3. comprar y mantener
    for w_, d0 in d0s.items():
        out['comprar_y_mantener'][w_] = {s: hold(Wd, j, years_d, d0) for s, j in ids.items()}
        mk = np.cumprod(1.0 + np.concatenate([[0.0], Wd.mkt[d0 + 1:]]))
        out['comprar_y_mantener'][w_]['universo'] = cartera.metrics(mk, years_d[d0:])
    # 4. carteras simuladas
    for w_, d0 in d0s.items():
        res: Dict = {}
        for rn, r in rules.items():
            m = Wd.rule_mask(r['regla'])
            res[rn] = {}
            for nm, mk_ in (('acciones', m & ~is_etf), ('acciones+etf', m), ('solo_etf', m & is_etf)):
                res[rn][nm] = {'plan': {k_: v for k_, v in line(Wd, mk_, BASE_EXIT, years_d, d0, rng, False).items() if k_ != '_entries'},
                               'todas': {k_: v for k_, v in line(Wd, mk_, BASE_EXIT, years_d, d0, rng, True).items() if k_ != '_entries'}}
            if rn == BASE_RULE:
                res[rn]['cada_etf'] = {s: {k_: v for k_, v in line(Wd, m & (D.sym == j), BASE_EXIT, years_d, d0, rng, True).items() if k_ != '_entries'} for s, j in ids.items()}
        out['cartera'][w_] = res
        # control de azar: entrar en los mismos ETF con el mismo calendario de entradas
        todas = line(Wd, m0 & is_etf, BASE_EXIT, years_d, d0, rng, True)
        if todas['operaciones']:
            ctrl = [cartera.simulate(Wd, uni_etf, BASE_EXIT, 'azar', rng, schedule=todas['_entries'], d0=d0) for _ in range(reps)]
            cm = [cartera.metrics(c['eq'], years_d[c['d0']:c['dend'] + 1]) for c in ctrl]
            out['azar'][w_] = {'anual': cartera._summary([x['anual'] for x in cm], todas['anual']), 'caida_max': cartera._summary([x['caida_max'] for x in cm], todas['caida_max']),
                               'operaciones_media': _r(float(np.mean([len(c['trades']) for c in ctrl])), 1)}
        # más riesgo en los ETF: el 1 % por operación con un stop de ~10 % da posiciones pequeñas y pocas señales, así que apenas pesan en la cuenta
        mas: Dict = {}
        for e in RISK_EXITS:
            mas[e] = {}
            for kx in RISK_MULT:
                rr = np.where(is_etf, float(kx), 1.0)
                res_ = {}
                for nm, mk_ in (('solo_etf', m0 & is_etf), ('acciones+etf', m0)):
                    sm = cartera.simulate(Wd, mk_, e, 'vol', rng, d0=d0, max_pos=10 ** 6, corr_min=2.0, risk_rows=rr)
                    res_[nm] = cartera.metrics(sm['eq'], years_d[sm['d0']:sm['dend'] + 1], sm['trades'], sm['expo'])
                mas[e][kx] = res_
        out['mas_riesgo_' + w_] = mas
        # el menú de salidas, solo ETF (todas las señales)
        out['salidas_cartera_' + w_] = {e: {k_: v for k_, v in line(Wd, m0 & is_etf, e, years_d, d0, rng, True).items() if k_ != '_entries'} for e in cartera.MN}
    if verbose:
        print(f'  listo · {time.time() - t0:.0f}s', flush=True)
    return out


def _lin(x: Dict) -> str:
    if not x or x.get('anual') is None:
        return 'sin operaciones'
    return (f"{x['anual']:+6.1f} %/año · caída {x['caida_max']:+6.1f} % · Sharpe {x['sharpe']} · {x.get('operaciones', 0)} ops · ganan {_f(x.get('ganan'))} % · invertido {_f(x.get('exposicion_media'), '.0f')} %"
            + ' · ' + ' '.join(f"{str(a)[2:]}:{v:+.0f}" for a, v in x['por_año'].items()))


def report(ev: Dict) -> str:
    m = ev['meta']
    L = [f"REGLA DE IMPULSO EN ETF APALANCADOS · {', '.join(f'{s} ({DESC[s]})' for s in m['etfs'])} · más {m['acciones']} acciones del universo · {m['from']} → {m['to']} · coste {search.COST} % · miembros {m['miembros']}",
         f"Mismo mundo que la cartera simulada: los percentiles de impulso y volatilidad del día se calculan con todo el universo, ETF incluidos. Entradas = las tres de la web; salida base = la escalera; "
         f"disciplina del plan de hoy (5 huecos, correlación < 0,6, 1 % de riesgo). El orden de entrada es por volatilidad, así que, al ser los más volátiles, los ETF pasan por delante de las acciones cuando coinciden. Datos de los ETF desde: " + ' · '.join(f'{s} {d}' for s, d in m['desde_dato'].items()), '']
    L.append('SEÑALES Y RESULTADO POR OPERACIÓN CON LA ESCALERA (neto tras costes; exceso = frente a la media del universo ese día con la misma salida; t con errores agrupados)')
    for rn, r in ev['rows'].items():
        for nm, tx in (('etfs', 'solo ETF'), ('acciones', 'acciones')):
            s = r[nm]
            L.append(f"   {r['corta']:10s} {tx:9s} " + (f"n {s['n']:5,d} · neto {s['neto']:+.2f} % · exceso {s['exceso']:+.2f} % (t {s['t']}) · ganan {s['ganan']} %" if s.get('neto') is not None else f"n {s['n']} (menos de 30 operaciones)"))
        L.append('             por ETF: ' + ' · '.join(f"{k_} n {v['n']}" + (f" neto {v['neto']:+.2f} % ganan {v['ganan']} %" if v['n'] else '') for k_, v in r['por_etf'].items()))
    L += ['', 'COMPRAR Y MANTENER (de referencia)']
    for w_, d_ in ev['comprar_y_mantener'].items():
        for s, x in d_.items():
            if x:
                L.append(f"   {w_:11s} {s:8s} {x['anual']:+7.1f} %/año · caída {x['caida_max']:+6.1f} % · Sharpe {x['sharpe']} · " + ' '.join(f"{str(a)[2:]}:{v:+.0f}" for a, v in x['por_año'].items()))
    for w_, res in ev['cartera'].items():
        L += ['', f"CARTERA SIMULADA · {w_} · salida: {BASE_EXIT}"]
        for rn, d_ in res.items():
            L.append(f"  {ev['rows'][rn]['etiqueta']}")
            for nm in ('acciones', 'acciones+etf', 'solo_etf'):
                for pl in ('plan', 'todas'):
                    L.append(f"     {nm:13s} {pl:6s} {_lin(d_[nm][pl])}")
            for s, x in d_.get('cada_etf', {}).items():
                L.append(f"     {s + ' solo':13s} todas  {_lin(x)}")
        if w_ in ev['azar']:
            z = ev['azar'][w_]
            L.append(f"  Control: entrar al azar en los mismos ETF con el mismo calendario: {z['anual']['mediana']:+.1f} %/año (p05 {z['anual']['p05']:+.1f}, p95 {z['anual']['p95']:+.1f}), caída mediana {z['caida_max']['mediana']:+.1f} % · "
                     f"p(azar ≥ regla) = {z['anual']['p']} ({z['anual']['n']} repeticiones)")
    for w_ in ev['cartera']:
        mas = ev.get('mas_riesgo_' + w_)
        if not mas:
            continue
        L += ['', f"MÁS RIESGO EN LOS ETF · {w_} · regla base, todas las señales; los ETF arriesgan ×k el 1 % por operación (tope del 20 % de la cuenta por posición) y las acciones siguen en el 1 %"]
        for e, d_ in mas.items():
            for kx, x in d_.items():
                a, b = x['solo_etf'], x['acciones+etf']
                L.append(f"   {e:5s} ×{kx}  solo ETF {a['anual']:+6.1f} %/año caída {a['caida_max']:+6.1f} % Sharpe {a['sharpe']:5.2f} invertido {_f(a.get('exposicion_media'), '3.0f')} %  |  acciones+ETF {b['anual']:+6.1f} %/año caída {b['caida_max']:+6.1f} % Sharpe {b['sharpe']:5.2f} invertido {_f(b.get('exposicion_media'), '3.0f')} %")
    L += ['', 'MENÚ DE SALIDAS CON LA REGLA BASE, SOLO ETF (todas las señales; ventana desde 2020)']
    L.append(f"   {'salida':9s} {'n':>5s} {'neto':>7s} {'exceso':>7s} {'t':>5s} {'ganan':>6s} | {'%/año':>7s} {'caída':>7s} {'ops':>4s}   acciones: neto exceso t")
    sal = ev.get('salidas_cartera_desde_2020', {})
    for e, d_ in ev['salidas'].items():
        s, a, c = d_['etfs'], d_['acciones'], sal.get(e, {})
        L.append(f"   {e:9s} {s['n']:5d} " + (f"{s['neto']:+6.2f}% {s['exceso']:+6.2f}% {s['t']:5.1f} {s['ganan']:5.1f}%" if s.get('neto') is not None else f"{'':26s}") + f" | {_f(c.get('anual'), '+7.1f')} {_f(c.get('caida_max'), '+7.1f')} {_f(c.get('operaciones'), '4d')}   "
                 + (f"{a['neto']:+.2f}% {a['exceso']:+.2f}% {a['t']:.1f}" if a.get('neto') is not None else ''))
    return '\n'.join(L)


def run(symbols_: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, members: str = 'asof', reps: int = 200) -> Dict:
    from . import data, universe
    bars, failed = data.fetch_many(sorted(set(symbols_) | set(ETFS)), rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    miss = [s for s in ETFS if s not in bars]
    if miss:
        raise SystemExit(f'no se pudieron descargar los ETF {miss}: se detiene')
    if len(bars) < 0.9 * len(symbols_):
        raise SystemExit(f'solo {len(bars)} de {len(symbols_)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    join = None
    if members == 'asof':
        try:
            join = universe.sp500_join_dates()
        except Exception as exc:
            print(f'  no se pudo leer la tabla de miembros del S&P 500 ({type(exc).__name__}: {exc}): se usan los miembros actuales', flush=True)
    ev = analyze(bars, join=join, workers=workers, reps=reps, verbose=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'etfs.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'etfs_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
