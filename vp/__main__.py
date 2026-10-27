"""python -m vp lab [--symbols AAPL,NVDA] [--out data]   ·   python -m vp study [--universe large|smallmid|mixed] [--out data] [--max 600] [--reps 150]   ·   python -m vp plan [--out data] [--max 600]
   ·   python -m vp ml [--out data] [--max 600]   ·   python -m vp context [--out data] [--reps 100]   ·   python -m vp search [--out data] [--reps 30]"""
from __future__ import annotations
import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog='vp', description='VP Scanner: laboratorio de chartismo')
    ap.add_argument('cmd', choices=['lab', 'study', 'plan', 'ml', 'context', 'search'])
    ap.add_argument('--symbols', default='', help='valores separados por comas (lab: watchlist.txt por defecto; study: S&P 500 + Nasdaq-100)')
    ap.add_argument('--out', default='data')
    ap.add_argument('--raw', default='', help='lab: carpeta donde volcar las series completas')
    ap.add_argument('--cache', default='.cache')
    ap.add_argument('--reps', type=int, default=0, help='repeticiones del umbral del mejor de N (0 = las de cada estudio: 150 en study, 100 en context, 30 en search)')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--max', type=int, default=0, help='study: máximo de valores (0 = todos)')
    ap.add_argument('--universe', default='large', choices=['large', 'smallmid', 'mixed'], help='study: large = S&P 500 + Nasdaq-100 · smallmid = S&P 400 + 600 · mixed = cripto, divisas, materias primas e índices')
    ap.add_argument('--ks', default='', help='study: tamaños del universo móvil por dinero negociado, separados por comas (por defecto 50,300; smallmid 100,500; mixed todos)')
    ap.add_argument('--cost', type=float, default=-1.0, help='study: coste de ida y vuelta en % (por defecto 0,15; smallmid 0,30)')
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.replace(',', ' ').split() if s.strip()]
    from . import context, lab, ml, planstudy, search, study, universe
    if a.cmd == 'lab':
        lab.run(syms or universe.watchlist(), a.out, raw_dir=a.raw or None)
    elif a.cmd == 'context':
        uni = syms or universe.study_universe()
        if a.max:
            uni = uni[:a.max]
        context.run(uni, a.out, cache_dir=a.cache, workers=a.workers or max(1, (os.cpu_count() or 2) - 0), reps=a.reps or 100)
    elif a.cmd == 'search':
        uni = syms or universe.study_universe()
        if a.max:
            uni = uni[:a.max]
        search.run(uni, a.out, cache_dir=a.cache, workers=a.workers or max(1, (os.cpu_count() or 2) - 0), reps=a.reps or 30)
    elif a.cmd == 'ml':
        uni = syms or universe.study_universe()
        if a.max:
            uni = uni[:a.max]
        ml.run(uni, a.out, cache_dir=a.cache, workers=a.workers or max(1, (os.cpu_count() or 2) - 0))
    elif a.cmd == 'plan':
        uni = syms or universe.study_universe()
        if a.max:
            uni = uni[:a.max]
        planstudy.run(uni, a.out, cache_dir=a.cache, workers=a.workers or max(1, (os.cpu_count() or 2) - 0))
    else:
        uni = syms or universe.universe_by_name(a.universe)
        if a.max:
            uni = uni[:a.max]
        if not uni:
            raise SystemExit(f'universo «{a.universe}» vacío (¿Wikipedia no responde?)')
        ks = tuple(int(x) for x in a.ks.split(',') if x) or {'large': study.KS, 'smallmid': (100, 500), 'mixed': (len(uni),)}[a.universe]
        cost = a.cost if a.cost >= 0 else {'large': study.COST, 'smallmid': 0.30, 'mixed': study.COST}[a.universe]
        tag = '' if a.universe == 'large' else f'_{a.universe}'
        note = {'large': '', 'smallmid': 'Universo = miembros actuales del S&P 400 y del S&P 600 (sesgo de supervivencia); coste 0,30 %.',
                'mixed': 'Cesta de 50 instrumentos: criptomonedas, divisas, materias primas, bonos, índices y futuros de índices (sin sesgo de supervivencia de acciones, pocas series).'}[a.universe]
        study.run(uni, a.out, cache_dir=a.cache, reps=a.reps or 150, workers=a.workers or max(1, (os.cpu_count() or 2) - 0), ks=ks, cost=cost, tag=tag, note=note)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
