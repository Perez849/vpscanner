"""python -m vp lab [--symbols AAPL,NVDA] [--out data]   ·   python -m vp study [--out data] [--max 600] [--reps 150]   ·   python -m vp plan [--out data] [--max 600]   ·   python -m vp ml [--out data] [--max 600]"""
from __future__ import annotations
import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog='vp', description='VP Scanner: laboratorio de chartismo')
    ap.add_argument('cmd', choices=['lab', 'study', 'plan', 'ml', 'context'])
    ap.add_argument('--symbols', default='', help='valores separados por comas (lab: watchlist.txt por defecto; study: S&P 500 + Nasdaq-100)')
    ap.add_argument('--out', default='data')
    ap.add_argument('--raw', default='', help='lab: carpeta donde volcar las series completas')
    ap.add_argument('--cache', default='.cache')
    ap.add_argument('--reps', type=int, default=150)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--max', type=int, default=0, help='study: máximo de valores (0 = todos)')
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.replace(',', ' ').split() if s.strip()]
    from . import context, lab, ml, planstudy, study, universe
    if a.cmd == 'lab':
        lab.run(syms or universe.watchlist(), a.out, raw_dir=a.raw or None)
    elif a.cmd == 'context':
        uni = syms or universe.study_universe()
        if a.max:
            uni = uni[:a.max]
        context.run(uni, a.out, cache_dir=a.cache, workers=a.workers or max(1, (os.cpu_count() or 2) - 0), reps=a.reps if a.reps != 150 else 100)
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
        uni = syms or universe.study_universe()
        if a.max:
            uni = uni[:a.max]
        study.run(uni, a.out, cache_dir=a.cache, reps=a.reps, workers=a.workers or max(1, (os.cpu_count() or 2) - 0))


if __name__ == '__main__':
    main()
    sys.stdout.flush()
