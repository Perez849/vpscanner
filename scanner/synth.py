"""synth.py — Datos sintéticos para probar el pipeline sin red (tests y --synthetic)."""
from __future__ import annotations
import time
import numpy as np


def series(n=2500, seed=0, mu=0.0, sig=0.015, ar=0.0, t_end=None, intraday_gap=0.4):
    r = np.random.default_rng(seed)
    e = r.normal(mu, sig, n)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = e[i] + ar * ret[i - 1]
    c = 100 * np.exp(np.cumsum(ret))
    o = np.concatenate([[100], c[:-1]]) * np.exp(r.normal(0, sig * intraday_gap, n))
    h = np.maximum(o, c) * np.exp(np.abs(r.normal(0, sig * 0.5, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(r.normal(0, sig * 0.5, n)))
    v = r.lognormal(16, 0.4, n)
    if t_end is None:
        t = (np.datetime64('2016-01-04').astype('datetime64[s]').astype(np.int64) + np.arange(n) * 86400).astype(float)
    else:
        t = (t_end - 86400 * (n - 1 - np.arange(n))).astype(float)
    return {'t': t, 'o': o, 'h': h, 'l': l, 'c': c, 'v': v}


def universe(n=60, ar=0.0, n_bars=3400, t_end=None, groups=('us_large', 'us_mid', 'us_small', 'thematic', 'etf')):
    uni, data = {}, {}
    for k in range(n):
        sym = f'SYN{k}'
        data[sym] = series(n_bars, seed=1000 + k, mu=0.0003, sig=0.016, ar=ar, t_end=t_end)
        uni[sym] = {'yahoo': sym, 'group': groups[k % len(groups)]}
    spy = data['SYN0']
    data['SPY'] = {**spy}
    data['^VIX'] = {**spy, 'c': 15 + 5 * np.abs(np.sin(np.arange(len(spy['c'])) / 50))}
    uni['SPY'] = {'yahoo': 'SPY', 'group': 'etf', 'aux': True}
    uni['^VIX'] = {'yahoo': '^VIX', 'group': 'index', 'aux': True}
    return uni, data
