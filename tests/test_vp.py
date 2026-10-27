"""
Pruebas de VP Scanner (ejecutar: python tests/test_vp.py). Sin red.

1. Indicadores y detectores son causales: recortar el futuro no cambia ninguna señal anterior.
2. Cada patrón plantado a mano salta en su barra de ruptura (velas, rupturas, figuras).
3. El plan: objetivo con la medida clásica, stop mínimo de 1 ATR, T1 intermedio, resultado simulado, figura en formación con sus dos escenarios.
4. El JSON del laboratorio es coherente; la descarga de Yahoo se interpreta bien.
5. El estudio de evidencia: t por clústeres, ventaja plantada detectada y sin ventaja en paseos aleatorios.
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vp import breakouts, candles, context, data, engine, lab, ml, plan, planstudy, screen, search, shapes, study, ta, universe
from vp.catalog import CATALOG


# ── generadores ───────────────────────────────────────────────────────────
def plant_bars(oc, wick=0.25):
    o = np.array([x[0] for x in oc], float); c = np.array([x[1] for x in oc], float)
    h = np.maximum(o, c) + wick; l = np.minimum(o, c) - wick
    for k, x in enumerate(oc):
        if len(x) == 4:
            h[k], l[k] = x[2], x[3]
    n = len(o)
    return {'t': 1577836800 + 86400 * np.arange(n), 'o': o, 'h': h, 'l': l, 'c': c, 'v': np.full(n, 1e6)}


def plant_closes(cl, wick=0.25):
    pc, oc = cl[0], []
    for x in cl:
        oc.append((pc, x)); pc = x
    return plant_bars(oc, wick)


def quiet(n, p, sd=0.12, seed=5):
    return list(p + np.random.default_rng(seed).normal(0, sd, n))


def custom(prefix, tail):
    base = plant_closes(np.array(prefix))
    oc = [(base['o'][k], base['c'][k], base['h'][k], base['l'][k]) for k in range(len(prefix))] + list(tail)
    return plant_bars(oc)


def walk(n=1500, seed=0, mu=0.0, sig=0.015, t0=1500000000):
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sig, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, sig / 3, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sig / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sig / 2, n)))
    return {'t': t0 + 86400 * np.arange(n), 'o': o, 'h': h, 'l': l, 'c': c, 'v': rng.integers(5e5, 2e6, n).astype(float)}


def events_of(b):
    F = ta.prepare(b)
    ev, fm = engine.scan(F)
    return F, ev, fm


# ── 1. causalidad ─────────────────────────────────────────────────────────
def test_causality():
    b = walk(1600, seed=2)
    F = ta.prepare(b)
    ev, _ = engine.scan(F)
    for cut in (700, 1100, 1400):
        G = ta.prepare({k: v[:cut] for k, v in b.items()})
        for k in ('atr', 'sma50', 'rsi14', 'pos52'):
            assert np.allclose(F[k][:cut], G[k], equal_nan=True), k
        ev2, _ = engine.scan(G)
        a = {(e['name'], e['i']) for e in ev if e['i'] < cut - 1}
        c = {(e['name'], e['i']) for e in ev2 if e['i'] < cut - 1}
        assert a == c, (cut, sorted(a ^ c)[:5])
    print(f'ok · causalidad: {len(ev)} eventos y los indicadores no cambian al recortar el futuro')


# ── 2. patrones plantados ─────────────────────────────────────────────────
def test_candles_planted():
    rng = np.random.default_rng(3)

    def candle(trend, tail):
        pre, p = [], 100.0
        for k in range(200):
            c = 100.0 + rng.normal(0, 0.3); pre.append((p, c, max(p, c) + 0.25, min(p, c) - 0.25)); p = c
        p = 100.0
        for k in range(25):
            c = p - 1.2 if trend == 'dn' else p + 1.2
            pre.append((p, c, max(p, c) + 0.3, min(p, c) - 0.3)); p = c
        return plant_bars(pre + tail)
    cases = {
        'hammer': ('dn', [(70, 70.4, 70.5, 67.5)]), 'inv_hammer': ('dn', [(70, 70.4, 73.0, 69.9)]), 'shooting_star': ('up', [(130, 129.6, 133.0, 129.5)]),
        'hanging_man': ('up', [(130, 130.4, 130.5, 127.5)]), 'bull_engulf': ('dn', [(70, 68.8, 70.2, 68.6), (68.6, 71, 71.2, 68.4)]),
        'bear_engulf': ('up', [(130, 131.2, 131.4, 129.8), (131.4, 129, 131.6, 128.8)]), 'piercing': ('dn', [(70, 67.8, 70.2, 67.5), (67.2, 69.2, 69.4, 67.0)]),
        'dark_cloud': ('up', [(130, 132.2, 132.5, 129.8), (132.8, 130.8, 133.0, 130.6)]),
        'morning_star': ('dn', [(70, 67.8, 70.2, 67.6), (67.2, 67.0, 67.4, 66.8), (67.3, 69.5, 69.7, 67.1)]),
        'evening_star': ('up', [(130, 132.2, 132.4, 129.8), (132.8, 133.0, 133.2, 132.6), (132.7, 130.5, 132.9, 130.3)]),
        'soldiers': ('dn', [(68, 69.4, 69.5, 67.9), (68.6, 70.4, 70.5, 68.5), (69.6, 71.5, 71.6, 69.5)]),
        'crows': ('up', [(132, 130.6, 132.1, 130.5), (131.4, 129.6, 131.5, 129.5), (130.4, 128.5, 130.5, 128.4)]),
        'harami_bull': ('dn', [(70, 67.8, 70.2, 67.6), (68.2, 69.0, 69.2, 68.0)]), 'harami_bear': ('up', [(130, 132.2, 132.4, 129.8), (131.8, 131.0, 132.0, 130.8)]),
        'tweezer_bottom': ('dn', [(70, 68.8, 70.2, 68.0), (68.6, 70.0, 70.2, 68.0)]), 'tweezer_top': ('up', [(130, 131.4, 132.0, 129.8), (131.2, 129.8, 132.0, 129.6)]),
        'dragonfly': ('dn', [(68.0, 68.1, 68.1, 65.5)]), 'gravestone': ('up', [(130.0, 129.9, 132.5, 129.9)]),
        'marubozu_bull': ('dn', [(70, 73.6, 73.6, 70.0)]), 'marubozu_bear': ('up', [(130, 126.4, 130, 126.4)]),
        'keyrev_bull': ('dn', [(70, 68.6, 70.1, 68.4), (68.2, 70.4, 70.5, 68.0)]), 'keyrev_bear': ('up', [(130, 131.4, 131.6, 129.9), (131.8, 129.5, 132.0, 129.3)]),
    }
    cases.update({
        'belt_bull': ('dn', [(70.0, 74.0, 74.2, 70.0)]), 'belt_bear': ('up', [(130.0, 126.0, 130.0, 125.8)]),
        'kicker_bull': ('dn', [(70.5, 68.5, 70.7, 68.3), (71.0, 74.0, 74.2, 70.9)]), 'kicker_bear': ('up', [(129.5, 131.5, 131.7, 129.3), (129.0, 126.0, 129.1, 125.8)]),
        'counter_bull': ('dn', [(70, 67.5, 70.2, 67.3), (65.5, 67.5, 67.7, 65.4)]), 'counter_bear': ('up', [(130, 132.5, 132.7, 129.8), (134.5, 132.5, 134.6, 132.3)]),
        'harami_cross_bull': ('dn', [(70, 66.5, 70.2, 66.3), (68.2, 68.25, 69.2, 67.2)]), 'harami_cross_bear': ('up', [(130, 133.5, 133.7, 129.8), (131.8, 131.75, 132.8, 130.8)]),
        'homing_pigeon': ('dn', [(70, 66.0, 70.2, 65.8), (69.0, 67.0, 69.2, 66.8)]), 'descending_hawk': ('up', [(130, 134.0, 134.2, 129.8), (131.0, 133.0, 133.2, 130.8)]),
        'matching_low': ('dn', [(70, 68.0, 70.2, 67.8), (69.5, 68.0, 69.7, 67.8)]), 'matching_high': ('up', [(130, 132.0, 132.2, 129.8), (130.5, 132.0, 132.2, 130.3)]),
        'three_outside_up': ('dn', [(70, 68.0, 70.2, 67.8), (67.5, 71.0, 71.2, 67.3), (71.0, 73.0, 73.2, 70.8)]),
        'three_outside_dn': ('up', [(130, 132.0, 132.2, 129.8), (132.5, 129.0, 132.7, 128.8), (129.0, 127.0, 129.2, 126.8)]),
        'three_strike_bull': ('up', [(130, 131.5, 131.7, 129.9), (131.4, 133.0, 133.2, 131.3), (133.0, 134.5, 134.7, 132.9), (135.0, 129.5, 135.2, 129.3)]),
        'three_strike_bear': ('dn', [(70, 68.5, 70.1, 68.3), (68.6, 67.0, 68.8, 66.8), (67.0, 65.5, 67.1, 65.3), (65.0, 70.5, 70.7, 64.8)]),
        'abandoned_bull': ('dn', [(70, 66.5, 70.2, 66.3), (65.0, 65.05, 65.5, 64.5), (66.0, 70.0, 70.2, 65.9)]),
        'abandoned_bear': ('up', [(130, 133.5, 133.7, 129.8), (135.0, 134.95, 135.5, 134.5), (134.0, 130.0, 134.1, 129.8)]),
        'advance_block': ('up', [(130, 133.0, 133.4, 129.9), (131.5, 134.0, 135.0, 131.4), (132.5, 134.5, 136.2, 132.4)]),
        'deliberation': ('up', [(130, 133.5, 133.6, 129.9), (132.0, 135.5, 135.6, 131.9), (135.4, 136.0, 137.0, 135.3)]),
        'tasuki_up': ('up', [(130, 132.5, 132.6, 129.9), (134.0, 136.0, 136.1, 133.9), (135.0, 133.5, 135.2, 133.3)]),
        'tasuki_dn': ('dn', [(70, 67.5, 70.1, 67.4), (66.0, 64.0, 66.1, 63.9), (65.0, 66.5, 66.7, 64.8)]),
        'two_crows_gap': ('up', [(130, 133.5, 133.7, 129.9), (135.0, 134.2, 135.2, 134.0), (135.5, 133.8, 135.7, 133.7)]),
        'stick_sandwich_bull': ('dn', [(70, 68.0, 70.2, 67.8), (68.5, 71.0, 71.2, 68.3), (70.5, 68.0, 70.7, 67.9)]),
        'stick_sandwich_bear': ('up', [(130, 132.0, 132.2, 129.8), (131.5, 129.0, 131.7, 128.8), (129.5, 132.0, 132.2, 129.4)]),
        'outside_up': ('dn', [(70, 68.5, 70.2, 68.3), (68.0, 71.5, 71.7, 67.8)]), 'outside_dn': ('up', [(130, 131.5, 131.7, 129.8), (132.0, 128.5, 132.2, 128.3)]),
    })
    for nm, (tr, tail) in cases.items():
        F = ta.prepare(candle(tr, tail))
        assert candles.detect(F)[nm][-1], f'{nm}: no se detecta en la barra final'
    # cinco velas, huecos con vela interior y consenso (varias formaciones del mismo sentido en la misma barra)
    F = ta.prepare(candle('up', [(130, 134.0, 134.2, 129.9), (133.5, 132.0, 134.0, 131.8), (132.2, 131.0, 132.6, 130.8), (131.2, 132.5, 132.9, 131.0), (132.3, 136.5, 136.7, 132.2)]))
    assert candles.detect(F)['rising_methods'][-1]
    F = ta.prepare(candle('dn', [(70, 66.0, 70.2, 65.8), (66.5, 67.8, 68.2, 66.3), (67.2, 68.2, 68.4, 67.0), (68.0, 67.0, 68.4, 66.8), (67.0, 63.0, 67.2, 62.8)]))
    assert candles.detect(F)['falling_methods'][-1]
    F = ta.prepare(candle('up', [(130, 134.0, 134.2, 129.8), (132.5, 133.5, 133.9, 131.8), (133.0, 136.0, 136.2, 132.9)]))
    assert candles.detect(F)['inside_break_up'][-1]
    F = ta.prepare(candle('dn', [(70, 66.0, 70.2, 65.8), (67.5, 66.5, 67.9, 66.2), (66.8, 63.0, 66.9, 62.8)]))
    assert candles.detect(F)['inside_break_dn'][-1]
    F = ta.prepare(candle('dn', [(70, 66.5, 70.2, 66.3), (68.2, 68.25, 69.2, 67.2)]))
    d_ = candles.detect(F)
    n_dn = sum(d_[k][-1] for k in d_ if CATALOG[k].fam == 'velas' and CATALOG[k].dir > 0 and not k.startswith('consenso'))
    assert n_dn >= 1 and d_['consenso_alcista_2'][-1] == (n_dn >= 2), (n_dn, d_['consenso_alcista_2'][-1])
    print(f'ok · velas: {len(cases) + 4} patrones plantados se detectan en su barra y el consenso cuenta bien')


def test_breakouts_planted():
    chk = lambda b, nm: bool(breakouts.detect(ta.prepare(b))[nm][-1])
    for nm, tail in (('donch55_up', (100.0, 103.0, 103.1, 99.9)), ('donch55_dn', (100.0, 97.0, 100.1, 96.9)), ('donch252_up', (100.0, 103.0, 103.1, 99.9)), ('donch100_dn', (100.0, 97.0, 100.1, 96.9))):
        assert chk(custom(quiet(300, 100, 0.2), [tail]), nm), nm
    hv = list(100 + np.cumsum(np.random.default_rng(9).normal(0, 1.0, 150)) * 0.3)
    base = plant_closes(np.array(hv), 0.25)
    oc = [(base['o'][k], base['c'][k], base['h'][k], base['l'][k]) for k in range(len(hv))]
    last = hv[-1]
    box = [(last, last + (0.15 if k % 2 else -0.15), last + 0.3, last - 0.3) for k in range(16)]
    assert chk(plant_bars(oc + box + [(last, last + 1.5, last + 1.6, last - 0.1)]), 'box_up')
    assert chk(plant_bars(oc + box + [(last, last - 1.5, last + 0.1, last - 1.6)]), 'box_dn')
    print('ok · rupturas: Donchian y caja de Darvas plantadas se detectan')


FIGS = {
    'double_bottom': [(0, 100), (20, 80), (32, 92), (44, 80.5), (52, 93.5)], 'double_top': [(0, 100), (20, 120), (32, 108), (44, 119.5), (52, 106.5)],
    'hs_top': [(0, 80), (15, 100), (24, 92), (36, 108), (48, 92), (58, 100.5), (70, 85)], 'hs_inverse': [(0, 120), (15, 100), (24, 108), (36, 92), (48, 108), (58, 99.5), (70, 115)],
    'asc_triangle': [(0, 90), (10, 100), (18, 93), (26, 100.1), (34, 96), (42, 100), (50, 98.5), (56, 102)],
    'desc_triangle': [(0, 110), (10, 100), (18, 107), (26, 99.9), (34, 104), (42, 100), (50, 101.5), (56, 98)],
    'sym_tri_up': [(0, 90), (10, 110), (20, 92), (30, 106), (40, 96), (48, 102), (54, 98), (60, 104)], 'sym_tri_dn': [(0, 110), (10, 90), (20, 108), (30, 94), (40, 104), (48, 98), (54, 102), (60, 95)],
    'rising_wedge': [(0, 90), (10, 100), (20, 94), (30, 103), (40, 100), (50, 105.5), (58, 103), (64, 99)], 'falling_wedge': [(0, 110), (10, 100), (20, 106), (30, 97), (40, 100), (50, 94.5), (58, 97), (64, 101)],
    'trendline_up': [(0, 100), (10, 110), (18, 100), (28, 107.5), (36, 100), (46, 105), (54, 100), (60, 108)], 'trendline_dn': [(0, 100), (10, 90), (18, 100), (28, 92.5), (36, 100), (46, 95), (54, 100), (60, 92)],
    'bull_flag': [(0, 100), (6, 112), (14, 110.5), (16, 114.5)], 'bear_flag': [(0, 100), (6, 88), (14, 89.5), (16, 85.5)],
    'triple_bottom': [(0, 100), (15, 80), (25, 92), (40, 80.4), (50, 92), (65, 80.2), (75, 93.5)], 'triple_top': [(0, 100), (15, 120), (25, 108), (40, 119.6), (50, 108), (65, 119.8), (75, 106.5)],
    'channel_up_up': [(0, 100), (7, 108.8), (14, 105.6), (21, 114.4), (28, 111.2), (35, 120), (42, 116.8), (49, 125.6), (56, 137)],
    'channel_up_dn': [(0, 100), (7, 108.8), (14, 105.6), (21, 114.4), (28, 111.2), (35, 120), (42, 116.8), (49, 125.6), (56, 108)],
    'channel_dn_dn': [(0, 100), (7, 91.2), (14, 94.4), (21, 85.6), (28, 88.8), (35, 80), (42, 83.2), (49, 74.4), (56, 63)],
    'channel_dn_up': [(0, 100), (7, 91.2), (14, 94.4), (21, 85.6), (28, 88.8), (35, 80), (42, 83.2), (49, 74.4), (56, 92)],
    'rect_up': [(0, 100), (8, 116), (16, 101), (24, 115.6), (32, 100.6), (40, 115.8), (48, 100.8), (54, 119)], 'rect_dn': [(0, 116), (8, 100), (16, 115), (24, 100.6), (32, 115.4), (40, 100.4), (48, 115.2), (54, 97)],
}
COMPRESS = ('asc_triangle', 'desc_triangle', 'sym_tri_up', 'sym_tri_dn', 'rising_wedge', 'falling_wedge', 'rect_up', 'rect_dn')


def plant_fig(nm, an=None, seed=5):
    """Figura con precios de cierre unidos por tramos; las compresiones, con bases realistas (mitad de amplitud y mechas de 0,5). Devuelve (barras, barra final de la figura)."""
    an = an or FIGS[nm]
    if nm in COMPRESS:
        y0 = an[0][1]
        an = [(x, y0 + (y - y0) * 0.5) for x, y in an]
    xs, ys = np.array([a[0] for a in an]), np.array([a[1] for a in an])
    cl = np.interp(np.arange(xs[0], xs[-1] + 1), xs, ys)
    pre = quiet(260, an[0][1], seed=seed)
    return plant_closes(np.array(pre + list(cl)), 0.5 if nm in COMPRESS else 0.25), len(pre) + int(xs[-1])


def test_figures_planted():
    for nm in FIGS:
        b, n_end = plant_fig(nm)
        _, ev, _ = events_of(b)
        hit = [e for e in ev if e['name'] == nm and n_end - 12 <= e['i'] <= n_end]
        assert hit, (nm, [(e['name'], e['i']) for e in ev if e['i'] > 250][:6], n_end)
        assert hit[0]['q'] is None or 0 <= hit[0]['q'] <= 100
    # las que se construyen con velas explícitas
    zig = lambda lo, hi, k: sum(([*np.linspace(hi, lo, 8), *np.linspace(lo, hi, 8)] for _ in range(k)), [])
    def has(b, nm):
        return any(e['name'] == nm and e['i'] == len(b['c']) - 1 for e in events_of(b)[1])
    assert has(custom(quiet(250, 100) + zig(90.2, 100, 3) + list(np.linspace(100, 92, 10)), [(91.0, 91.8, 91.9, 90.3)]), 'support_bounce')
    assert has(custom(quiet(250, 100) + sum(([*np.linspace(100, 109.8, 8), *np.linspace(109.8, 100, 8)] for _ in range(3)), []) + list(np.linspace(100, 108, 10)), [(109.0, 108.2, 109.7, 108.1)]), 'resistance_reject')
    v = list(np.linspace(40, 100, 300)) + list(np.linspace(100, 88, 8)) + list(np.linspace(88, 100.5, 8)) + list(np.linspace(100.5, 93.5, 8)) + list(np.linspace(93.5, 101, 8)) + list(np.linspace(101, 97.5, 8)) + list(np.linspace(97.5, 101.2, 6)) + [102.6]
    assert has(plant_closes(np.array(v)), 'vcp')
    x = np.linspace(-1, 1, 72); cup = list(100 - 20 * (1 - x ** 2)); cup[-1] = 99.0
    allc = quiet(250, 80) + list(np.linspace(80, 100, 30)) + cup + list(np.linspace(99, 95.5, 8)) + list(np.linspace(95.5, 98.5, 3)) + [101.0]
    assert has(plant_closes(np.array(allc)), 'cup_handle')
    widths = [2, 4, 7, 10, 8, 5, 3, 2]
    dia = []
    for w in widths:
        for k in range(5):
            hi, lo = 100 + w / 2, 100 - w / 2
            dia.append((lo + 0.3, hi - 0.3, hi, lo) if k % 2 == 0 else (hi - 0.3, lo + 0.3, hi, lo))
    assert has(custom(quiet(250, 130) + list(np.linspace(130, 100, 20)), dia + [(100.5, 103.5, 103.7, 100.4)]), 'diamond_bull')
    assert has(custom(quiet(250, 70) + list(np.linspace(70, 100, 20)), dia + [(99.5, 96.5, 99.6, 96.3)]), 'diamond_bear')
    pen = lambda c0, s: [((c0 + s * (1.8 * (1 - k / 9))) - 0.2 if k % 2 else (c0 - s * (1.8 * (1 - k / 9))) + 0.2, (c0 - s * (1.8 * (1 - k / 9))) + 0.2 if k % 2 else (c0 + s * (1.8 * (1 - k / 9))) - 0.2,
                          c0 + 1.8 * (1 - k / 9), c0 - 1.8 * (1 - k / 9)) for k in range(8)]
    assert has(custom(quiet(250, 100) + list(np.linspace(100, 112, 7)), pen(111, 1) + [(111.2, 114.5, 114.6, 111.1)]), 'bull_pennant')
    assert has(custom(quiet(250, 100) + list(np.linspace(100, 88, 7)), pen(89, 1) + [(88.8, 85.5, 88.9, 85.4)]), 'bear_pennant')
    assert has(custom(quiet(250, 130) + list(np.linspace(130, 100, 20)), [(99.0, 98.6, 99.2, 98.4), (98.5, 98.9, 99.0, 98.3), (101.5, 102.4, 102.6, 101.3)]), 'island_bull')
    assert has(custom(quiet(250, 70) + list(np.linspace(70, 100, 20)), [(101.0, 101.4, 101.6, 100.8), (101.5, 101.1, 101.7, 101.0), (98.5, 97.6, 98.7, 97.4)]), 'island_bear')
    print(f'ok · figuras: {len(FIGS) + 11} figuras plantadas saltan en su barra de ruptura')


def test_forming_figures():
    """Figuras dibujadas pero sin romper: salen como «en formación» con su disparador, la distancia en ATR y un solo escenario; al romper pasan a ser eventos."""
    cases = {
        'double_bottom': ([(0, 100), (20, 80), (32, 92), (44, 80.5), (50, 89.5)], 1, 92.0),
        'double_top': ([(0, 100), (20, 120), (32, 108), (44, 119.5), (50, 110.5)], -1, 108.0),
        'hs_top': ([(0, 80), (15, 100), (24, 92), (36, 108), (48, 92), (58, 100.5), (64, 94.5)], -1, 92.0),
        'hs_inverse': ([(0, 120), (15, 100), (24, 108), (36, 92), (48, 108), (58, 99.5), (64, 105.5)], 1, 108.0),
        'triple_bottom': ([(0, 100), (15, 80), (25, 92), (40, 80.4), (50, 92), (65, 80.2), (70, 90)], 1, 92.0),
        'bull_flag': ([(0, 100), (6, 112), (14, 110.5), (15, 111.3)], 1, 112.0),
    }
    for nm, (an, d, trig) in cases.items():
        b, _ = plant_fig(nm, an=an)
        F = ta.prepare(b)
        ev, fm = engine.scan(F)
        f = [x for x in fm if x['type'] == 'fig' and x['name'] == nm or (nm == 'bull_flag' and x.get('name') == 'bull_pennant')]     # una bandera corta puede salir como banderín
        assert f, (nm, [(x['type'], x.get('name') or x.get('kind')) for x in fm])
        f = f[0]
        assert f['d'] == d and 0 <= f['dist'] <= shapes.NEAR and f['q'] >= shapes.QFORM, (nm, f['d'], f['dist'], f['q'])
        assert abs(f['trig'] - trig) <= 0.12 * abs(trig - float(F['c'][-1])) + 1.5, (nm, f['trig'], trig)
        assert not [e for e in ev if e['name'] == nm and e['i'] >= len(F['c']) - 3], nm              # aún no ha roto: no es un evento
        nm = f['name']
        res = lab.analyze('T', b, nb=300)
        it = [p for p in res['patterns'] if p['key'] == 'forming_' + nm]
        assert it and it[0]['state'] == 'forming' and it[0]['dir'] == d and set(it[0]['sc']) == {'up' if d > 0 else 'dn'}, nm
        sc = it[0]['sc']['up' if d > 0 else 'dn']
        assert (sc['stop'] < sc['entry'] < sc['target']) if d > 0 else (sc['target'] < sc['entry'] < sc['stop']), (nm, sc)
        assert it[0]['trig']['dist'] == round(f['dist'], 2) and any(L_['k'] == 'neck' for L_ in it[0]['lines']), nm
    # taza con asa y VCP: la misma construcción que en test_figures_planted, sin la barra de ruptura
    x = np.linspace(-1, 1, 72); cup = list(100 - 20 * (1 - x ** 2)); cup[-1] = 99.0
    allc = quiet(250, 80) + list(np.linspace(80, 100, 30)) + cup + list(np.linspace(99, 95.5, 8)) + list(np.linspace(95.5, 98.5, 3))
    v = list(np.linspace(40, 100, 300)) + list(np.linspace(100, 88, 8)) + list(np.linspace(88, 100.5, 8)) + list(np.linspace(100.5, 93.5, 8)) + list(np.linspace(93.5, 101, 8)) + list(np.linspace(101, 97.5, 8)) + list(np.linspace(97.5, 101.2, 6))
    for nm, closes in (('cup_handle', allc), ('vcp', v)):
        F = ta.prepare(plant_closes(np.array(closes)))
        ev, fm = engine.scan(F)
        f = [x_ for x_ in fm if x_['type'] == 'fig' and x_['name'] == nm]
        assert f and f[0]['d'] == 1 and f[0]['dist'] <= shapes.NEAR and not [e for e in ev if e['name'] == nm and e['i'] >= len(F['c']) - 3], (nm, fm)
    # una figura ya rota y aún activa no se anuncia además como «en formación»
    b, _ = plant_fig('double_bottom')
    res = lab.analyze('T', b, nb=300)
    assert any(p_['key'] == 'double_bottom' and p_['active'] for p_ in res['patterns']) and not any(p_['key'] == 'forming_double_bottom' for p_ in res['patterns'])
    print(f'ok · figuras en formación: {len(cases) + 2} figuras sin romper salen con su disparador, su distancia en ATR y un solo escenario')


def test_spent_target_and_first_break():
    """Si la ruptura ya recorrió la medida clásica no hay recorrido que operar; y el cuello solo se rompe la primera vez que el cierre lo pasa."""
    b, _ = plant_fig('double_bottom', an=[(0, 100), (20, 80), (32, 92), (44, 80.5), (51, 90), (52, 112)])
    res = lab.analyze('T', b, nb=300)
    db = [p for p in res['patterns'] if p['key'] == 'double_bottom']
    assert db and db[-1]['plan']['spent'] and 'recorrió casi toda la medida' in db[-1]['note']
    ok, _ = plant_fig('double_bottom')
    res = lab.analyze('T', ok, nb=300)
    db = [p for p in res['patterns'] if p['key'] == 'double_bottom']
    assert db and not db[-1]['plan']['spent']
    # tras romper el cuello, caer por debajo y volver a cruzarlo no es una ruptura nueva de la misma figura
    b, n_end = plant_fig('double_bottom', an=[(0, 100), (20, 80), (32, 92), (44, 80.5), (52, 93.5), (60, 91), (64, 91.5), (68, 95)])
    ev = [e for e in events_of(b)[1] if e['name'] == 'double_bottom']
    assert len(ev) == 1 and ev[0]['i'] <= n_end + 2, [(e['name'], e['i']) for e in ev]
    print('ok · plan: objetivo consumido por la ruptura y cuello roto solo la primera vez')


def test_watchlist():
    got = universe.parse_watchlist('# comentario\naapl, msft;nvda  # y esto\n  BRK-B ^GSPC $$$ AAPL\n\n')
    assert got == ['AAPL', 'MSFT', 'NVDA', 'BRK-B', '^GSPC'], got
    assert universe._clean(['BRK.B', 'aapl', 1, 'TOOLONGSYMBOL', ' msft ']) == ['AAPL', 'BRK-B', 'MSFT']
    mx = universe.mixed()
    assert len(mx) >= 40 and len(set(mx)) == len(mx) and 'BTC-USD' in mx and 'EURUSD=X' in mx and universe.universe_by_name('mixed') == mx
    w = universe.watchlist()
    assert len(w) >= 50 and len(set(w)) == len(w) and 'AAPL' in w and all(universe.TICKER.match(x) for x in w)
    assert universe.watchlist('/ruta/que/no/existe.txt') == list(universe.WATCHLIST)
    print(f'ok · lista de seguimiento: watchlist.txt con {len(w)} valores sin repetidos y lector tolerante')


def test_catalog():
    """Todo lo que detecta el motor está en el catálogo y viceversa (salvo lo que no salta en datos aleatorios)."""
    names = set(candles.detect(ta.prepare(walk(600))).keys()) | set(breakouts.detect(ta.prepare(walk(600))).keys())
    assert names <= set(CATALOG), names - set(CATALOG)
    for nm in FIGS:
        assert nm in CATALOG
    assert all(p.fam in ('velas', 'rupturas', 'figuras', 'directrices', 'rebotes') and p.rule and p.desc for p in CATALOG.values())
    print(f'ok · catálogo: {len(CATALOG)} patrones con descripción y regla')


# ── 3. plan ───────────────────────────────────────────────────────────────
def test_plan():
    # doble suelo con ruptura que alcanza el objetivo; y otro que se hunde hasta el stop
    def run(tail):
        an = [(0, 100), (20, 80), (32, 92), (44, 80.5), (52, 93.5)] + tail
        xs, ys = np.array([a[0] for a in an]), np.array([a[1] for a in an])
        cl = np.interp(np.arange(xs[0], xs[-1] + 1), xs, ys)
        b = plant_closes(np.array(quiet(260, 100) + list(cl)), 0.4)
        res = lab.analyze('T', b, nb=300)
        return res, [p for p in res['patterns'] if p['key'] == 'double_bottom']
    res, dbs = run([(60, 112)])
    assert dbs
    p = dbs[-1]
    rr = p['plan']
    neck = next(l['a'][1] for l in p['lines'] if l['k'] == 'neck')
    low = min(q[1] for q in p['pts'] if q[2] in ('1', '2'))
    assert rr['stop'] < rr['entry'] < rr['target'] and p['dir'] == 1
    assert abs(rr['target_c'] - (neck + (neck - low))) < 0.02, (rr, neck, low)        # objetivo clásico = cuello + altura de la figura
    assert rr['stop_s'] < low                                                         # stop clásico bajo el mínimo más bajo
    assert rr['stop'] > rr['stop_s'] and rr['stop_cap']                               # ese stop está lejísimos en ATR: el plan del operador lo acorta
    assert abs(rr['target'] - rr['entry'] * 1.04) < 0.02                              # nivel que se asegura: +4 %
    assert p['out_s']['res'] == 'target' and p['out_s']['ret'] > 0 and p['out_s']['xi'] > p['i']          # el plan clásico llega al objetivo
    assert p['out']['res'] in ('target', 'open') and p['out']['locked'] and p['out']['ret'] >= 3.9 and p['out']['xi'] > p['i']      # y el del operador asegura el +4 % (la serie acaba con la operación en curso)
    res2, dbs2 = run([(56, 70)])
    assert dbs2 and dbs2[-1]['out']['res'] == 'stop' and dbs2[-1]['out']['ret'] < 0
    # invariantes de todos los patrones del valor: stop a ≥ 1 ATR, T1 intermedio, lados correctos
    for res_, b_ in ((res, None), (res2, None)):
        for q in res_['patterns']:
            r_ = q['plan']
            if r_ is None:
                continue
            assert (r_['stop'] < r_['entry'] < r_['target']) if q['dir'] > 0 else (r_['target'] < r_['entry'] < r_['stop']), (q['key'], r_)
    s = json.dumps(res)
    assert all(z['touches'] >= 2 and z['lo'] < z['p'] < z['hi'] for z in res['zones'])
    assert len(res['bars']['c']) == res['n'] == len(res['sma50']) and all(0 <= pp['i'] < res['n'] for pp in res['patterns'])
    print(f'ok · plan: objetivo = cuello + altura, stop bajo la figura, T1 intermedio, resultado simulado y JSON de {len(s) // 1024} KB')


def test_plan_rules():
    """Plan del operador: stop estructural acotado a [1, 2] ATR, nivel de aseguramiento a +4 % y avisos informativos que NO excluyen la señal."""
    n = 80
    c = np.full(n, 100.0)
    F = {'atr': np.full(n, 2.0), 'c': c, 'o': c, 'h': c + 1, 'l': c - 1}                    # ATR 2 % del precio
    a = plan.box(F, 60, 1, 100.0, 80.0, 140.0)                                                # stop estructural a 20 % y objetivo clásico a 40 %
    assert a['stop'] == 96.0 and a['stop_cap'] and a['risk_atr'] == 2.0 and a['target'] == 104.0 and a['lock_pct'] == 4.0 and a['rr'] == 1.0 and a['ok'], a      # 2 ATR = 4 %; se asegura a +4 %
    assert a['stop_s'] == 80.0 and a['target_c'] == 140.0 and a['rr_s'] == 2.0                # el plan clásico se conserva como referencia
    b = plan.box(F, 60, 1, 100.0, 99.0, 105.0)                                                # stop pegado: se ensancha a 1 ATR
    assert b['stop'] == 98.0 and b['stop_min'] and not b['stop_cap'] and b['rr'] == 2.0 and b['ok'], b
    e = plan.box(F, 60, -1, 100.0, 105.0, 90.0)                                               # corto: stop de 5 % acortado a 4 %, se asegura a −4 %
    assert e['stop'] == 104.0 and e['target'] == 96.0 and e['stop_cap'] and e['ok'], e
    F2 = {'atr': np.full(n, 6.0), 'c': c, 'o': c, 'h': c + 1, 'l': c - 1}                    # ATR 6 %: el stop de 6 % está más lejos que el +4 % que se asegura
    g = plan.box(F2, 60, 1, 100.0, 94.0, 140.0)
    assert g['risk_pct'] == 6.0 and g['rr_low'] and not g['risk_hi'] and g['ok'], g           # aviso, no exclusión
    g2 = plan.box({'atr': np.full(n, 6.0), 'c': c, 'o': c, 'h': c + 1, 'l': c - 1}, 60, 1, 100.0, 85.0, 140.0)
    assert g2['risk_pct'] == 12.0 and g2['risk_hi'] and g2['ok'], g2                          # riesgo alto: aviso, sigue contando
    print('ok · plan del operador: stop acotado a 2 ATR, aseguramiento a +4 %, avisos sin excluir ninguna señal')


def test_min_stop_and_forming():
    b, _ = plant_fig('rising_wedge')
    F = ta.prepare(b)
    res = lab.analyze('T', b, nb=300)
    atr = {p['i']: float(F['atr'][p['i'] + len(F['c']) - 300]) for p in res['patterns'] if p['plan']}
    for p in res['patterns']:
        if p['plan']:
            risk = abs(p['plan']['entry'] - p['plan']['stop'])
            assert 0.99 * atr[p['i']] - 1e-6 <= risk <= (plan.MAX_RISK_ATR + 0.01) * atr[p['i']] + 1e-6, (p['key'], p['plan'])           # riesgo entre 1 y 2 ATR
    # triángulo simétrico sin romper: figura en formación con escenario alcista y bajista
    an = [(0, 100), (10, 110), (20, 102), (30, 108), (40, 104), (48, 106.5), (54, 105)]
    y0 = an[0][1]; an = [(x, y0 + (y - y0) * 0.5) for x, y in an]
    xs, ys = np.array([a[0] for a in an]), np.array([a[1] for a in an])
    b = plant_closes(np.array(quiet(260, 100) + list(np.interp(np.arange(xs[0], xs[-1] + 1), xs, ys))), 0.5)
    res = lab.analyze('T', b, nb=300)
    fm = [p for p in res['patterns'] if p['state'] == 'forming' and p['key'].startswith('forming_sym_tri')]
    assert len(fm) == 1 and fm[0]['key'] == 'forming_sym_tri' and fm[0]['dir'] == 0 and set(fm[0]['sc']) == {'up', 'dn'}
    up, dn = fm[0]['sc']['up'], fm[0]['sc']['dn']
    last = res['summary']['close']
    assert dn['stop'] > dn['entry'] > dn['target'] and up['stop'] < up['entry'] < up['target'] and dn['entry'] < last < up['entry']
    assert fm[0]['ann'] and 0 <= fm[0]['q'] <= 100 and all(ln.get('n', 2) >= 2 for ln in fm[0]['lines'])
    print('ok · figuras en formación con dos escenarios y stop mínimo de 1 ATR')


def test_shapes_geometry():
    b, n_end = plant_fig('rising_wedge')
    _, ev, _ = events_of(b)
    e = next(e for e in ev if e['name'] == 'rising_wedge' and n_end - 12 <= e['i'] <= n_end)
    g = e['g']
    assert len(g['tu']) >= 2 and len(g['tl']) >= 2 and g['U'][2] > 0 and g['D'][2] > g['U'][2] and 0 <= e['q'] <= 100
    xs = np.arange(g['x0'], e['i'])
    assert (shapes.at(g['U'], xs) - shapes.at(g['D'], xs)).min() > 0
    # la recta de soporte prefiere los pivotes recientes a una recta plana vieja con más toques
    b, n_end = plant_fig('sym_tri_up')
    assert any(e['name'] == 'sym_tri_up' for e in events_of(b)[1])
    print('ok · geometría: rectas con ≥ 2 toques por lado, la superior sobre la inferior')


# ── 4. datos y laboratorio ────────────────────────────────────────────────
def test_data_parse():
    ts = [1700000000 + 86400 * k for k in range(12)]
    q = {'open': [10 + k for k in range(12)], 'high': [11 + k for k in range(12)], 'low': [9 + k for k in range(12)], 'close': [10.5 + k for k in range(12)], 'volume': [1000 + k for k in range(12)]}
    q['close'][3] = None
    b = data.parse({'chart': {'result': [{'meta': {'longName': 'Prueba SA'}, 'timestamp': ts, 'indicators': {'quote': [q]}}], 'error': None}})
    assert len(b['c']) == 11 and b['name'] == 'Prueba SA' and np.all(np.diff(b['t']) > 0)
    q2 = {k: list(v) for k, v in q.items()}
    q2['high'][5] = q2['close'][5] - 0.5                                   # máximo por debajo del cierre: error de dato que se corrige a max(o, h, c)
    q2['low'][6] = q2['open'][6] + 0.7                                     # mínimo por encima de la apertura
    b3 = data.parse({'chart': {'result': [{'meta': {}, 'timestamp': ts, 'indicators': {'quote': [q2]}}], 'error': None}})
    assert (b3['h'] >= np.maximum(b3['o'], b3['c'])).all() and (b3['l'] <= np.minimum(b3['o'], b3['c'])).all()
    assert data.parse({'chart': {'result': None, 'error': {'code': 'Not Found'}}}) is None
    assert data.span('3y', 1.7e9).startswith('period1=') and data.span('6mo') == 'range=6mo'
    last = int(datetime(2023, 11, 15, 14, 30, tzinfo=timezone.utc).timestamp())
    b2 = {'t': np.array([last - 86400, last]), 'o': np.ones(2), 'h': np.ones(2), 'l': np.ones(2), 'c': np.ones(2), 'v': np.ones(2), 'name': 'x'}
    assert len(data.drop_open_bar(b2, datetime(2023, 11, 15, 17, 0, tzinfo=timezone.utc))['c']) == 1      # 12:00 en Nueva York: la sesión sigue abierta
    assert len(data.drop_open_bar(b2, datetime(2023, 11, 15, 22, 0, tzinfo=timezone.utc))['c']) == 2      # 17:00: cerrada
    print('ok · datos: se interpreta la respuesta de Yahoo y se descarta la barra de la sesión abierta')


def test_data_retries():
    """Ante un 429 de Yahoo se reintenta con pausa común y el otro servidor; un 404 no se reintenta."""
    import io
    import urllib.error
    import urllib.request
    ts = [1700000000 + 86400 * k for k in range(8)]
    q = {'open': list(range(10, 18)), 'high': list(range(11, 19)), 'low': list(range(9, 17)), 'close': [10.5 + k for k in range(8)], 'volume': [1000] * 8}
    payload = json.dumps({'chart': {'result': [{'meta': {}, 'timestamp': ts, 'indicators': {'quote': [q]}}], 'error': None}}).encode()
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout=0):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, 'Too Many', {}, None)
        return Resp(payload)

    real, real_wait = urllib.request.urlopen, data.WAIT
    urllib.request.urlopen, data.WAIT = fake, 0.0
    try:
        b = data.fetch_one('ZZZ', '1y')
        assert b is not None and len(b['c']) == 8 and 'query1' in calls[0] and 'query2' in calls[1]
        calls.clear()
        urllib.request.urlopen = lambda req, timeout=0: (calls.append(req.full_url), Resp(payload))[1]
        assert data.fetch_one('^GSPC', '1y') is not None and '%5EGSPC' in calls[0] and data.fetch_one('EURUSD=X', '1y') is not None and 'EURUSD%3DX' in calls[1]      # símbolos de índices y divisas
        urllib.request.urlopen = fake
        calls.clear()

        def nf(req, timeout=0):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 404, 'Not Found', {}, None)
        urllib.request.urlopen = nf
        assert data.fetch_one('NOPE', '1y') is None and len(calls) == 1
        # límite persistente: tras MAX_STRIKES respuestas 429 seguidas se abandona sin seguir esperando
        calls.clear(); data._strikes = 0

        def tm(req, timeout=0):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 429, 'Too Many', {}, None)
        urllib.request.urlopen = tm
        res = [data.fetch_one(f'T{k}', '1y') for k in range(8)]
        assert all(r_ is None for r_ in res) and len(calls) <= data.MAX_STRIKES + 1 and data.throttled(), (len(calls), data._strikes)
    finally:
        urllib.request.urlopen, data.WAIT = real, real_wait
        data._strikes = 0
    print('ok · datos: reintento ante 429 con el otro servidor, sin reintentos ante 404 y abandono ante un límite persistente')


def test_lab_json(tmp='/tmp/vp_lab_test'):
    b = walk(1200, seed=4)
    res = lab.analyze('RW', b, name='Paseo')
    json.dumps(res)
    assert res['name'] == 'Paseo' and res['sym'] == 'RW'                       # el nombre de la empresa no se pisa con el de una figura
    n = res['n']
    assert n == 340 and len(res['bars']['t']) == n
    for p in res['patterns']:
        assert 0 <= p['i'] < n and p['key'] in CATALOG or p['key'].startswith('forming_')
        for pt in p['pts']:
            assert 0 <= pt[0] < n + 60
    row = lab.index_row(res)
    assert row['sym'] == 'RW' and len(row['spark']) == lab.SPARK
    T = res['trades']
    done = [p for p in res['patterns'] if p['state'] == 'done']
    assert len(T['list']) < len(done) and T['all']['n'] <= len(T['list']) and all(p['trade']['role'] in ('lider', 'confirma', 'contraria', 'na') for p in done)     # las señales solapadas no se cuentan una a una
    assert sum(1 for p in done if p['trade']['role'] == 'lider') == len(T['list'])
    for p in done:
        o = p['out']
        if o['res'] in ('target', 'stop', 'timeout'):
            assert {'net', 'r', 'mfe', 'mae', 'prog', 't1'} <= set(o) and (o['res'] != 'target' or o['net'] > 0) and (o['res'] != 'stop' or o['net'] <= 0), o
    os.makedirs(tmp, exist_ok=True)
    lab.write_catalog(tmp)
    cat = json.load(open(os.path.join(tmp, 'catalog.json')))
    assert set(cat['patterns']) == set(CATALOG)
    print(f'ok · laboratorio: JSON coherente ({len(res["patterns"])} patrones en el paseo aleatorio)')


# ── 5. estudio ────────────────────────────────────────────────────────────
def test_study_mechanics():
    # t con clústeres: coincide con la fórmula a mano
    work = {f'S{j}': study._work((f'S{j}', walk(1200, seed=10 + j, t0=1500000000)))[1] for j in range(6)}
    P = study.Panel(work, ks=(3, 6))
    rng = np.random.default_rng(0)
    ex = rng.normal(0.3, 2.0, 400)
    code = rng.integers(0, 40, 400)
    s = np.bincount(code, weights=ex, minlength=P.ndays); k = np.bincount(code, minlength=P.ndays)
    manual = ex.mean() / (np.sqrt(((s - ex.mean() * k) ** 2).sum()) / 400)
    assert abs(P.t_cluster(ex, code) - manual) < 1e-9
    # una ventaja plantada se detecta: en las barras señaladas el rendimiento a 10 sesiones es +3 %
    work2 = {}
    for j in range(8):
        b = walk(1200, seed=30 + j)
        _, w = study._work((f'S{j}', b))
        idx = np.arange(100 + j, 1100, 9)                       # cada valor en días distintos: la referencia del mismo día no absorbe el efecto (si todos saltaran a la vez, sería cosa del mercado)
        w['idx'] = {'hammer': idx}
        r = w['ret'][10].copy()
        r[idx] += 3.0
        w['ret'][10] = r
        pa = w['path'].copy()
        pa[idx, 9:] += 3.0                                      # el efecto plantado también aparece en la trayectoria desde la sesión 10
        w['path'] = pa
        work2[f'S{j}'] = w
    P2 = study.Panel(work2, ks=(8,))
    st = P2.stat(P2.ev['hammer'], 8, 'long', 10)
    assert st['n'] > 500 and st['ex'] > 2.3 and st['t'] > 8 and st['exy'] > 2.5 and st['ty'] > 8, st
    ps = P2.path_stat(P2.ev['hammer'], 8, 'long')
    assert abs(ps['m'][9] - 3.0) < 0.9 and abs(ps['m'][0]) < 0.4 and ps['se'][9] > 0 and len(ps['m']) == study.PATH, ps
    ps_s = P2.path_stat(P2.ev['hammer'], 8, 'short')
    assert ps_s['m'][9] < -2.0                                  # en corto el mismo efecto cuenta al revés
    print(f'ok · estudio: t por clústeres exacta y ventaja plantada de +3 % detectada (t = {st["t"]:.0f})')


def test_controls_detect_planted_momentum():
    """Control positivo: un efecto plantado en el 10 % de mejor rendimiento a 120 sesiones de cada día lo ve el mismo motor; en paseos aleatorios no sale nada."""
    work = {f'S{j}': study._work((f'S{j}', walk(1500, seed=900 + j, t0=1500000000)))[1] for j in range(30)}
    P0 = study.Panel(work, ks=(30,))
    cr = study.control_rows(P0, 30)
    top = cr['momentum alto (10 % mejor a 120 sesiones)']
    base = study.controls(P0, (30,))['K30']
    assert all(abs(st['t']) < 3.6 for st in base.values() if 't' in st), base                       # sin efecto: ningún control sale significativo
    for j, sym in enumerate(work):
        rows = top[(top >= P0.off[j]) & (top < P0.off[j] + P0.nbar[j])] - P0.off[j]
        r = work[sym]['ret'][10].copy()
        r[rows] += 1.0
        work[sym]['ret'][10] = r
    P1 = study.Panel(work, ks=(30,))
    st = study.controls(P1, (30,))['K30']['momentum alto (10 % mejor a 120 sesiones)']
    assert st['n'] > 2000 and st['ex'] > 0.8 and st['t'] > 5 and st['se'] > 0, st
    print(f'ok · controles positivos: el efecto plantado en el momentum alto sale con t = {st["t"]:.0f} y sin él ningún control pasa el umbral')


def test_study_no_edge_on_random_walk():
    bars = {f'S{j}': walk(1500, seed=100 + j, t0=1500000000) for j in range(24)}
    ev = study.analyze_panel(bars, ks=(8, 24), reps=40, workers=1)
    for K in (8, 24):
        t_ = ev['meta']['tested'][f'K{K}']
        thr = ev['meta']['thr'][f'K{K}']['p95']
        assert t_['n'] >= 10 and np.isfinite(thr)
        assert len(t_['above']) <= max(2, 0.2 * t_['n']), (K, t_['above'])        # sin ventaja real, casi ninguno supera el umbral del mejor de N
    assert ev['meta']['universe'] == 24 and set(ev['patterns']) and all('dir' in r for r in ev['patterns'].values())
    txt = study.report(ev)
    assert 'umbral del mejor de' in txt
    print(f'ok · estudio: en 24 paseos aleatorios {len(ev["meta"]["tested"]["K24"]["above"])} de {ev["meta"]["tested"]["K24"]["n"]} patrones superan el umbral del mejor de N')


def test_plan_study():
    """El simulador del estudio del plan coincide con plan.outcome y, sobre paseos aleatorios, el plan no «bate» al azar con las mismas distancias."""
    from vp import plan as P_
    rng = np.random.default_rng(1)
    tot = 0
    for seed in range(3):
        b = walk(900, seed=seed)
        F = ta.prepare(b)
        o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
        for _ in range(120):
            i, d = int(rng.integers(60, 800)), int(rng.choice([-1, 1]))
            a = float(atr[i])
            stop, tgt = c[i] - d * rng.uniform(0.6, 6) * a, c[i] + d * rng.uniform(0.6, 8) * a
            s1, r = planstudy.sim(o, h, l, c, i, d, stop, tgt), P_.outcome(F, i, d, stop, tgt)
            if s1 is None:
                assert r['res'] in ('na', 'open'), r
                continue
            tot += 1
            assert {'target': 0, 'stop': 1, 'timeout': 2}[r['res']] == s1[0] and abs(s1[1] - r['ret']) < 1e-2, (s1, r)
    # el plan del operador vectorizado coincide con plan.outcome_lock (con 3 años de paseo aleatorio, largos y cortos, varios niveles)
    nl = 0
    for seed in range(3):
        b = walk(900, seed=40 + seed)
        F = ta.prepare(b)
        o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
        for _ in range(150):
            i, d, lp = int(rng.integers(60, 780)), int(rng.choice([-1, 1])), float(rng.choice([3.0, 4.0, 5.0]))
            a = float(atr[i])
            stop = c[i] - d * rng.uniform(1.0, 2.0) * a
            v1, r1 = planstudy.sim_lock(o, h, l, c, i, d, stop, lp, a), P_.outcome_lock(F, i, d, stop, lock_pct=lp)
            if v1 is None:
                assert r1['res'] in ('na', 'open') or r1['bars'] < planstudy.TRAIL_HOLD, (v1, r1)
                continue
            nl += 1
            code = 0 if r1['res'] == 'target' else 1 if r1['res'] == 'stop' else 2
            assert code == v1[0] and abs(v1[1] - r1['ret']) < 1e-2, (v1, r1)
    assert nl > 200
    # stop que persigue: sube, y al caer sale a 3 ATR del mejor cierre (a 127 si el máximo fue 130); en corto, sin tocarse, cierra por tiempo
    n_ = 200
    cc = np.concatenate([np.linspace(100, 130, 60), np.linspace(130, 90, 140)])
    oo, hh, ll = np.concatenate([[100.0], cc[:-1]]), cc + 0.5, cc - 0.5
    res_l = planstudy.sim_trail(oo, hh, ll, cc, 10, 1, 3.0, 1.0)
    assert res_l[0] == 1 and abs(res_l[1] - (127.0 / oo[11] - 1) * 100) < 1e-6, res_l
    res_s = planstudy.sim_trail(oo, hh, ll, cc, 100, -1, 3.0, 1.0)
    assert res_s[0] == 2 and abs(res_s[1] - (1 - cc[160] / oo[101]) * 100) < 1e-6, res_s
    assert planstudy.sim_trail(oo, hh, ll, cc, n_ - 30, 1, 3.0, 1.0) is None                      # sin sesiones suficientes: no se opera
    bars = {f'S{j}': walk(1500, seed=700 + j, t0=1500000000) for j in range(24)}
    ev = planstudy.analyze_plan(bars, ks=(24,), workers=1)
    assert set(ev['pooled']) == set(planstudy.VARIANTS) and ev['meta']['universe'] == 24
    for v in planstudy.VARIANTS:
        s = ev['pooled'][v]['K24']['todos']
        assert s['n'] > 1500 and abs(s['ex']) < 0.6 and abs(s['t']) < 6, (v, s)           # sin ventaja posible: el exceso queda cerca de 0
        assert (0 < s['tgt'] < 100 or v.startswith('trail')) and abs(s['tgt'] + s['stp'] + s['tmo'] - 100) < 0.05      # con stop que persigue no hay objetivo
    txt = planstudy.report(ev)
    assert 'PLAN DEL OPERADOR' in txt and 'PLAN CLÁSICO' in txt and 'FIJO' in txt and 'ganan' in txt
    print(f'ok · estudio del plan: simulador idéntico a plan.outcome ({tot} casos) y sin ventaja sobre paseos aleatorios')


def _walk_ar(n, seed, theta):
    """Paseo con reversión a 5 sesiones de intensidad theta (0 = paseo aleatorio puro)."""
    rng = np.random.default_rng(seed)
    sig = 0.015
    e = rng.normal(0, sig, n)
    r = np.zeros(n)
    for t in range(n):
        r[t] = e[t] - theta * r[max(t - 5, 0):t].sum()
    c = 100 * np.exp(np.cumsum(r))
    o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, sig / 3, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sig / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sig / 2, n)))
    return {'t': 1500000000 + 86400 * np.arange(n), 'o': o, 'h': h, 'l': l, 'c': c, 'v': np.random.default_rng(seed + 1).integers(5e5, 2e6, n).astype(float)}


def test_ml_pipeline():
    """El aprendizaje automático fuera de muestra encuentra un efecto plantado (reversión) y no inventa nada en paseos aleatorios."""
    try:
        import pandas, sklearn                                            # noqa: F401
    except ImportError:
        print('omitida · aprendizaje automático: faltan pandas o scikit-learn')
        return
    fast = dict(max_iter=40, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=10.0, max_bins=32, random_state=0)
    kw = dict(workers=1, test_years=[2020, 2021], k=60, groups=('ctx',), min_train=3000, min_rows=2000, params=fast)
    ev = ml.analyze({f'S{j}': _walk_ar(1500, 100 + j, 0.08) for j in range(60)}, **kw)
    s = ev['modelos']['ctx']['todo']
    assert s['ic'] > 0.08 and s['ic_t'] > 8 and s['top'] > 0.4 and ev['base']['reversión 5 sesiones (−r5)']['todo']['ic_t'] > 8, (s, ev['base'])
    assert abs(ev['base']['azar']['todo']['ic_t']) < 3.5
    ev0 = ml.analyze({f'S{j}': _walk_ar(1500, 300 + j, 0.0) for j in range(60)}, **kw)
    s0 = ev0['modelos']['ctx']['todo']
    assert abs(s0['ic']) < 0.02 and abs(s0['ic_t']) < 3.5, s0                                  # sin efecto no hay IC: el modelo no se inventa nada
    assert abs(ml.nw_t(np.random.default_rng(0).normal(0.3, 1, 400)) - 0.3 * 20) < 3 and 'años de desarrollo' in ml.report(ev)
    g2 = {'p': dict(fast, max_iter=20), 'q': dict(fast, max_leaf_nodes=4)}
    ev2 = ml.analyze({f'S{j}': _walk_ar(1500, 100 + j, 0.08) for j in range(60)}, **dict(kw, params=None, grid=g2, dev_end=2020, train_stride=2))
    assert ev2['seleccion']['elegida'] in g2 and ev2['meta']['dev_years'] == [2020] and ev2['meta']['final_years'] == [2021] and ev2['modelos']['ctx']['final']['ic'] > 0.08 and 'AÑOS FINALES' in ml.report(ev2)
    print(f'ok · aprendizaje automático: reversión plantada detectada fuera de muestra (IC {s["ic"]:+.3f}, t {s["ic_t"]:.0f}) y nada en paseos aleatorios (IC {s0["ic"]:+.3f})')


def test_ml_features_causal():
    """Las variables de contexto, los niveles, el volumen relativo y las señales de una barra no cambian al quitar las barras posteriores."""
    try:
        import pandas, sklearn                                            # noqa: F401
    except ImportError:
        print('omitida · variables del aprendizaje automático: faltan pandas o scikit-learn')
        return
    b = _walk_ar(700, 77, 0.0)
    cut = 500
    full = ml._work(('X', b))[1]
    part = ml._work(('X', {k: v[:cut] for k, v in b.items()}))[1]
    for key in ('ctx', 'lvl', 'volraw'):
        a, p = np.asarray(full[key])[:cut - 20], np.asarray(part[key])[:cut - 20]
        assert np.allclose(a, p, equal_nan=True, rtol=1e-5, atol=1e-6), (key, np.nanmax(np.abs(a - p)))
    ef = {tuple(r) for r in full['ev'] if r[0] < cut - 20}
    ep = {tuple(r) for r in part['ev'] if r[0] < cut - 20}
    assert ef == ep, (len(ef), len(ep), list(ef ^ ep)[:5])
    print(f'ok · variables del aprendizaje automático: causales ({len(ef)} señales y {cut - 20} barras comparadas)')


def test_confluence_causal():
    """La confluencia de una señal solo cuenta señales de su misma sesión y las anteriores: añadir señales posteriores no la cambia."""
    sym = np.array([0, 0, 0, 0, 1, 0])
    bar = np.array([10, 12, 12, 30, 12, 13])
    pid = np.array([0, 1, 2, 0, 1, 1])
    d = np.array([1, 1, 1, 1, 1, -1])
    c = context.confluence(sym, bar, pid, d, 2)
    assert c[0] == 0, c                                                   # la 12 y la 12 son posteriores a la 10
    assert c[1] == 2 and c[2] == 2, c                                     # la 10 y la otra señal de la sesión 12 (otro patrón, mismo sentido)
    assert c[3] == 0 and c[4] == 0 and c[5] == 0, c                       # lejos / otro valor / sentido contrario
    keep = np.arange(len(c)) < 3
    c2 = context.confluence(sym[keep], bar[keep], pid[keep], d[keep], 2)
    assert (c2 == c[keep]).all()
    rng = np.random.default_rng(1)
    n = 400
    sy, ba, pi_, di = rng.integers(0, 5, n), rng.integers(0, 200, n), rng.integers(0, 6, n), rng.choice([-1, 1], n)
    full = context.confluence(sy, ba, pi_, di, 5)
    early = ba <= 100
    assert (context.confluence(sy[early], ba[early], pi_[early], di[early], 5) == full[early]).all()      # recortar el futuro no cambia el pasado
    print('ok · confluencia: solo mira hacia atrás')


def test_context_experiment():
    """Los filtros de contexto: un efecto plantado solo para «velas alcistas con volumen» aparece justo ahí y el umbral del mejor de N no lo confunde con ruido; el meta-etiquetado lo aprende."""
    try:
        import pandas, sklearn                                            # noqa: F401
    except ImportError:
        print('omitida · contexto y confluencia: faltan pandas o scikit-learn')
        return
    bars = {f'S{j}': _walk_ar(1500, 500 + j, 0.0) for j in range(50)}
    work = {s: ml._work((s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}))[1] for s, b in bars.items()}
    D = ml.Data(work, k=50)
    E = context.events(D)
    assert (E['row'] >= 0).sum() > 5000 and E['conf'].max() >= 1
    row = E['row']
    r = np.where(row >= 0, row, 0)
    F = context.filters(D, row, E['conf'], E['dir'])
    assert set(F) == set(context.FILTERS) and all(v.dtype == bool for v in F.values()) and not F['volumen'][row < 0].any()
    # efecto plantado: +2 % de exceso en las velas alcistas con volumen alto (se suma al rendimiento relativo de esas filas)
    plant = (E['fam'] == 0) & (E['dir'] == 1) & F['volumen'] & (row >= 0)
    assert plant.sum() > 100
    D.rel[row[plant]] += 3.0
    ex = context.experiment(D, E, reps=15, dev_end=2019)
    g = ex['grupos']['velas alcistas']['filtros']
    assert g['volumen']['cumple']['t'] > 4 and g['volumen']['cumple_final']['mean'] > 0.5 and g['volumen']['cumple']['mean'] > g['volumen']['no_cumple']['mean'] + 0.5, g['volumen']
    other = ex['grupos']['figuras bajistas']['filtros']
    assert all(abs(v['cumple']['t']) < ex['umbral']['p99'] + 2 for v in other.values() if 't' in v['cumple']), (other, ex['umbral'])
    assert ex['umbral']['p95'] > 2 and ex['umbral']['combinaciones'] == 12 * 13
    # meta-etiquetado: el modelo aprende dónde está el efecto plantado
    fast = {'A': dict(max_iter=40, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=100, l2_regularization=10.0, max_bins=128, random_state=0)}
    mt = context.meta(D, E, [2020, 2021], dev_end=2020, grid=fast, min_train=2000)
    f_ = mt['modelo']['final']
    assert mt['configuracion']['elegida'] == 'A' and f_['mejor'] > f_['todas'] + 0.3 and f_['ic'] > 0.03, f_
    assert mt['contexto']['final']['ic'] > 0.03 and mt['azar']['final']['ic'] < 0.03 and 'AÑOS' not in context.report({'meta': {'universe': 50, 'señales': 1, 'k': 50, 'from': 'a', 'to': 'b', 'horizonte': 10, 'desarrollo_hasta': 2022}, 'experimento': ex, 'meta_etiquetado': mt})[:5]
    print(f'ok · contexto y confluencia: efecto plantado en «velas alcistas con volumen» detectado (t {g["volumen"]["cumple"]["t"]:.0f}) y aprendido por el meta-etiquetado (IC {f_["ic"]:+.2f})')


def test_outcome_counts_real_result():
    """Gana quien termina con ganancia neta, llegue o no a ningún nivel. Plan clásico (objetivo) y plan del operador (asegurar +4 % y dejar correr)."""
    o = np.array([100, 100, 101, 104, 106, 108, 109.0]); h = np.array([101, 102, 104, 106, 108, 109.5, 111.0]); l = np.array([99, 99, 100, 103, 105, 107, 108.0]); c = np.array([100, 101, 103, 105.5, 107, 109, 110.5])
    F = dict(o=o, h=h, l=l, c=c, atr=np.full(7, 1.0))
    a = plan.outcome(F, 0, 1, 95, 110)
    assert a['res'] == 'target' and a['bars'] == 6 and abs(a['ret'] - 10) < 1e-6 and abs(a['net'] - (10 - plan.COST)) < 1e-6 and abs(a['r'] - 2.0) < 1e-6 and a['mfe'] >= 10, a
    b = plan.outcome(F, 0, 1, 95, 120, maxb=4)                                                       # objetivo +20 %: no llega, cierra a las 4 sesiones con +7 %
    assert b['res'] == 'timeout' and abs(b['ret'] - 7.0) < 1e-6 and b['net'] > 0 and abs(b['prog'] - 0.4) < 1e-6, b          # prog = 8 % de recorrido máximo sobre el 20 % del objetivo
    d = plan.outcome(F, 0, 1, 99.5, 110)                                                              # stop pegado: salta en la primera sesión (mínimo 99) y pierde
    assert d['res'] == 'stop' and d['net'] < 0 and d['r'] < 0 and d['xi'] == 1, d
    assert plan.outcome(F, 0, 1, 100, 110)['res'] == 'na'                                             # abre en el stop: no se opera
    # corto: espejo del largo
    sh = plan.outcome(dict(o=200 - o, h=200 - l, l=200 - h, c=200 - c), 0, -1, 105, 90)
    assert sh['res'] == 'target' and abs(sh['ret'] - 10) < 0.6 and sh['r'] > 1.5, sh
    # plan del operador: entrada 100 (apertura de la barra 1), stop 95; la barra 3 cierra en 105,5 (≥ +4 %): el stop sube a 104 y persigue a 3 ATR (ATR 1 → 102,5 con el cierre de 105,5)
    lk = plan.outcome_lock(F, 0, 1, 95.0, maxb=6)
    assert lk['locked'] and lk['res'] == 'target' and lk['net'] > 3.5, lk                          # llega al final con lo asegurado (cierre 110,5)
    F2 = dict(o=o, h=h, l=l, c=c, atr=np.full(7, 1.0))
    l2 = l.copy(); l2[4] = 103.5                                                                      # la barra 4 toca 103,5: el stop asegurado (104) salta y sale a 104 = +4 %
    lk2 = plan.outcome_lock(dict(F2, l=l2), 0, 1, 95.0)
    assert lk2['locked'] and lk2['res'] == 'target' and abs(lk2['ret'] - 4.0) < 1e-6 and lk2['bars'] == 4, lk2
    l3 = l.copy(); l3[1] = 94.0                                                                       # antes de asegurar toca el stop inicial: pierde
    lk3 = plan.outcome_lock(dict(F2, l=l3), 0, 1, 95.0)
    assert lk3['res'] == 'stop' and not lk3['locked'] and lk3['ret'] < 0, lk3
    print('ok · resultado de cada operación: ganancia neta, múltiplo del riesgo, recorrido máximo; plan clásico y plan del operador (asegurar +4 % y dejar correr)')


def test_group_trades():
    """Una operación por valor: velas y figuras que saltan a la vez no se cuentan dos veces, las contrarias avisan del conflicto y tras cerrar se abre otra."""
    def it(key, fam, i, d, res, net, xi=None, q=None, label=None):
        return {'id': f'{key}@{i}', 'key': key, 'label': label or key, 'fam': fam, 'i': i, 'dir': d, 'state': 'done', 'q': q, 'out': {'res': res, 'net': net, 'r': None if net is None else net / 2.0, 'xi': xi, 'ret_m': net, 'mfe': 1.0, 'prog': 0.5}}
    items = [it('hammer', 'velas', 10, 1, 'target', 5.0, xi=18), it('double_bottom', 'figuras', 10, 1, 'target', 7.0, xi=25, q=80),         # misma barra: manda la figura
             it('bull_engulf', 'velas', 12, 1, 'target', 3.0, xi=16),                                                                              # con la operación abierta: confirma
             it('shooting_star', 'velas', 14, -1, 'stop', -2.0, xi=15),                                                                            # contraria
             it('marubozu_bull', 'velas', 30, 1, 'timeout', 4.0, xi=45),                                                                           # operación nueva: cerrada por tiempo con ganancia
             it('hs_top', 'figuras', 31, -1, 'na', None),                                                                                          # fuera de niveles: no abre nada
             it('donch55_dn', 'rupturas', 60, -1, 'stop', -3.0, xi=62)]
    tr = lab.group_trades(items, 100)
    assert [t['key'] for t in tr['list']] == ['double_bottom', 'marubozu_bull', 'donch55_dn'], [t['key'] for t in tr['list']]
    first = tr['list'][0]
    assert first['conf'] == ['hammer', 'bull_engulf'] or sorted(first['conf']) == ['bull_engulf', 'hammer'], first
    assert first['contra'] == ['shooting_star'] and tr['conflicts'] == 1
    roles = {x['key']: x['trade']['role'] for x in items}
    assert roles == {'hammer': 'confirma', 'double_bottom': 'lider', 'bull_engulf': 'confirma', 'shooting_star': 'contraria', 'marubozu_bull': 'lider', 'hs_top': 'na', 'donch55_dn': 'lider'}, roles
    A = tr['all']
    assert A['n'] == 3 and A['won'] == 2 and A['lost'] == 1 and A['tmo'] == 1 and A['tmo_won'] == 1 and abs(A['mean_net'] - (7.0 + 4.0 - 3.0) / 3) < 1e-3, A
    print('ok · operaciones únicas: 7 señales solapadas = 3 operaciones, con las contrarias avisadas y el cierre por tiempo con ganancia contado como ganado')


def test_combos_unique_valuedays():
    """Coincidencias: un valor-día con varias señales cuenta una sola vez y se separa solo-velas / solo-estructura / juntas / contradictorias."""
    work = {f'S{j}': study._work((f'S{j}', walk(700, seed=60 + j, t0=1500000000)))[1] for j in range(8)}
    for j, w in enumerate(work.values()):
        base = np.arange(100 + j, 650, 7)
        w['idx'] = {'hammer': base, 'double_bottom': base[::2], 'bull_engulf': base[::3], 'shooting_star': base[1::5], 'consenso_alcista_2': base[::4]}
        w['path'] = np.full_like(w['path'], np.nan)
    P = study.Panel(work, ks=(8,))
    cb = study.combos(P, (8,))['K8']
    all_up = cb['alcistas']['todas (cada valor-día una vez)']['n'] + 0
    rows = set(np.concatenate([P.ev['hammer'], P.ev['double_bottom'], P.ev['bull_engulf']]).tolist())
    contra = set(P.ev['shooting_star'].tolist())
    n_unique = len([r for r in rows if r not in contra and np.isfinite(P.ret[('long', 10)][r]) and np.isfinite(P.bday[(8, 'long', 10)][P.dcode[r]])])
    assert all_up == n_unique, (all_up, n_unique)
    a = cb['alcistas']
    assert a['solo velas']['n'] + a['solo estructura']['n'] + a['velas y estructura juntas']['n'] == all_up and a['velas y estructura juntas']['n'] > 0 and a['solo velas']['n'] > 0
    assert cb['valor_dias']['con_varias'] > 0 and 'long' in cb['contradictorias']
    txt = study.report({'meta': {'universe': 8, 'bars': 1, 'from': 'a', 'to': 'b', 'cost': 0.15, 'primary': 10, 'ks': [8], 'thr': {'K8': {'p95': 3.0, 'p99': 4.0, 'bars': 1}}, 'tested': {'K8': {'n': 0, 'above': [], 'above_positive': []}}},
                        'patterns': {}, 'combos': {'K8': cb}})
    assert 'coincidencias' in txt and 'solo velas' in txt
    print(f'ok · coincidencias: {all_up} valor-días alcistas contados una vez ({a["solo velas"]["n"]} solo velas, {a["solo estructura"]["n"]} solo estructura, {a["velas y estructura juntas"]["n"]} juntas)')


def test_search_sim():
    """La salida del operador vectorizada para todas las barras coincide con plan.outcome_lock y con el stop que persigue del estudio del plan; el cierre a 10 sesiones es el del panel."""
    from vp import plan as P_
    rng = np.random.default_rng(3)
    tot = 0
    for seed in range(3):
        b = walk(900, seed=70 + seed, sig=0.02)
        F = ta.prepare(b)
        o, h, l, c, atr = F['o'], F['h'], F['l'], F['c'], F['atr']
        for d in (1, -1):
            for stop_atr, lock, trail in ((1.5, 4.0, 3.0), (2.5, 6.0, 4.0)):
                net = search.sim_all(o, h, l, c, atr, d, stop_atr, lock, trail, 60)
                for _ in range(60):
                    i = int(rng.integers(60, 820))
                    entry = float(o[i + 1])
                    r = P_.outcome_lock(F, i, d, entry - d * stop_atr * float(atr[i]), lock, trail, 60)
                    assert r['res'] != 'na'
                    if r['res'] == 'open':
                        assert np.isnan(net[i])
                        continue
                    tot += 1
                    assert abs(net[i] - r['net']) < 2e-2, (seed, d, i, net[i], r['net'], r['res'])
            tr = search.sim_all(o, h, l, c, atr, d, 5.0, None, 5.0, 60)
            for _ in range(40):
                i = int(rng.integers(60, 820))
                s1 = planstudy.sim_trail(o, h, l, c, i, d, 5.0, float(atr[i]), 60)
                if s1 is None:
                    continue
                tot += 1
                assert abs(tr[i] - (s1[1] - 0.15)) < 1e-6, (d, i, tr[i], s1)
            h10 = search.hold_all(o, c, d, 10)
            i = 100
            assert abs(h10[i] - (d * (c[i + 10] / o[i + 1] - 1) * 100 - 0.15)) < 1e-9
    print(f'ok · búsqueda: la salida del operador por barras coincide con la simulación de la web ({tot} casos)')


def test_search_procedure():
    """La búsqueda hacia delante encuentra una reversión plantada (y la elige con el pasado, no con el año que mide) y no inventa nada en paseos aleatorios; el nulo desplazado no la ve."""
    kw = dict(workers=1, k=40, reps=8, min_day=15, min_combo=600, test_years=[2019, 2020, 2021])
    bars = {f'S{j}': _walk_ar(1500, 800 + j, 0.25) for j in range(40)}
    ev = search.analyze(bars, **kw)
    w = ev['walk_forward']['contexto']
    assert w['n'] > 2000 and w['exceso'] > 0.25 and w['t'] > 4, w
    under, over = {'cae5', 'rsi2_bajo', 'bajo_media20', 'rsi14_bajo'}, {'sube5', 'rsi2_alto', 'sobre_media20', 'rsi14_alto'}       # reversión: largo tras caer, corto tras subir
    assert len(w['años']) == 3 and all(r['train_exceso'] > 0 and (set(r['regla']) & (under if r['sentido'] == 'largo' else over)) for r in w['años']), w['años']
    nl = ev['nulo']['contexto']
    assert nl['t_p95'] < w['t'] and nl['p_t'] <= 0.2, (nl, w['t'])
    assert 'FUERA DE MUESTRA' in search.report(ev) and ev['meta']['pruebas'] == ev['meta']['combinaciones'] * 2 * len(search.EXITS)
    ev0 = search.analyze({f'S{j}': _walk_ar(1500, 900 + j, 0.0) for j in range(40)}, **kw)
    w0 = ev0['walk_forward']['todo']
    assert abs(w0['t']) < 4 and abs(w0['exceso']) < 0.8, w0                         # sin efecto: lo que se «encuentra» buscando entre cientos de reglas no se sostiene fuera de muestra
    print(f'ok · búsqueda hacia delante: reversión plantada (exceso {w["exceso"]:+.2f} %, t {w["t"]:.0f}) y sin ventaja en paseos aleatorios (t {w0["t"]:+.1f})')


def test_search_no_lookahead():
    """La regla elegida para un año de prueba solo depende de datos anteriores a ese año menos 60 sesiones: cambiar lo que pasa en el año de prueba (o después) no la cambia."""
    bars = {f'S{j}': _walk_ar(1500, 1200 + j, 0.25) for j in range(25)}
    work = {}
    for s_, b in bars.items():
        work[s_] = search._work((s_, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}))[1]
    D = ml.Data(work, k=25)
    P = search.Panel(D, work, min_day=10, min_combo=500)
    T = P.tables(P.rel, absolute=True)
    yrs = [2019, 2020, 2021]
    base = search.walk_forward(P, T, P.sub['todo'], yrs)
    T2 = {k: v.copy() for k, v in T.items()}
    mod = P.byear >= 2020                                                    # se cambia todo lo que ocurre desde 2020 (resultados y número de operaciones)
    T2['S'][..., mod] = -T2['S'][..., mod] * 3.0 + 1.0
    T2['SA'][..., mod] *= 0.5
    alt = search.walk_forward(P, T2, P.sub['todo'], yrs)
    f = lambda r: (r['regla'], r['sentido'], r['salida'], r['train_n'], r['train_t'])
    assert f(base['años'][0]) == f(alt['años'][0]), (base['años'][0], alt['años'][0])                 # 2019 se elige con datos de antes de 2019
    assert f(base['años'][1]) == f(alt['años'][1]), (base['años'][1], alt['años'][1])                 # 2020 se elige con datos de antes de 2020 (menos 60 sesiones): lo que cambia desde 2020 no cuenta
    assert base['años'][1]['exceso'] != alt['años'][1]['exceso']                                    # y lo que se mide en 2020 sí cambia (la prueba de verdad toca los resultados del año)
    d0 = P.ystart[2020]
    tr = ((np.arange(P.nb) + 1) * search.BLOCK <= d0 - search.PURGE)
    assert P.byear[tr].max() < 2020 and tr.sum() > 20 and (np.flatnonzero(tr).max() + 1) * search.BLOCK <= d0 - search.PURGE   # el entrenamiento acaba al menos PURGE sesiones antes
    print('ok · búsqueda: la elección de cada año no mira ni el año de prueba ni los 60 días anteriores')


def test_universe_changes_and_survivorship():
    """La tabla de miembros del S&P 500 da la fecha de incorporación de cada valor; las barras anteriores salen del universo; una regla de un solo régimen no es elegible por muy alto que sea su t."""
    try:
        import pandas as pd
    except ImportError:
        print('omitida · fechas de incorporación al S&P 500: falta pandas')
        return
    members = pd.DataFrame({'Symbol': ['ABC', 'BRK.B', 'DEF', 'GHI', 'JKL'], 'Security': list('abcde'), 'GICS Sector': list('vwxyz'),
                            'Date added': ['2023-01-10', '2010-02-16', '2021-09-20', 'n/a', '1976-06-30 (1957-03-04)']})
    nav = pd.DataFrame({'vte S&P 500 companies': ['x'], 'vte S&P 500 companies.1': ['y']})
    got = universe.parse_sp500_join_dates([nav, members])
    assert got == {'ABC': '2023-01-10', 'BRK-B': '2010-02-16', 'DEF': '2021-09-20', 'JKL': '1976-06-30'}, got        # sin fecha legible → se ignora; punto → guion
    assert universe.parse_sp500_join_dates(members) == got
    cols = pd.MultiIndex.from_tuples([('Symbol', 'Symbol'), ('Date first added', 'Date first added')])
    assert universe.parse_sp500_join_dates(pd.DataFrame([['ABC', '2020-03-23']], columns=cols)) == {'ABC': '2020-03-23'}
    try:
        universe.parse_sp500_join_dates([nav])
        raise AssertionError('debía fallar sin tabla de miembros')
    except ValueError as exc:
        assert 'ninguna tabla' in str(exc) and 'vte' in str(exc)
    work = {'ABC': {'t': np.arange(18000, 18010), 'dv': np.ones(10)}, 'DEF': {'t': np.arange(18000, 18010), 'dv': np.ones(10)}, 'GHI': {'t': np.arange(18000, 18010), 'dv': np.ones(10)}}
    n = search.apply_join(work, {'ABC': str(np.datetime64(18004, 'D')), 'DEF': '2000-01-01'})
    assert n == 4 and np.isnan(work['ABC']['dv'][:4]).all() and np.isfinite(work['ABC']['dv'][4:]).all() and np.isfinite(work['DEF']['dv']).all() and np.isfinite(work['GHI']['dv']).all(), (n, work['ABC']['dv'])
    # cobertura: una regla que solo opera en 3 de 60 periodos no se elige aunque su t sea enorme; la estable sí
    from types import SimpleNamespace
    nb = 60
    NE_ = len(search.EXITS)
    S = np.zeros((3, 2, NE_, nb)); N = np.zeros((3, 2, NE_, nb))
    N[0] = 400.0
    N[1, :, :, :3] = 300.0
    S[1, :, :, :3] = 300.0 * 2.0
    rng = np.random.default_rng(0)
    N[2] = 40.0
    S[2] = 40.0 * 0.4 + rng.normal(0, 25, S[2].shape)
    Pf = SimpleNamespace(G=[search.block_groups(nb, 1) for _ in range(NE_)])
    sub = np.array([False, True, True])
    tr = np.ones(nb, bool)
    n_, m_, t_, cov = search.stats_by_exit(Pf, {'S': S, 'N': N}, tr)
    assert cov[1].max() < 0.1 and cov[2].min() > 0.99
    c, d, e, *_ = search.pick(Pf, {'S': S, 'N': N}, sub, tr, min_n=500)
    assert c == 2, (c, t_[1].max(), t_[2].max())                                                   # la de un solo régimen (regla 1) queda fuera
    print('ok · fechas de incorporación al S&P 500; barras anteriores quitadas; reglas de un solo régimen no elegibles')


def test_search_confirm():
    """La confirmación evalúa reglas fijas: con reversión plantada, «largo tras caer» gana y «largo tras subir» pierde, con p-valores pequeños; en paseos aleatorios ninguna se distingue del azar."""
    rules = ({'nombre': 'T1', 'regla': ['cae5'], 'sentido': 'largo', 'salida': 'h10'}, {'nombre': 'T2', 'regla': ['sube5'], 'sentido': 'largo', 'salida': 'h10'},
             {'nombre': 'T3', 'regla': ['cae5', 'bajo_200'], 'sentido': 'largo', 'salida': 'op4b'})
    kw = dict(workers=1, k=40, reps=80, min_day=15, rules=rules)
    ev = search.confirm({f'S{j}': _walk_ar(1500, 1500 + j, 0.25) for j in range(40)}, **kw)
    r1, r2, r3 = ev['reglas']
    assert r1['exceso'] > 0.8 and r1['t'] > 5 and r1['p_exceso'] <= 0.05 and r1['p_t'] <= 0.05 and r1['n'] > 2000 and sum(v > 0 for v in r1['por_año'].values()) >= len(r1['por_año']) - 1, r1
    assert r2['exceso'] < -0.5 and r2['t'] < -4 and r2['p_exceso'] > 0.9, r2
    assert r3['n'] > 500 and r3['exceso'] > 0.3 and 'neto' in r3 and 'ganan' in r3, r3
    txt = search.confirm_report(ev, 'prueba')
    assert 'T1' in txt and 'caída a 5 sesiones' in txt and 'p exceso' in txt and 'años positivos' in txt
    ev0 = search.confirm({f'S{j}': _walk_ar(1500, 1700 + j, 0.0) for j in range(40)}, **kw)
    for r in ev0['reglas']:
        assert abs(r['t']) < 4 and r['p_exceso'] > 0.01 and r['p_exceso'] < 0.99, r
    assert all(r['nombre'] in {c['nombre'] for c in search.CANDIDATAS} or r['nombre'].startswith('T') for r in ev0['reglas']) and len(search.CANDIDATAS) == 5
    print(f'ok · confirmación: reglas fijas (T1 exceso {r1["exceso"]:+.2f} %, t {r1["t"]:.0f}; T2 {r2["exceso"]:+.2f} %) y nada en paseos aleatorios')


def test_screen():
    """El cribador del último cierre da las mismas acciones que el panel completo en esa fecha (sin mirar el futuro), el plan sale del cierre y el ATR, y el registro no repite días."""
    bars = {f'S{j}': _walk_ar(1500, 2000 + j, 0.0) for j in range(40)}
    rules = search.CANDIDATAS
    work = {s: ml._work((s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}))[1] for s, b in bars.items()}
    Df = ml.Data(work, k=40)
    ff = screen._flags(Df, rules)
    total = 0
    last_ev = None
    for cut in (800, 950, 1100, 1250, 1400):
        sub = {s: {a: v[:cut + 1] for a, v in b.items()} for s, b in bars.items()}
        ev = screen.screen(sub, k=40, rules=rules)
        rows = Df.bar == cut                                                 # el mismo día en el panel con todo el futuro
        assert ev['meta']['asof'] == str(Df.date[Df.dcode[np.flatnonzero(rows)[0]]]) and ev['meta']['en_universo'] == int(rows.sum()), ev['meta']
        for r in rules:
            want = {Df.syms[j] for j in Df.sym[ff[r['nombre']] & rows]}
            got = {p['sym'] for p in ev['picks'] if r['nombre'] in p['reglas']}
            assert got == want, (cut, r['nombre'], got ^ want)
            assert ev['resumen'][r['nombre']]['hoy'] == len(want)
            total += len(want)
        for p in ev['picks']:
            F = ta.prepare(bars[p['sym']])
            c, a = float(F['c'][cut]), float(F['atr'][cut])
            assert abs(p['close'] - c) < 1e-3 and abs(p['atr'] - a) < 1e-3 * c and abs(p['plan']['stop'] - (c - 2.5 * a)) < 2e-3 * c and abs(p['plan']['asegura4'] - c * 1.04) < 1e-3 * c, (p, c, a)
            assert 0 <= p['mom121'] <= 1 and set(p['nuevas']) <= set(p['reglas']) and p['bajistas'] >= 0
        last_ev = ev if ev['picks'] else last_ev
    assert total >= 3 and last_ev, total                                       # el control no es vacío: hubo acciones que cumplían alguna regla
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), 'h.json')
    h1 = screen.update_history(path, last_ev)
    json.dump(h1, open(path, 'w'))
    h2 = screen.update_history(path, last_ev)
    assert len(h1) == 1 and len(h2) == 1 and h2[0]['date'] == last_ev['meta']['asof'] and set(h2[0]['reglas']) == {r['nombre'] for r in rules}
    ev2 = dict(last_ev, meta=dict(last_ev['meta'], asof='2099-01-01'))
    json.dump(h2, open(path, 'w'))
    h3 = screen.update_history(path, ev2)
    assert [h['date'] for h in h3] == [last_ev['meta']['asof'], '2099-01-01']
    print(f'ok · cribador: coincide con el panel completo en 5 fechas ({total} acciones-regla) y el registro no repite días')


def test_web_impulso_constants():
    """La web copia las condiciones y las salidas de la regla de impulso: tienen que ser exactamente las de vp/search.py, y las dos partes (figuras / impulso) tienen sus propias rutas y pestañas."""
    import re
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'index.html'), encoding='utf-8').read()
    cond = json.loads(re.search(r'const COND = (\{.*?\});\n', html).group(1))
    exits = json.loads(re.search(r'const EXITS = (\{.*?\});\n', html).group(1))
    assert cond == search.DESC and exits == search.EXIT_TXT, (set(cond) ^ set(search.DESC), set(exits) ^ set(search.EXIT_TXT))
    for needle in ("impulso: 'imp'", "'impulso-evidencia': 'impev'", "'impulso-guia': 'impguide'", "['imp', 'Hoy'], ['impev', 'Evidencia'], ['impguide', 'Guía']", "['radar', 'Radar'], ['chart', 'Gráfico'], ['ev', 'Evidencia'], ['guide', 'Guía']", 'Parte 2', 'Parte 1'):
        assert needle in html, needle
    for f in ('impulso.json', 'search.json', 'search_confirm.json', 'search_confirm_smallmid.json'):
        assert f"'{f}'" in html, f
    print('ok · web: condiciones y salidas de la regla de impulso idénticas a las del código; dos partes con rutas y pestañas propias')


ALL = [test_causality, test_candles_planted, test_breakouts_planted, test_figures_planted, test_forming_figures, test_spent_target_and_first_break, test_watchlist, test_catalog, test_plan, test_plan_rules, test_min_stop_and_forming, test_shapes_geometry,
       test_data_parse, test_data_retries, test_lab_json, test_study_mechanics, test_controls_detect_planted_momentum, test_study_no_edge_on_random_walk, test_plan_study, test_outcome_counts_real_result, test_group_trades, test_combos_unique_valuedays, test_ml_pipeline, test_ml_features_causal, test_confluence_causal, test_context_experiment, test_search_sim, test_search_procedure, test_search_no_lookahead, test_search_confirm, test_universe_changes_and_survivorship, test_screen, test_web_impulso_constants]

if __name__ == '__main__':
    only = sys.argv[1:]                                                      # python tests/test_vp.py screen → solo las pruebas cuyo nombre contiene «screen»
    for fn in ALL:
        if not only or any(w in fn.__name__ for w in only):
            fn()
    print('TODO OK')
