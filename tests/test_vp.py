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
from vp import breakouts, candles, data, engine, lab, shapes, study, ta
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
    for nm, (tr, tail) in cases.items():
        F = ta.prepare(candle(tr, tail))
        assert candles.detect(F)[nm][-1], f'{nm}: no se detecta en la barra final'
    print(f'ok · velas: {len(cases)} patrones plantados se detectan en su barra')


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
    assert rr['stop'] < rr['entry'] < rr['t1'] < rr['target'] and p['dir'] == 1
    assert abs(rr['target'] - (neck + (neck - low))) < 0.02, (rr, neck, low)          # objetivo = cuello + altura de la figura
    assert rr['stop'] < low
    assert p['out']['res'] == 'target' and p['out']['ret'] > 0 and p['out']['xi'] > p['i']
    res2, dbs2 = run([(56, 70)])
    assert dbs2 and dbs2[-1]['out']['res'] == 'stop' and dbs2[-1]['out']['ret'] < 0
    # invariantes de todos los patrones del valor: stop a ≥ 1 ATR, T1 intermedio, lados correctos
    for res_, b_ in ((res, None), (res2, None)):
        for q in res_['patterns']:
            r_ = q['plan']
            if r_ is None:
                continue
            assert (r_['stop'] < r_['entry'] < r_['t1'] < r_['target']) if q['dir'] > 0 else (r_['target'] < r_['t1'] < r_['entry'] < r_['stop']), (q['key'], r_)
    s = json.dumps(res)
    assert all(z['touches'] >= 2 and z['lo'] < z['p'] < z['hi'] for z in res['zones'])
    assert len(res['bars']['c']) == res['n'] == len(res['sma50']) and all(0 <= pp['i'] < res['n'] for pp in res['patterns'])
    print(f'ok · plan: objetivo = cuello + altura, stop bajo la figura, T1 intermedio, resultado simulado y JSON de {len(s) // 1024} KB')


def test_min_stop_and_forming():
    b, _ = plant_fig('rising_wedge')
    F = ta.prepare(b)
    res = lab.analyze('T', b, nb=300)
    atr = {p['i']: float(F['atr'][p['i'] + len(F['c']) - 300]) for p in res['patterns'] if p['plan']}
    for p in res['patterns']:
        if p['plan']:
            assert abs(p['plan']['entry'] - p['plan']['stop']) >= 0.99 * atr[p['i']] - 1e-6, (p['key'], p['plan'])
    # triángulo simétrico sin romper: figura en formación con escenario alcista y bajista
    an = [(0, 100), (10, 110), (20, 102), (30, 108), (40, 104), (48, 106.5), (54, 105)]
    y0 = an[0][1]; an = [(x, y0 + (y - y0) * 0.5) for x, y in an]
    xs, ys = np.array([a[0] for a in an]), np.array([a[1] for a in an])
    b = plant_closes(np.array(quiet(260, 100) + list(np.interp(np.arange(xs[0], xs[-1] + 1), xs, ys))), 0.5)
    res = lab.analyze('T', b, nb=300)
    fm = [p for p in res['patterns'] if p['state'] == 'forming']
    assert len(fm) == 1 and fm[0]['key'] == 'forming_sym_tri' and fm[0]['dir'] == 0 and set(fm[0]['sc']) == {'up', 'dn'}
    up, dn = fm[0]['sc']['up'], fm[0]['sc']['dn']
    last = res['summary']['close']
    assert dn['stop'] > dn['entry'] > dn['t1'] > dn['target'] and up['stop'] < up['entry'] < up['t1'] < up['target'] and dn['entry'] < last < up['entry']
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
    assert data.parse({'chart': {'result': None, 'error': {'code': 'Not Found'}}}) is None
    assert data.span('3y', 1.7e9).startswith('period1=') and data.span('6mo') == 'range=6mo'
    last = int(datetime(2023, 11, 15, 14, 30, tzinfo=timezone.utc).timestamp())
    b2 = {'t': np.array([last - 86400, last]), 'o': np.ones(2), 'h': np.ones(2), 'l': np.ones(2), 'c': np.ones(2), 'v': np.ones(2), 'name': 'x'}
    assert len(data.drop_open_bar(b2, datetime(2023, 11, 15, 17, 0, tzinfo=timezone.utc))['c']) == 1      # 12:00 en Nueva York: la sesión sigue abierta
    assert len(data.drop_open_bar(b2, datetime(2023, 11, 15, 22, 0, tzinfo=timezone.utc))['c']) == 2      # 17:00: cerrada
    print('ok · datos: se interpreta la respuesta de Yahoo y se descarta la barra de la sesión abierta')


def test_lab_json(tmp='/tmp/vp_lab_test'):
    b = walk(1200, seed=4)
    res = lab.analyze('RW', b, name='Paseo')
    json.dumps(res)
    n = res['n']
    assert n == 340 and len(res['bars']['t']) == n
    for p in res['patterns']:
        assert 0 <= p['i'] < n and p['key'] in CATALOG or p['key'].startswith('forming_')
        for pt in p['pts']:
            assert 0 <= pt[0] < n + 60
    row = lab.index_row(res)
    assert row['sym'] == 'RW' and len(row['spark']) == lab.SPARK
    os.makedirs(tmp, exist_ok=True)
    lab.write_catalog(tmp)
    cat = json.load(open(os.path.join(tmp, 'catalog.json')))
    assert set(cat['patterns']) == set(CATALOG)
    print(f'ok · laboratorio: JSON coherente ({len(res["patterns"])} patrones en el paseo aleatorio)')


# ── 5. estudio ────────────────────────────────────────────────────────────
def test_study_mechanics():
    # t con clústeres de día: coincide con la fórmula a mano
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
        idx = np.arange(100, 1100, 9)
        w['idx'] = {'hammer': idx}
        r = w['ret'][10].copy()
        r[idx] += 3.0
        w['ret'][10] = r
        work2[f'S{j}'] = w
    P2 = study.Panel(work2, ks=(8,))
    st = P2.stat(P2.ev['hammer'], 8, 'long', 10)
    assert st['n'] > 500 and st['ex'] > 2.5 and st['t'] > 10, st
    print(f'ok · estudio: t por clústeres exacta y ventaja plantada de +3 % detectada (t = {st["t"]:.0f})')


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


ALL = [test_causality, test_candles_planted, test_breakouts_planted, test_figures_planted, test_catalog, test_plan, test_min_stop_and_forming, test_shapes_geometry,
       test_data_parse, test_lab_json, test_study_mechanics, test_study_no_edge_on_random_walk]

if __name__ == '__main__':
    for fn in ALL:
        fn()
    print('TODO OK')
