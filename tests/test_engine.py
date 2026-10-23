"""
Pruebas del motor (ejecutar: python tests/test_engine.py). Sin dependencias de red.

1. El simulador vectorizado coincide con el seguimiento escalar barra a barra (track.resolve).
2. En un paseo aleatorio sin deriva NO aparece ventaja: ningún setup debe superar a la entrada aleatoria.
3. Causalidad: los indicadores en la barra t no cambian si se recortan los datos posteriores.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scanner'))
import feats, setups, simulate, track


import synth as _synth


def synth(n=2500, seed=0, mu=0.0, sig=0.015, ar=0.0):
    return _synth.series(n, seed, mu, sig, ar)


def test_vectorized_equals_scalar():
    bad = tot = 0
    for seed in range(6):
        F = feats.build(synth(seed=seed, ar=-0.1))
        for sgn in (1, -1):
            m = np.isfinite(F['rsi2']) & ((F['rsi2'] < 25) if sgn > 0 else (F['rsi2'] > 75)) & np.isfinite(F['atr'])
            idx = np.flatnonzero(m)[:300]
            vs = simulate.default_variants()
            res = simulate.simulate(F, idx, sgn, vs, 0.1)
            for v in vs:
                ex = {'kind': v.kind, 'T': v.T, 'S': v.S, 'H': v.H}
                for q, i in enumerate(idx):
                    r = track.resolve(F, int(i), sgn, ex, 0.1)
                    p = res[v.name].pnl[q]
                    if r['status'] != 'cerrada':
                        assert np.isnan(p) or r.get('pendingExit') is None
                        continue
                    if not np.isfinite(p):
                        continue
                    tot += 1
                    bad += abs(r['pnl'] - p) > 1e-9
    assert tot > 10000 and bad == 0, (tot, bad)
    print(f'ok · simulador == seguimiento en {tot} operaciones')


def test_no_edge_on_random_walk():
    v = [simulate.Variant('atr', 1.0, 4.0, 10)]
    ids = ['L_rsi2_5', 'L_streak3', 'L_low10', 'B_up']
    acc = {k: [] for k in ids}
    for s in range(250):
        F = feats.build(synth(seed=500 + s, mu=0.0, sig=0.016))
        ev = setups.detect(F, 'etf', [setups.BY_ID[k] for k in ids])
        for k in ids:
            r = simulate.simulate(F, ev[k], +1, v, 0.10)
            acc[k].append(r[v[0].name].pnl)
    base = np.concatenate(acc['B_up']); base = base[np.isfinite(base)]
    for k in ids[:-1]:
        x = np.concatenate(acc[k]); x = x[np.isfinite(x)]
        se = np.sqrt(x.var() / len(x) + base.var() / len(base))
        z = (x.mean() - base.mean()) / se
        assert abs(z) < 3.5, (k, z)
        print(f'ok · {k}: ventaja sobre aleatorio en RW z={z:+.2f} (n={len(x)})')


def test_causality():
    b = synth(seed=3, ar=-0.1)
    F_full = feats.build(b)
    cut = 1800
    F_cut = feats.build({k: v[:cut] for k, v in b.items()})
    for k in ('sma200', 'rsi2', 'rsi14', 'atr', 'ibs', 'bbz', 'dn_streak', 'vol_ratio', 'dd20', 'pos52', 'slope200'):
        a, c = F_full[k][:cut], F_cut[k]
        assert np.allclose(a, c, equal_nan=True, atol=1e-9), k
    # perfil de volumen móvil: idéntico con y sin futuro
    idx = np.array([1500, 1700, 1799])
    p1 = feats.vp_levels(F_full, idx)
    p2 = feats.vp_levels(F_cut, idx)
    assert all(np.allclose(x, y, equal_nan=True) for x, y in zip(p1, p2))
    # y las señales en la barra t no dependen de lo que venga después
    ev_full = setups.detect(F_full, 'etf'); ev_cut = setups.detect(F_cut, 'etf')
    for sid, idx_c in ev_cut.items():
        got = ev_full[sid][ev_full[sid] < cut - 10]
        exp = idx_c[idx_c < cut - 10]
        assert np.array_equal(got, exp), sid
    print('ok · indicadores, perfil de volumen y señales son causales')


def test_exit_trigger_formula():
    """El cierre que da RSI(2)=70 según la fórmula coincide con recalcular el RSI con ese cierre."""
    b = synth(seed=11, ar=-0.1)
    F = feats.build(b)
    n = 0
    for i in range(300, 2400, 37):
        ru, rd = F['rsi2_ru'][i], F['rsi2_rd'][i]
        d = max(0.0, (7.0 / 3.0) * rd - ru)
        if d <= 0 or not np.isfinite(d):
            continue
        for delta, expect in ((d * 1.001, True), (d * 0.999, False)):
            bb = {k: v[:i + 2].copy() for k, v in b.items()}
            bb['c'][i + 1] = b['c'][i] + delta
            bb['h'][i + 1] = max(bb['h'][i + 1], bb['c'][i + 1]); bb['l'][i + 1] = min(bb['l'][i + 1], bb['c'][i + 1])
            r = feats.build(bb)['rsi2'][i + 1]
            assert (r > 70) == expect, (i, delta, r)
        n += 1
    assert n > 10
    print(f'ok · fórmula del precio de activación de la salida RSI(2)>70 ({n} casos)')


def test_trailing_stop_and_close_entry():
    """El trailing stop vectorizado y la entrada al cierre coinciden con una implementación escalar barra a barra."""
    bad = tot = 0
    for seed in range(5):
        F = feats.build(synth(seed=seed + 30, ar=0.05, sig=0.02))
        n = len(F['c'])
        idx = np.flatnonzero(np.isfinite(F['atr']))[200:260 + 300:3][:250]
        idx = idx[idx + 70 < n]
        for sgn in (1, -1):
            for (K, S0, H) in ((2.5, 2.5, 20), (3.5, 3.5, 40), (5.0, 5.0, 60)):
                v = simulate.Variant('trl', K, S0, H)
                res = simulate.simulate(F, idx, sgn, [v], 0.2)[v.name]
                for q, i in enumerate(idx):
                    E = F['o'][i + 1]; atr = F['atr'][i]
                    hh = E if sgn > 0 else E       # extremo favorable alcanzado hasta la barra anterior
                    stop = E - sgn * S0 * atr
                    ex = None
                    for j in range(1, H + 1):
                        b = i + j
                        o, h, l, c = F['o'][b], F['h'][b], F['l'][b], F['c'][b]
                        if sgn > 0:
                            stop = max(E - S0 * atr, hh - K * atr)
                            if l <= stop:
                                ex = min(o, stop); break
                            hh = max(hh, h)
                        else:
                            stop = min(E + S0 * atr, hh + K * atr)
                            if h >= stop:
                                ex = max(o, stop); break
                            hh = min(hh, l)
                    if ex is None:
                        ex = F['c'][i + H]
                    ref = sgn * (ex - E) / E * 100 - 0.2
                    tot += 1
                    bad += abs(ref - res.pnl[q]) > 1e-9
    assert tot > 3000 and bad == 0, (tot, bad)
    # entrada al cierre: el precio de entrada es el cierre de la barra de señal
    F = feats.build(synth(seed=3))
    idx = np.array([400, 500, 600])
    for sgn in (1, -1):
        r = simulate.simulate(F, idx, sgn, [simulate.Variant('rsi', 0, 4, 10)], 0.0, entry_mode='close')
        assert np.allclose(r['rsi_S4_H10'].entry, F['c'][idx])
    print(f'ok · trailing stop == referencia escalar ({tot} operaciones) y entrada al cierre')


def test_resolve_trailing_equals_simulate():
    bad = tot = 0
    for seed in range(3):
        F = feats.build(synth(seed=seed + 60, ar=0.05, sig=0.02))
        idx = np.arange(300, 1500, 9)
        for sgn in (1, -1):
            v = simulate.Variant('trl', 5.0, 5.0, 60)
            res = simulate.simulate(F, idx, sgn, [v], 0.2)[v.name]
            for q, i in enumerate(idx):
                r = track.resolve(F, int(i), sgn, {'kind': 'trl', 'T': 5.0, 'S': 5.0, 'H': 60}, 0.2)
                tot += 1
                bad += (r['status'] != 'cerrada') or abs(r['pnl'] - res.pnl[q]) > 1e-9
    assert tot > 500 and bad == 0, (tot, bad)
    print(f'ok · seguimiento (trailing) == simulador en {tot} operaciones')


def test_portfolio_sim():
    """Cartera con capital limitado: aritmética exacta con operaciones a mano."""
    import research
    cal = np.arange(100, 130)
    # 3 señales: A (sesión 2, 3 sesiones, +10 %), B (sesión 2, puesto 2, +5 %: no cabe con M=1), C (sesión 10, 1 sesión, −4 %)
    ti = np.array([2, 2, 10]); bars = np.array([3, 3, 1]); res = np.array([10.0, 5.0, -4.0]); rank = np.array([1, 2, 1])
    r = research.portfolio_sim(cal, ti, bars, res, rank, 1, 0.5)
    eq = (1 + 0.5 * 0.10) * (1 - 0.5 * 0.04)
    assert abs(r['curve'][-1] - eq) < 1e-12, (r['curve'][-1], eq)
    assert abs(r['tradesYear'] * (len(r['curve']) / 252.0) - 2) < 1e-9          # B se queda fuera por falta de hueco
    r2 = research.portfolio_sim(cal, ti, bars, res, rank, 2, 0.25)               # con 2 huecos entran A y B
    eq2 = 1 + 0.25 * 0.10 + 0.25 * 0.05
    eq2 *= (1 - 0.25 * 0.04)
    assert abs(r2['curve'][-1] - eq2) < 1e-12, (r2['curve'][-1], eq2)
    print('ok · simulación de cartera con capital limitado (aritmética exacta)')


def test_midterm_signals():
    """Tendencia y rotación de medio plazo: lógica determinista con series construidas a mano."""
    import midsig
    n = 400
    t0 = int(np.datetime64('2025-01-02').astype('datetime64[s]').astype(np.int64))
    t = (t0 + np.arange(n) * 86400).astype(float)
    mk = lambda c: {'t': t, 'o': c, 'h': c, 'l': c, 'c': c, 'v': np.full(n, 1e6)}
    up = np.linspace(100, 200, n); down = np.linspace(200, 100, n); flat = np.full(n, 100.0)
    bars = {'SPY': mk(up), 'QQQ': mk(down), 'IWM': mk(flat), 'EFA': mk(up * 1.0), 'XLK': mk(up * 2), 'XLE': mk(down)}
    last_day = int(t[-1] // 86400)
    r = midsig.rotation_at(bars, last_day, pool=['SPY', 'QQQ', 'IWM', 'EFA', 'XLK', 'XLE'])
    held = {h['sym'] for h in r['holdings']}
    assert held <= {'SPY', 'EFA', 'XLK'} and len(held) == 3 and abs(r['cashW']) < 1e-9, held       # solo los de momentum positivo
    r2 = midsig.rotation_at(bars, last_day, pool=['QQQ', 'IWM', 'XLE'])
    assert r2['holdings'] == [] and abs(r2['cashW'] - 1.0) < 1e-9                                     # sin momentum positivo → todo a liquidez
    assert midsig.trend_status(mk(up))['above'] is True and midsig.trend_status(mk(down))['above'] is False
    # fin de mes: 2026-10-30 (viernes) cierra octubre; 2026-10-07 queda dentro de octubre → el corte es el 30-sep
    d = lambda s_: int(np.datetime64(s_).astype('datetime64[D]').astype(np.int64))
    assert midsig.month_end_before(d('2026-10-30')) == d('2026-10-30') and midsig.month_end_before(d('2026-10-07')) == d('2026-09-30')
    print('ok · señales de medio plazo (tendencia, rotación, fin de mes)')


def test_ema_exit_kinds():
    """Salidas «cierre bajo la EMA 34/89» (solo estudios): el simulador vectorizado coincide con una referencia escalar (largos y cortos)."""
    import emastudy
    bad = tot = 0
    for seed in range(3):
        F = feats.build(synth(seed=seed + 80, ar=0.05, sig=0.018))
        F['ema34'], F['ema89'] = emastudy.ema(F['c'], 34), emastudy.ema(F['c'], 89)
        idx = np.arange(150, 1500, 11)
        for sgn in (1, -1):
            for kind, H, S in (('e34', 30, 3.0), ('e89', 40, 4.0)):
                v = simulate.Variant(kind, 0.0, S, H)
                res = simulate.simulate(F, idx, sgn, [v], 0.2)[v.name]
                for q, i in enumerate(idx):
                    n = len(F['c']); atr = F['atr'][i]
                    if not np.isfinite(atr) or i + H >= n:
                        continue
                    E = F['o'][i + 1]; stop = E - sgn * S * atr
                    em = F['ema34' if kind == 'e34' else 'ema89']
                    exit_px, bars = None, 0
                    for j in range(1, H + 1):
                        b = i + j
                        o, h, l, c = F['o'][b], F['h'][b], F['l'][b], F['c'][b]
                        hit_gap = (o <= stop) if sgn > 0 else (o >= stop)
                        hit = (l <= stop) if sgn > 0 else (h >= stop)
                        if hit_gap:
                            exit_px, bars = o, j; break
                        if hit:
                            exit_px, bars = stop, j; break
                        cond = np.isfinite(em[b]) and ((c < em[b]) if sgn > 0 else (c > em[b]))
                        if cond:
                            if j < H:
                                exit_px, bars = F['o'][b + 1], j + 1
                            else:
                                exit_px, bars = c, j
                            break
                        if j == H:
                            exit_px, bars = c, j
                    ref = sgn * (exit_px - E) / E * 100.0 - 0.2
                    tot += 1
                    bad += (abs(ref - res.pnl[q]) > 1e-9) or (bars != res.bars[q])
    assert tot > 800 and bad == 0, (tot, bad)
    print(f'ok · salidas por EMA == referencia escalar ({tot} operaciones)')


def test_stockmid():
    """Búsqueda a medio plazo con acciones: causalidad de las señales, aritmética exacta de la cartera y potencia del contraste del «mejor de miles»."""
    import io, re, contextlib
    import pandas as pd
    import stockmid
    rng = np.random.default_rng(5)
    T, n = 1700, 70
    days, d = [], pd.Timestamp('2014-01-02')
    while len(days) < T:
        if d.weekday() < 5:
            days.append(int(d.timestamp()))
        d += pd.Timedelta(days=1)
    t = np.array(days)
    mkt = rng.normal(0.0004, 0.01, T)

    def make(skill):
        mu = rng.normal(0, skill, n)
        data, uni = {}, {}
        for i in range(n):
            c = 50 * np.cumprod(1 + mkt * 0.8 + mu[i] + rng.normal(0, 0.015, T))
            data[f'S{i}'] = {'t': t, 'o': c, 'h': c, 'l': c, 'c': c, 'v': np.full(T, 1e6)}
            uni[f'S{i}'] = {'group': 'us_large'}
        c = 100 * np.cumprod(1 + mkt)
        data['SPY'] = {'t': t, 'o': c, 'h': c, 'l': c, 'c': c, 'v': np.full(T, 1e7)}
        return uni, data

    # 1) causalidad: alterar los precios posteriores a la fecha de decisión no cambia ninguna señal en esa fecha ni en las anteriores
    uni, data = make(0.0)
    cal = (t // 86400).astype(np.int64)
    R = np.array([i for i in stockmid.month_ends(cal) if 320 <= i < T - 40])
    P = np.column_stack([data[f'S{i}']['c'] for i in range(n)])
    F1 = stockmid.build_factors(P, R)
    k = len(R) // 2
    P2 = P.copy()
    P2[R[k] + 1:] *= rng.uniform(0.5, 1.5, size=P2[R[k] + 1:].shape)
    F2 = stockmid.build_factors(P2, R)
    for name in F1:
        assert np.allclose(F1[name][:k + 1], F2[name][:k + 1], equal_nan=True), name
    # 2) aritmética: h=1 con costes de rotación, y h=3 con tres selecciones solapadas (la que falta antes del mes 0 vale liquidez)
    C = stockmid.COST_RT
    RET1 = np.array([[0.10, 0.00, -0.10, 0.20], [0.00, 0.10, 0.00, 0.00], [0.05, 0.05, 0.05, 0.05], [0.01, 0.02, 0.06, 0.04], [0.1, 0.1, 0.1, 0.1]])
    cash = np.full(5, 0.002)
    sel = [np.array(x) for x in ([0, 3], [1, 2], [1, 2], [0, 3], [0, 3])]
    r1 = stockmid._variant_returns(sel, RET1, cash, 1)
    assert abs(r1[0] - 0.15) < 1e-12
    assert abs(r1[1] - (0.05 - C)) < 1e-12                                      # {1,2} no solapa con {0,3}: rotación completa
    assert abs(r1[2] - 0.05) < 1e-12                                            # misma selección: sin coste
    assert abs(r1[3] - (0.025 - C)) < 1e-12
    r3 = stockmid._variant_returns(sel, RET1, cash, 3)
    assert abs(r3[1] - (0.05 + 0.0 + 0.002) / 3) < 1e-12                        # tramos de los meses 1, 0 y (inexistente) -1
    assert abs(r3[3] - (0.025 + 0.04 + 0.04) / 3) < 1e-12                      # tramos 3, 2, 1; el mes 3 reemplaza al 0 con la misma selección: sin coste
    assert abs(r3[4] - (0.1 - C / 3)) < 1e-12                                   # el mes 4 reemplaza al 1 ({1,2}→{0,3}): rotación completa de un tercio
    # 3) potencia: sin habilidad el «mejor de miles» no es significativo; con habilidad sí
    stockmid.KS, stockmid.HS = (10, 30), (1, 3)

    def pval(skill):
        uni_, data_ = make(skill)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stockmid.core(uni_, data_, [f'S{i}' for i in range(n)], False, 'test')
        return float(re.search(r'p-valor ≈ ([0-9.]+)', buf.getvalue()).group(1))
    p0, p1 = pval(0.0), pval(0.0008)
    assert p0 > 0.05 and p1 < 0.05, (p0, p1)
    print(f'ok · medio plazo con acciones: señales causales, aritmética exacta, contraste sin habilidad p={p0:.2f} / con habilidad p={p1:.3f}')


def test_pit_membership():
    """Pertenencia histórica a un índice: lectura de la tabla de cambios de Wikipedia, intervalos y máscara de eventos."""
    import pit
    import pandas as pd
    cons = ''.join(f'<tr><td>M{i}</td><td>Sec</td></tr>' for i in range(6))
    html = f"""<html><body>
    <table><tr><th>Symbol</th><th>Security</th></tr>{cons}<tr><td>NEW</td><td>Newco</td></tr><tr><td>BRK.B</td><td>Berkshire</td></tr></table>
    <table><tr><th rowspan=2>Effective Date</th><th colspan=2>Added</th><th colspan=2>Removed</th><th rowspan=2>Reason</th></tr>
    <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
    <tr><td>March 23, 2020</td><td>NEW</td><td>Newco</td><td>OLD</td><td>Oldco</td><td>x</td></tr>
    <tr><td>June 1, 2018</td><td>OLD</td><td>Oldco</td><td></td><td></td><td>y</td></tr>
    <tr><td>January 5, 2022</td><td></td><td></td><td>GONE</td><td>Goneco</td><td>z</td></tr>
    </table></body></html>"""
    cur, chg = pit.parse_page(html, min_rows=5)
    assert cur == {f'M{i}' for i in range(6)} | {'NEW', 'BRK-B'}, cur
    assert len(chg) == 3 and any(a == 'NEW' and r == 'OLD' for _, a, r in chg)
    iv = pit.intervals(cur, chg)
    d = lambda s_: int(pd.Timestamp(s_).value // 86400 // 10 ** 9)
    assert iv['M0'] == [(pit.NEG, pit.POS)] and iv['BRK-B'] == [(pit.NEG, pit.POS)]
    assert iv['NEW'] == [(d('2020-03-23'), pit.POS)]                              # alta en 2020: antes NO era miembro
    assert iv['OLD'] == [(d('2018-06-01'), d('2020-03-23'))]                      # alta 2018, baja 2020
    assert iv['GONE'] == [(pit.NEG, d('2022-01-05'))]                             # primera noticia = baja: ya era miembro antes
    names = ['M0', 'NEW', 'OLD', 'GONE', 'ETF1']
    sym = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4]); day = np.array([d('2019-01-01'), d('2019-01-01'), d('2021-01-01'), d('2017-01-01'), d('2019-01-01'), d('2021-01-01'), d('2021-12-31'), d('2022-01-05'), d('2019-01-01')])
    ok = pit.event_mask(names, sym, day, iv)
    assert ok.tolist() == [True, False, True, False, True, False, True, False, True], ok.tolist()
    ok2 = pit.event_mask(names, sym, day, iv, require_info=True)
    assert not ok2[-1] and ok2[0]
    Mx = pit.member_matrix(names, np.array([d('2019-01-01'), d('2021-01-01')]), iv, True)
    assert Mx.tolist() == [[True, False, True, True, True], [True, True, False, True, True]]
    print('ok · pertenencia histórica a índices (lectura de tabla, intervalos, máscaras)')


def test_longidx_clean():
    """Índices largos: se descarta el tramo sin máximos/mínimos reales y lo anterior a un salto de más de 15 días."""
    import longidx
    uni, data = _synth.universe(2, 0.0, n_bars=5000)
    b = {k: v.copy() for k, v in data['SPY'].items()}
    n = len(b['c'])
    b2 = {k: v.copy() for k, v in b.items()}
    b2['h'][:1500] = b2['c'][:1500]; b2['l'][:1500] = b2['c'][:1500]
    seg = longidx.clean_series(b2)
    assert seg is not None and abs(len(seg['c']) - (n - 1500)) <= 30, len(seg['c'])
    b3 = {k: v.copy() for k, v in b.items()}
    b3['t'][3000:] += 40 * 86400
    seg3 = longidx.clean_series(b3)
    assert seg3 is not None and len(seg3['c']) == n - 3000
    print('ok · índices largos: limpieza de tramos (OHLC real, saltos)')


def test_stop_variants():
    """Stop con tope en % (cap) y stop «al cierre» (cs): el simulador vectorizado coincide con una referencia escalar barra a barra."""
    bad = tot = 0
    for seed in range(4):
        F = feats.build(synth(seed=seed + 40, ar=-0.1, sig=0.02))
        n = len(F['c'])
        idx = np.flatnonzero(np.isfinite(F['rsi2']) & (F['rsi2'] < 20) & np.isfinite(F['atr']))[:400]
        vs = [simulate.Variant('rsi', 0.0, S, 10, cap, cs) for S in (4.0, 99.0) for cap in (0.0, 3.0, 5.0, 8.0) for cs in (False, True)]
        res = simulate.simulate(F, idx, 1, vs, 0.1)
        for v in vs:
            for q, i in enumerate(idx):
                if i + v.H >= n:
                    continue
                E = F['o'][i + 1]; atr = F['atr'][i]
                dist = v.S * atr if v.cap <= 0 else min(v.S * atr, v.cap / 100.0 * E)
                stop = E - dist
                exit_px = None; bars = None
                for j in range(1, v.H + 1):
                    b = i + j
                    if v.cs:
                        if F['c'][b] <= stop:
                            exit_px, bars = (F['o'][b + 1], j + 1) if j < v.H else (F['c'][b], j); break
                    else:
                        if F['o'][b] <= stop:
                            exit_px, bars = F['o'][b], j; break
                        if F['l'][b] <= stop:
                            exit_px, bars = stop, j; break
                    if F['rsi2'][b] > 70:
                        exit_px, bars = (F['o'][b + 1], j + 1) if j < v.H else (F['c'][b], j); break
                    if j == v.H:
                        exit_px, bars = F['c'][b], j
                ref = (exit_px - E) / E * 100.0 - 0.1
                p = res[v.name].pnl[q]
                tot += 1
                bad += (abs(ref - p) > 1e-9) or (int(res[v.name].bars[q]) != bars)
    assert tot > 5000 and bad == 0, (tot, bad)
    print(f'ok · stop con tope en % y stop al cierre == referencia escalar ({tot} operaciones)')


def test_clock_sectors():
    """Reloj de inversión: máscaras de sector, calendario mensual (los meses sin dato heredan el último) y contraste de permutaciones."""
    import clock
    gics = {'XOM': {'sector': 'Energy', 'sub': 'Oil & Gas'}, 'NVDA': {'sector': 'Information Technology', 'sub': 'Semiconductors'},
            'MSFT': {'sector': 'Information Technology', 'sub': 'Systems Software'}, 'PLD': {'sector': 'Real Estate', 'sub': 'REITs'}}
    assert clock.tags_of('XOM', {}, gics) == clock.BIT['Energía']
    assert clock.tags_of('NVDA', {}, gics) == clock.BIT['Tecnología'] | clock.BIT['Semiconductores']       # un valor de chips cuenta para tecnología y para semiconductores
    assert clock.tags_of('MSFT', {}, gics) == clock.BIT['Tecnología']
    assert clock.tags_of('PLD', {}, gics) == clock.BIT['Inmobiliario'] and not (clock.BIT['Inmobiliario'] & clock.EQ_MASK)
    assert clock.tags_of('XLE', {}, {}) == clock.BIT['Energía'] and clock.tags_of('GLD', {}, {}) == clock.BIT['Oro (lingote)']
    assert clock.tags_of('ZZZZ', {}, gics) == 0 and clock.tags_of('ZZZZ', {'group': 'crypto'}, gics) == 0
    picks = {'months': {'2020-01': {'equity': {'Energía': 0.5, 'Salud': 0.5}, 'gold': {}}, '2020-03': {'equity': {'Tecnología': 1.0}, 'gold': {'Oro (lingote)': 0.0}}}}
    pm = clock.pick_masks(picks)
    day = lambda s: np.array([(np.datetime64(s) - np.datetime64('1970-01-01')).astype(int)])
    em = np.array([clock.BIT['Energía']], dtype=np.int64)
    for s, want in (('2020-01-15', True), ('2020-02-10', True), ('2020-03-02', False), ('2019-12-31', False)):
        got, have = clock.event_picked(clock.month_index(day(s)), em, pm)
        assert bool(got[0]) == want and bool(have[0]) == (s >= '2020-01'), (s, got, have)
    assert clock.month_index(day('2020-03-31'))[0] == 2020 * 12 + 2
    # contraste de permutaciones: sin ventaja del sector, la media real no se distingue del azar; con ventaja, la supera
    rng = np.random.default_rng(3)
    n = 6000
    mi = rng.integers(2020 * 12, 2020 * 12 + 36, n)
    bits = np.array([clock.BIT[x] for x in clock.EQ], dtype=np.int64)
    em = bits[rng.integers(0, len(bits), n)]
    pm2 = {m: int(bits[(m * 7) % len(bits)] | bits[(m * 3 + 1) % len(bits)]) for m in range(2020 * 12, 2020 * 12 + 36)}
    for edge in (0.0, 1.0):
        picked, _ = clock.event_picked(mi, em, pm2)
        x = rng.normal(0, 3, n) + edge * picked
        null = clock.permutation_null(mi, em, x, pm2, 60, np.random.default_rng(5))
        act = x[picked].mean()
        pct = (null < act).mean()
        assert (pct < 0.98) if edge == 0 else (pct == 1.0), (edge, act, null.mean(), pct)
    print('ok · reloj de inversión: sectores, calendario y contraste de permutaciones')


def test_osc_causal():
    """Los ~55 osciladores complejos son causales: recortar el futuro no cambia ni un valor del pasado, y no hay infinitos."""
    import osc
    b = synth(seed=77, ar=-0.1, sig=0.02)
    n0 = len(b['c'])
    F = feats.build(b)
    B = osc.battery(F)
    assert len(B) >= 50 and all(len(v) == n0 for v in B.values())
    assert not any(np.isinf(v).any() for v in B.values())
    for cut in (n0 - 400, n0 - 37):
        F2 = feats.build({k: (v[:cut] if isinstance(v, np.ndarray) and len(v) == n0 else v) for k, v in b.items()})
        B2 = osc.battery(F2)
        for k in B:
            a, c2 = B[k][:cut - 3], B2[k][:cut - 3]
            m = np.isfinite(a) & np.isfinite(c2)
            assert np.array_equal(np.isfinite(a), np.isfinite(c2)) and np.allclose(a[m], c2[m], atol=1e-9, rtol=1e-9), (k, cut)
    print(f'ok · {len(B)} osciladores complejos son causales')


def _fake_alt_sources(days_all, syms):
    """Fuentes alternativas sintéticas con la misma forma que las reales."""
    import datetime as dt
    import alt
    rng = np.random.default_rng(5)
    d = np.arange(days_all[0] - 400, days_all[-1] + 10, dtype=np.int64)
    d = d[(d + 4) % 7 < 5]
    mk = lambda lo, hi: (d, rng.uniform(lo, hi, len(d)).cumsum() / 50 + 10)
    src = {}
    for k in ('BAA10Y', 'AAA10Y', 'T10Y2Y', 'T10Y3M', 'DFII10', 'T5YIE', 'DTWEXBGS', 'DCOILWTICO', 'DGS10', 'USEPUINDXD', 'NFCI', 'STLFSI4', 'ICSA', 'WALCL', 'RRPONTSYD'):
        src[('fred', k)] = mk(0.5, 1.5)
    for k in ('VIX', 'VIX9D', 'VIX3M', 'VIX6M', 'VVIX', 'SKEW', 'OVX', 'GVZ', 'VXN', 'RVX'):
        src[('cboe', k)] = mk(0.5, 1.5)
    src[('aaii', 'all')] = (d[::5], rng.uniform(0.2, 0.5, len(d[::5])), rng.uniform(0.2, 0.5, len(d[::5])))
    cal = {}
    for q in range(len(d) // 3):
        day = dt.date(1970, 1, 1) + dt.timedelta(days=int(d[q * 3]))
        cal[day] = [(syms[int(rng.integers(0, len(syms)))], ['time-pre-market', 'time-after-hours', 'time-not-supplied'][q % 3], 0.5, 0.4, float(rng.normal(0, 10)), 1e9, 8.0) for _ in range(4)]
    src['nasdaq'] = cal
    ins = {s_: np.array([[d[i], 1e5 * (i % 3 == 0), 5e4, 1e5 * (i % 6 == 0), 0, 1.0 * (i % 3 == 0)] for i in range(0, len(d), 40)], dtype=float) for s_ in syms[: len(syms) // 2]}
    src['insiders'] = ins
    sd = d[d > d[0] + 300]
    SH = rng.uniform(1e5, 5e5, (len(sd), len(syms))).astype(np.float32)
    src['short_volume'] = (sd, list(syms), SH, SH * 2.0)
    src['short_interest'] = {s_: np.array([[d[i] + 11, 1e6, 1e5, 10.0, 9e5] for i in range(0, len(d), 15)], dtype=float) for s_ in syms}
    src['cik'] = {s_: 1000 + i for i, s_ in enumerate(syms)}
    per = lambda f, step: {1000 + i: np.array([[d[k], f(i, k)] for k in range(0, len(d), step)], dtype=float) for i in range(len(syms))}
    src['fundamentals'] = {'assets': per(lambda i, k: 1e9 + 1e6 * k, 60), 'liab': per(lambda i, k: 6e8, 60), 'equity': per(lambda i, k: 4e8, 60), 'shares': per(lambda i, k: 1e8, 60),
                           'ni': per(lambda i, k: 5e7, 250), 'rev1': per(lambda i, k: 4e8 + 1e6 * k, 250), 'rev2': {}, 'rev3': {}}
    return src


def test_alt_joins():
    """Dimensiones alternativas: retrasos de publicación, ventanas estrictas y construcción completa de las variables con fuentes sintéticas."""
    import alt
    # asof con retraso: el valor del día t solo se ve desde t+lag
    d = np.array([10, 11, 12, 15], dtype=np.int64); v = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([10, 11, 12, 13, 16, 40])
    got = alt.asof_vals(d, v, q, lag=1)
    assert np.isnan(got[0]) and got[1] == 1.0 and got[2] == 2.0 and got[3] == 3.0 and got[4] == 4.0 and got[5] == 4.0
    got = alt.asof_vals(d, v, q, lag=1, max_age=10)
    assert np.isnan(got[5]) and got[4] == 4.0
    # ventana estricta (q-width, q-1]: nada del propio día ni del futuro
    days = np.array([100, 110, 120, 130], dtype=np.int64); vals = np.array([1.0, 10.0, 100.0, 1000.0])
    w = alt.window_sum(days, vals, np.array([100, 101, 121, 130, 131, 250]), 30)
    assert list(w) == [0.0, 1.0, 111.0, 110.0, 1110.0, 0.0], w
    # resultados: antes de la apertura se conocen el mismo día, tras el cierre o sin hora desde el siguiente
    import datetime as dt
    e0 = (dt.date(2024, 5, 1) - dt.date(1970, 1, 1)).days
    cal = {dt.date(2024, 5, 1): [('AAA', 'time-pre-market', 1.0, 0.8, 25.0, 1e9, 9.0), ('BBB', 'time-after-hours', 1.0, 1.1, -9.0, 1e9, 5.0), ('CCC', 'time-not-supplied', 0.5, 0.5, 0.0, 1e9, 3.0)]}
    names = ['AAA', 'BBB', 'CCC']
    sym = np.array([0, 1, 2, 0, 1, 2]); day = np.array([e0, e0, e0, e0 + 1, e0 + 1, e0 + 1], dtype=np.int64)
    M, cn = alt.earn_block(names, sym, day, cal)
    ix = {n: i for i, n in enumerate(cn)}
    assert M[0, ix['earn_since']] == 0 and np.isnan(M[1, ix['earn_since']]) and np.isnan(M[2, ix['earn_since']])          # el mismo día solo AAA (pre-apertura)
    assert M[3, ix['earn_since']] == 1 and M[4, ix['earn_since']] == 1 and M[4, ix['earn_sur_neg']] == 1 and M[3, ix['earn_react3']] == 1
    # directivos: la presentación del propio día no cuenta; después del último trimestre publicado todo es NaN
    ins = {'AAA': np.array([[e0 - 5, 5e5, 1e5, 5e5, 0, 1], [e0, 7e5, 0, 0, 0, 1]], dtype=float)}
    Mi, ni = alt.insider_block(names, np.array([0, 0, 1]), np.array([e0, e0 + 1, e0], dtype=np.int64), ins, last_day=e0 + 10)
    jx = {n: i for i, n in enumerate(ni)}
    assert abs(Mi[0, jx['ins_buy90']] - np.log1p(5e5)) < 1e-4 and abs(Mi[1, jx['ins_buy90']] - np.log1p(1.2e6)) < 1e-4 and Mi[2, jx['ins_buy90']] == 0
    Mi2, _ = alt.insider_block(names, np.array([0]), np.array([e0 + 11], dtype=np.int64), ins, last_day=e0 + 10)
    assert np.isnan(Mi2).all()
    # volumen corto: el archivo del día t no se usa el día t
    sd = np.arange(e0 - 70, e0 + 3, dtype=np.int64)
    SH = np.full((len(sd), 1), 100.0, np.float32); TV = np.full((len(sd), 1), 200.0, np.float32)
    SH[-1, 0] = 190.0                                                   # el último archivo (día e0+2) es distinto
    Ms, ns = alt.short_block(['AAA'], np.array([0, 0]), np.array([e0 + 2, e0 + 3], dtype=np.int64), (sd, ['AAA'], SH, TV), {})
    assert abs(Ms[0, 0] - 0.5) < 1e-6 and abs(Ms[1, 0] - 0.95) < 1e-6, Ms[:, 0]
    # construcción completa con fuentes sintéticas
    n = 600
    rng = np.random.default_rng(1)
    syms = [f'S{i}' for i in range(12)]
    EV = {'sym': rng.integers(0, 12, 3 * n), 'day': np.sort(rng.integers(e0 - 200, e0 + 300, 3 * n)).astype(np.int64), 'sym_names': syms}
    rows = np.arange(0, 3 * n, 3)
    src = _fake_alt_sources(np.arange(e0 - 400, e0 + 400), syms)
    M, cols, fam = alt.build_features(None, EV, rows, src)
    assert M.shape == (n, len(cols)) and len(cols) == len(fam) and len(set(cols)) == len(cols) and len(cols) >= 40, (M.shape, len(cols))
    assert np.isfinite(M).mean() > 0.4 and not np.isinf(M).any()
    # fundamentales: un hecho solo se ve 75 días después del cierre del periodo (balance) y 100 (ejercicio anual)
    fa = {'assets': {1: np.array([[1000.0, 10.0], [1090.0, 20.0]])}, 'liab': {1: np.array([[1000.0, 5.0], [1090.0, 10.0]])}, 'equity': {1: np.array([[1000.0, 5.0], [1090.0, 10.0]])},
          'shares': {1: np.array([[1000.0, 2.0]])}, 'ni': {1: np.array([[1000.0, 1.0], [1365.0, 3.0]])}, 'rev1': {1: np.array([[1000.0, 10.0], [1365.0, 20.0]])}, 'rev2': {}, 'rev3': {}}
    Mf, nf = alt.fund_block(['X'], np.array([0, 0, 0, 0, 0]), np.array([1074, 1075, 1164, 1465, 1466], dtype=np.int64), np.full(5, 5.0), fa, {'X': 1})
    jf = {n: i for i, n in enumerate(nf)}
    assert np.isnan(Mf[0, jf['fd_lev']]) and Mf[1, jf['fd_lev']] == 0.5 and Mf[1, jf['fd_age']] == 75                      # el balance del día 1000 se ve desde el 1075
    assert Mf[2, jf['fd_roa']] == 0.1 and np.isnan(Mf[2, jf['fd_revg']])                                               # el ejercicio anual de 1000 se ve desde el 1100; aún no hay año anterior
    assert Mf[3, jf['fd_lev']] == 0.5 and abs(Mf[3, jf['fd_logmcap']] - np.log10(10.0)) < 1e-6                          # el de 1090 se ve desde el 1165 (aquí aún no)
    assert abs(Mf[4, jf['fd_roa']] - 3.0 / 20.0) < 1e-6 and abs(Mf[4, jf['fd_revg']] - 1.0) < 1e-6                      # el de 1365 se ve desde el 1465: ingresos 10 → 20 = +100 %
    print(f'ok · dimensiones alternativas: retrasos, ventanas estrictas y {len(cols)} variables ({", ".join(sorted(set(fam)))})')


def _plant_bars(oc, wick=0.25):
    o = np.array([x[0] for x in oc], float); c = np.array([x[1] for x in oc], float)
    h = np.maximum(o, c) + wick; l = np.minimum(o, c) - wick
    for k, x in enumerate(oc):
        if len(x) == 4:
            h[k], l[k] = x[2], x[3]
    n = len(o)
    return {'t': 1577836800 + 86400 * np.arange(n), 'o': o, 'h': h, 'l': l, 'c': c, 'v': np.full(n, 1e6)}


def _plant_closes(cl, wick=0.25):
    pc, oc = cl[0], []
    for x in cl:
        oc.append((pc, x)); pc = x
    return _plant_bars(oc, wick)


def test_charts_planted():
    """Chartismo: cada patrón plantado a mano se detecta justo en su barra de ruptura, y las señales son causales (recortar el futuro no cambia el pasado)."""
    import charts
    rng = np.random.default_rng(3)
    quiet = lambda n, p, sd=0.12, seed=5: list(p + np.random.default_rng(seed).normal(0, sd, n))
    # velas: 200 barras laterales, 25 de tendencia (1,2 por barra) y las velas del patrón (o, c, h, l)
    def candle(trend, tail):
        pre, p = [], 100.0
        for k in range(200):
            c = 100.0 + rng.normal(0, 0.3); pre.append((p, c, max(p, c) + 0.25, min(p, c) - 0.25)); p = c
        p = 100.0
        for k in range(25):
            c = p - 1.2 if trend == 'dn' else p + 1.2
            pre.append((p, c, max(p, c) + 0.3, min(p, c) - 0.3)); p = c
        return _plant_bars(pre + tail)
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
        b = candle(tr, tail)
        sg = charts.signals(feats.build(b))[nm]
        assert sg[-1], f'{nm}: no se detecta en la barra final'
    # figuras con precios de cierre unidos por tramos (la ruptura es el último ancla)
    figs = {
        'double_bottom': [(0, 100), (20, 80), (32, 92), (44, 80.5), (52, 93.5)], 'double_top': [(0, 100), (20, 120), (32, 108), (44, 119.5), (52, 106.5)],
        'hs_top': [(0, 80), (15, 100), (24, 92), (36, 108), (48, 92), (58, 100.5), (70, 85)], 'hs_inverse': [(0, 120), (15, 100), (24, 108), (36, 92), (48, 108), (58, 99.5), (70, 115)],
        'asc_triangle': [(0, 90), (10, 100), (18, 93), (26, 100.1), (34, 96), (42, 100), (50, 98.5), (56, 102)],
        'desc_triangle': [(0, 110), (10, 100), (18, 107), (26, 99.9), (34, 104), (42, 100), (50, 101.5), (56, 98)],
        'sym_tri_up': [(0, 90), (10, 110), (20, 92), (30, 106), (40, 96), (48, 102), (54, 98), (60, 104)], 'sym_tri_dn': [(0, 110), (10, 90), (20, 108), (30, 94), (40, 104), (48, 98), (54, 102), (60, 95)],
        'rising_wedge': [(0, 90), (10, 100), (20, 94), (30, 103), (40, 100), (50, 105.5), (58, 103), (64, 99)], 'falling_wedge': [(0, 110), (10, 100), (20, 106), (30, 97), (40, 100), (50, 94.5), (58, 97), (64, 101)],
        'trendline_up': [(0, 100), (10, 110), (18, 100), (30, 104), (40, 98), (48, 108)], 'trendline_dn': [(0, 100), (10, 90), (18, 100), (30, 96), (40, 102), (48, 92)],
        'bull_flag': [(0, 100), (6, 112), (14, 110.5), (16, 114.5)], 'bear_flag': [(0, 100), (6, 88), (14, 89.5), (16, 85.5)],
    }
    for nm, an in figs.items():
        xs, ys = np.array([a[0] for a in an]), np.array([a[1] for a in an])
        cl = np.interp(np.arange(xs[0], xs[-1] + 1), xs, ys)
        pre = quiet(260, an[0][1])
        b = _plant_closes(np.array(pre + list(cl)))
        idx = np.flatnonzero(charts.signals(feats.build(b))[nm])
        n_end = len(pre) + int(xs[-1])
        assert any(n_end - 12 <= i <= n_end for i in idx), (nm, list(idx[-3:]), n_end)
    # tramos con velas explícitas
    def custom(prefix, tail):
        base = _plant_closes(np.array(prefix))
        oc = [(base['o'][k], base['c'][k], base['h'][k], base['l'][k]) for k in range(len(prefix))] + list(tail)
        return _plant_bars(oc)
    chk = lambda b, nm: bool(charts.signals(feats.build(b))[nm][-1])
    zig = lambda lo, hi, k: sum(([*np.linspace(hi, lo, 8), *np.linspace(lo, hi, 8)] for _ in range(k)), [])
    assert chk(custom(quiet(250, 100) + zig(90.2, 100, 3) + list(np.linspace(100, 92, 10)), [(91.0, 91.8, 91.9, 90.3)]), 'support_bounce')
    assert chk(custom(quiet(250, 100) + sum(([*np.linspace(100, 109.8, 8), *np.linspace(109.8, 100, 8)] for _ in range(3)), []) + list(np.linspace(100, 108, 10)), [(109.0, 108.2, 109.7, 108.1)]), 'resistance_reject')
    v = list(np.linspace(40, 100, 300)) + list(np.linspace(100, 88, 8)) + list(np.linspace(88, 100.5, 8)) + list(np.linspace(100.5, 93.5, 8)) + list(np.linspace(93.5, 101, 8)) + list(np.linspace(101, 97.5, 8)) + list(np.linspace(97.5, 101.2, 6)) + [102.6]
    assert charts.signals(feats.build(_plant_closes(np.array(v))))['vcp'][-1]
    x = np.linspace(-1, 1, 72); cup = list(100 - 20 * (1 - x ** 2)); cup[-1] = 99.0
    allc = quiet(250, 80) + list(np.linspace(80, 100, 30)) + cup + list(np.linspace(99, 95.5, 8)) + list(np.linspace(95.5, 98.5, 3)) + [101.0]
    assert charts.signals(feats.build(_plant_closes(np.array(allc))))['cup_handle'][-1]
    widths = [2, 4, 7, 10, 8, 5, 3, 2]
    dia = []
    for w in widths:
        for k in range(5):
            hi, lo = 100 + w / 2, 100 - w / 2
            dia.append((lo + 0.3, hi - 0.3, hi, lo) if k % 2 == 0 else (hi - 0.3, lo + 0.3, hi, lo))
    assert chk(custom(quiet(250, 130) + list(np.linspace(130, 100, 20)), dia + [(100.5, 103.5, 103.7, 100.4)]), 'diamond_bull')
    assert chk(custom(quiet(250, 70) + list(np.linspace(70, 100, 20)), dia + [(99.5, 96.5, 99.6, 96.3)]), 'diamond_bear')
    pen = lambda c0, s: [((c0 + s * (1.8 * (1 - k / 9))) - 0.2 if k % 2 else (c0 - s * (1.8 * (1 - k / 9))) + 0.2, (c0 - s * (1.8 * (1 - k / 9))) + 0.2 if k % 2 else (c0 + s * (1.8 * (1 - k / 9))) - 0.2,
                          c0 + 1.8 * (1 - k / 9), c0 - 1.8 * (1 - k / 9)) for k in range(8)]
    assert chk(custom(quiet(250, 100) + list(np.linspace(100, 112, 7)), pen(111, 1) + [(111.2, 114.5, 114.6, 111.1)]), 'bull_pennant')
    assert chk(custom(quiet(250, 100) + list(np.linspace(100, 88, 7)), pen(89, 1) + [(88.8, 85.5, 88.9, 85.4)]), 'bear_pennant')
    assert chk(custom(quiet(250, 130) + list(np.linspace(130, 100, 20)), [(99.0, 98.6, 99.2, 98.4), (98.5, 98.9, 99.0, 98.3), (101.5, 102.4, 102.6, 101.3)]), 'island_bull')
    assert chk(custom(quiet(250, 70) + list(np.linspace(70, 100, 20)), [(101.0, 101.4, 101.6, 100.8), (101.5, 101.1, 101.7, 101.0), (98.5, 97.6, 98.7, 97.4)]), 'island_bear')
    for nm, tail in (('donch20_up', (100.0, 103.0, 103.1, 99.9)), ('donch20_dn', (100.0, 97.0, 100.1, 96.9)), ('donch55_up', (100.0, 103.0, 103.1, 99.9)), ('donch252_up', (100.0, 103.0, 103.1, 99.9)), ('donch100_dn', (100.0, 97.0, 100.1, 96.9))):
        assert chk(custom(quiet(300, 100, 0.2), [tail]), nm), nm
    assert chk(custom(quiet(250, 100, 0.4), [(100.0, 100.15, 100.25, 99.95), (100.2, 101.0, 101.1, 100.15)]), 'nr7_up')
    assert chk(custom(quiet(250, 100, 0.4), [(100.0, 100.05, 100.15, 99.9), (99.9, 99.0, 99.95, 98.9)]), 'nr7_dn')
    assert chk(custom(quiet(250, 100, 0.4), [(99.0, 101.0, 101.5, 98.5), (100.0, 100.4, 100.8, 99.2), (100.5, 102.0, 102.1, 100.4)]), 'inside_up')
    assert chk(custom(quiet(250, 100, 0.4), [(101.0, 99.0, 101.5, 98.5), (100.0, 99.6, 100.8, 99.2), (99.5, 97.9, 99.6, 97.8)]), 'inside_dn')
    hv = list(100 + np.cumsum(np.random.default_rng(9).normal(0, 1.0, 150)) * 0.3)
    base = _plant_closes(np.array(hv), 0.25)
    oc = [(base['o'][k], base['c'][k], base['h'][k], base['l'][k]) for k in range(len(hv))]
    p = hv[-1]
    for xx in np.random.default_rng(10).normal(0, 0.03, 25):
        c = hv[-1] + xx; oc.append((p, c, max(p, c) + 0.02, min(p, c) - 0.02)); p = c
    for nm, last in (('bb_squeeze_up', (p, p + 2.0, p + 2.1, p - 0.03)), ('atr_squeeze_up', (p, p + 2.0, p + 2.1, p - 0.03)), ('bb_squeeze_dn', (p, p - 2.0, p + 0.03, p - 2.1))):
        assert chk(_plant_bars(oc + [last]), nm), nm
    flat = quiet(250, 100, 0.5) + list(100 + np.random.default_rng(1).normal(0, 0.12, 20))
    assert chk(custom(flat, [(100.0, 102.0, 102.1, 99.95)]), 'box_up') and chk(custom(flat, [(100.0, 98.0, 100.05, 97.9)]), 'box_dn')
    assert chk(custom(list(np.linspace(70, 100, 260)) + list(99 + np.random.default_rng(2).normal(0, 0.5, 25)), [(99.0, 101.5, 101.6, 98.9)]), 'base_near_high')
    # causalidad: recortar el futuro no cambia ninguna señal anterior
    b = synth(seed=61, ar=0.0, sig=0.016)
    n0 = len(b['c'])
    full = charts.signals(feats.build(b))
    for cut in (n0 - 300, n0 - 41):
        cp = charts.signals(feats.build({k: (v[:cut] if isinstance(v, np.ndarray) and len(v) == n0 else v) for k, v in b.items()}))
        for k in full:
            assert np.array_equal(full[k][:cut - 1], cp[k][:cut - 1]), (k, cut)
    print(f'ok · chartismo: {len(cases) + len(figs) + 24} patrones plantados se detectan en su barra y las {len(full)} señales son causales')


if __name__ == '__main__':
    test_charts_planted()
    test_alt_joins()
    test_osc_causal()
    test_clock_sectors()
    test_stop_variants()
    test_longidx_clean()
    test_pit_membership()
    test_stockmid()
    test_ema_exit_kinds()
    test_midterm_signals()
    test_portfolio_sim()
    test_resolve_trailing_equals_simulate()
    test_trailing_stop_and_close_entry()
    test_exit_trigger_formula()
    test_causality()
    test_vectorized_equals_scalar()
    test_no_edge_on_random_walk()
    print('TODO OK')
