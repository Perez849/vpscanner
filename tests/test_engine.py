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


if __name__ == '__main__':
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
