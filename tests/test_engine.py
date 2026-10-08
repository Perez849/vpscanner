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


if __name__ == '__main__':
    test_trailing_stop_and_close_entry()
    test_exit_trigger_formula()
    test_causality()
    test_vectorized_equals_scalar()
    test_no_edge_on_random_walk()
    print('TODO OK')
