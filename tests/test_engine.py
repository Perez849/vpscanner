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


def synth(n=2500, seed=0, mu=0.0, sig=0.015, ar=0.0):
    r = np.random.default_rng(seed)
    e = r.normal(mu, sig, n)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = e[i] + ar * ret[i - 1]
    c = 100 * np.exp(np.cumsum(ret))
    o = np.concatenate([[100], c[:-1]]) * np.exp(r.normal(0, sig * 0.4, n))
    h = np.maximum(o, c) * np.exp(np.abs(r.normal(0, sig * 0.5, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(r.normal(0, sig * 0.5, n)))
    v = r.lognormal(15, 0.4, n)
    t = (np.arange(n) * 86400 + 1.5e9).astype(float)
    return {'t': t, 'o': o, 'h': h, 'l': l, 'c': c, 'v': v}


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


if __name__ == '__main__':
    test_causality()
    test_vectorized_equals_scalar()
    test_no_edge_on_random_walk()
    print('TODO OK')
