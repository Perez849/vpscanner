"""
Contexto y confluencia: ¿funcionan los patrones cuando se operan como los discrecionales?

El estudio de evidencia opera cada patrón «a ciegas». Quien vive de dibujar y de leer velas dice que el patrón solo vale en su sitio: en un soporte o una resistencia, a favor de la
tendencia o del mercado, con volumen, en un valor líder, con varias señales a la vez. Aquí se prueba, con las mismas reglas de rigor:

  · Señal = cada evento de patrón (velas, figuras, rupturas, directrices, rebotes) de las 300 acciones más negociadas, con su sentido. Resultado = rendimiento a 10 sesiones del valor
    menos el del conjunto ese día, con el signo del sentido (en corto cuenta al revés): es «exceso», no depende de si el mercado sube.
  · Doce filtros definidos de antemano (abajo). Para cada grupo (familia y sentido) y cada filtro: exceso medio de las señales que lo cumplen y t con errores estándar agrupados por día.
  · Umbral del mejor de N: las señales de cada valor se desplazan en el tiempo al azar (conservando su agrupación) y se anota el mayor |t| entre TODAS las combinaciones grupo × filtro.
    Solo cuenta lo que lo supera.
  · Desarrollo (≤ 2022) y final (≥ 2023) por separado: un hallazgo del desarrollo que no se repite en el final no vale.
  · Meta-etiquetado: un modelo (gradient boosting) aprende, solo con el pasado, cuándo fiarse de cada señal; se evalúa año a año fuera de muestra con el mismo reparto desarrollo / final.
"""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ml
from .catalog import CATALOG
from .ml import CTX, COST

ALL = sorted(CATALOG)
FAMS = ('velas', 'figuras', 'rupturas', 'directrices', 'rebotes')
DEV_END = 2022
FILTERS = {
    'tendencia': 'a favor de la tendencia: precio sobre su media de 200 sesiones si es alcista (bajo si es bajista)',
    'contra': 'contra la tendencia (giro): el patrón va en sentido contrario al de la media de 200',
    'volumen': 'volumen de la sesión ≥ 1,5 veces su media de 20',
    'nivel': 'en un nivel: el mínimo (alcista) o el máximo (bajista) está a ≤ 0,5 ATR de un soporte o resistencia previo',
    'lider': 'valor líder: rendimiento a 120 sesiones en el 30 % mejor (alcista) o peor (bajista) del día',
    'mercado': 'mercado a favor: más del 55 % de las acciones sobre su media de 200 (alcista) o menos del 45 % (bajista)',
    'mercado_contra': 'mercado en contra del patrón',
    'confluencia': 'confluencia: al menos 2 señales más del mismo sentido, de otros patrones, en ±2 sesiones',
    'volatilidad_baja': 'volatilidad baja: ATR / precio en el 40 % más bajo del día',
    'sobreextendido': 'sobreextendido a favor del giro: RSI(14) en el 30 % más bajo (alcista) o más alto (bajista)',
    'extremo52': 'junto al extremo anual: cerca del máximo de 52 semanas (alcista) o del mínimo (bajista)',
    'todo_favorable': 'contexto favorable: cumple al menos 4 de [tendencia, volumen, nivel, líder, mercado, confluencia]',
}


def events(D: ml.Data) -> Dict[str, np.ndarray]:
    """Señales con su fila del panel (-1 si el valor no estaba entre los más negociados ese día), sentido, familia y confluencia."""
    pdir = np.array([CATALOG[k].dir for k in ALL])
    pfam = np.array([FAMS.index(CATALOG[k].fam) for k in ALL])
    d = pdir[D.ev_pid]
    row = D.b2r[D.boff[D.ev_sym] + D.ev_bar]
    conf = np.zeros(len(d), dtype=np.int32)
    order = np.lexsort((D.ev_bar, D.ev_sym))                         # por valor y barra
    s_, b_, p_, d_ = D.ev_sym[order], D.ev_bar[order], D.ev_pid[order], d[order]
    start = np.searchsorted(s_, np.arange(len(D.syms)), side='left')
    end = np.searchsorted(s_, np.arange(len(D.syms)), side='right')
    cv = np.zeros(len(order), dtype=np.int32)
    for j in range(len(D.syms)):
        a, z = start[j], end[j]
        if z <= a:
            continue
        bb, pp, dd = b_[a:z], p_[a:z], d_[a:z]
        lo, hi = np.searchsorted(bb, bb - 2, side='left'), np.searchsorted(bb, bb + 2, side='right')
        for q in range(len(bb)):                                    # otros patrones del mismo sentido en ±2 barras
            w = slice(lo[q], hi[q])
            cv[a + q] = int(((dd[w] == dd[q]) & (pp[w] != pp[q])).sum())
    conf[order] = cv
    return {'row': row, 'dir': d, 'fam': pfam[D.ev_pid], 'pid': D.ev_pid, 'q': D.ev_q, 'conf': conf, 'sym': D.ev_sym, 'bar': D.ev_bar}


def filters(D: ml.Data, row: np.ndarray, conf: np.ndarray, d: np.ndarray) -> Dict[str, np.ndarray]:
    """Filtros booleanos por señal (los de las filas que no existen quedan en False). `row` puede ser el real o el desplazado del umbral del mejor de N."""
    ok = row >= 0
    r = np.where(ok, row, 0)
    ci = lambda name: D.ctx[r, CTX.index(name)]
    rk = lambda name: D.X_ctx[r, CTX.index(name)]
    dist200, pos52 = ci('dist200'), ci('pos52')
    bre = D.breadth_d[D.dcode[r]]
    up = d > 0
    F: Dict[str, np.ndarray] = {}
    F['tendencia'] = (d * np.sign(dist200)) > 0
    F['contra'] = (d * np.sign(dist200)) < 0
    F['volumen'] = D.volraw[r] >= 1.5
    F['nivel'] = np.where(up, D.lvl[r, 0] <= 0.5, D.lvl[r, 1] <= 0.5)
    F['lider'] = np.where(up, rk('r120') >= 0.7, rk('r120') <= 0.3)
    F['mercado'] = np.where(up, bre >= 0.55, bre <= 0.45)
    F['mercado_contra'] = np.where(up, bre <= 0.45, bre >= 0.55)
    F['confluencia'] = conf >= 2
    F['volatilidad_baja'] = rk('atrp') <= 0.4
    F['sobreextendido'] = np.where(up, rk('rsi14') <= 0.3, rk('rsi14') >= 0.7)
    F['extremo52'] = np.where(up, pos52 >= 0.8, pos52 <= 0.2)
    score = sum(F[k].astype(np.int8) for k in ('tendencia', 'volumen', 'nivel', 'lider', 'mercado', 'confluencia'))
    F['todo_favorable'] = score >= 4
    return {k: (v & ok) for k, v in F.items()}


def t_cluster(x: np.ndarray, code: np.ndarray, nd: int) -> float:
    n = len(x)
    if n < 20:
        return float('nan')
    m = x.mean()
    s = np.bincount(code, weights=x, minlength=nd)
    k = np.bincount(code, minlength=nd)
    se = np.sqrt(((s - m * k) ** 2).sum()) / n
    return float(m / se) if se > 0 else float('nan')


def _stat(x: np.ndarray, code: np.ndarray, nd: int) -> Dict:
    if len(x) < 20:
        return {'n': int(len(x))}
    return {'n': int(len(x)), 'mean': float(x.mean()), 't': t_cluster(x, code, nd), 'pos': float((x > 0).mean() * 100)}


def _groups(E: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    out = []
    for f, fn in enumerate(FAMS):
        for d, dn in ((1, 'alcistas'), (-1, 'bajistas')):
            out.append((f'{fn} {dn}', (E['fam'] == f) & (E['dir'] == d)))
    out.append(('todas alcistas', E['dir'] == 1))
    out.append(('todas bajistas', E['dir'] == -1))
    return out


def _grid_t(D: ml.Data, E: Dict[str, np.ndarray], row: np.ndarray, groups) -> np.ndarray:
    """t de todas las combinaciones grupo × (sin filtro + cada filtro) para unas filas dadas; se usa con las filas reales y con las desplazadas."""
    d = E['dir']
    F = filters(D, row, E['conf'], d)
    ok = row >= 0
    r = np.where(ok, row, 0)
    x = d * D.rel[r]
    code = D.dcode[r]
    out = []
    for _, gm in groups:
        for fm in [None] + list(FILTERS):
            m = gm & ok & (F[fm] if fm else True)
            out.append(t_cluster(x[m], code[m], D.nd) if m.sum() >= 30 else np.nan)
    return np.array(out)


def null_max_t(D: ml.Data, E: Dict[str, np.ndarray], groups, reps: int, seed: int = 7) -> np.ndarray:
    """Mayor |t| de todas las combinaciones cuando las señales de cada valor se desplazan en el tiempo al azar (circularmente), con las variables de contexto de la nueva barra."""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    nb = D.nbar[E['sym']]
    for q in range(reps):
        sh = np.array([rng.integers(30, max(int(x) - 30, 31)) for x in D.nbar])
        nbar = (E['bar'] + sh[E['sym']]) % nb
        row = D.b2r[D.boff[E['sym']] + nbar]
        t = _grid_t(D, E, row, groups)
        out[q] = np.nanmax(np.abs(t)) if np.isfinite(t).any() else 0.0
    return out


def experiment(D: ml.Data, E: Dict[str, np.ndarray], reps: int = 100, verbose: bool = False, dev_end: int = DEV_END) -> Dict:
    groups = _groups(E)
    d = E['dir']
    row = E['row']
    ok = row >= 0
    r = np.where(ok, row, 0)
    x = d * D.rel[r]
    code = D.dcode[r]
    dev = D.year[r] <= dev_end
    F = filters(D, row, E['conf'], d)
    res: Dict = {}
    for gname, gm in groups:
        base = gm & ok
        rec = {'base': _stat(x[base], code[base], D.nd), 'filtros': {}}
        for fm in FILTERS:
            m = base & F[fm]
            rec['filtros'][fm] = {'cumple': _stat(x[m], code[m], D.nd), 'cumple_desarrollo': _stat(x[m & dev], code[m & dev], D.nd), 'cumple_final': _stat(x[m & ~dev], code[m & ~dev], D.nd),
                                  'no_cumple': _stat(x[base & ~F[fm]], code[base & ~F[fm]], D.nd)}
        res[gname] = rec
    t0 = time.time()
    null = null_max_t(D, E, groups, reps)
    if verbose:
        print(f'    umbral del mejor de {len(groups) * (1 + len(FILTERS))}: p95 {np.quantile(null, 0.95):.2f} · {time.time() - t0:.0f}s', flush=True)
    return {'grupos': res, 'umbral': {'p95': round(float(np.quantile(null, 0.95)), 2), 'p99': round(float(np.quantile(null, 0.99)), 2), 'combinaciones': len(groups) * (1 + len(FILTERS)), 'reps': reps}}


# ── meta-etiquetado ───────────────────────────────────────────────────────
def meta_features(D: ml.Data, E: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    row = E['row']
    r = np.where(row >= 0, row, 0)
    d = E['dir'].astype(np.float32)
    X = [D.X_ctx[r]]
    names = list(CTX)
    al = lambda nm: d * (D.X_ctx[r, CTX.index(nm)] - 0.5)         # percentil alineado con el sentido de la señal
    for nm in ('r5', 'r20', 'r120', 'dist200', 'pos52', 'rsi14', 'volrel'):
        X.append(al(nm)[:, None]); names.append('al_' + nm)
    X.append(D.lvl[r]); names += ['dsup', 'dres']
    X.append(np.log(np.clip(D.volraw[r], 0.05, 20))[:, None]); names.append('volraw')
    X.append(E['conf'][:, None].astype(np.float32)); names.append('conf')
    X.append(D.breadth_d[D.dcode[r]][:, None].astype(np.float32)); names.append('amplitud')
    X.append(D.mkt_r20_d[D.dcode[r]][:, None].astype(np.float32)); names.append('mercado_r20')
    X.append(np.where(E['q'] >= 0, E['q'], np.nan)[:, None].astype(np.float32)); names.append('calidad')
    X.append((d[:, None] > 0).astype(np.float32)); names.append('sentido')                  # 0 bajista, 1 alcista (variable categórica: sin negativos)
    X.append(E['fam'][:, None].astype(np.float32)); names.append('familia')
    X.append(E['pid'][:, None].astype(np.float32)); names.append('patron')
    return np.hstack(X).astype(np.float32), names


META_GRID = {
    'A': dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=300, l2_regularization=10.0, max_bins=128, random_state=0),
    'B': dict(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=10.0, max_bins=128, random_state=0),
    'C': dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=500, l2_regularization=30.0, max_bins=128, random_state=0),
}


def _meta_predict(D, E, X, names, y, test_years, params, min_train=3000) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    row = E['row']
    r = np.where(row >= 0, row, 0)
    ok = row >= 0
    year = D.year[r]
    date = D.date[D.dcode[r]]
    cat = [names.index(c) for c in ('familia', 'patron', 'sentido')]
    pred = np.full(len(y), np.nan, dtype=np.float32)
    for Y in test_years:
        cut = np.datetime64(f'{Y}-01-01') - np.timedelta64(15, 'D')
        tr = ok & (date < cut)
        te = ok & (year == Y)
        if tr.sum() < min_train or te.sum() == 0:
            continue
        m = HistGradientBoostingRegressor(categorical_features=cat, **params)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def _decile_stats(D, E, pred, y, years_mask, cost: float = COST, frac: float = 0.1) -> Dict:
    """Rendimiento relativo medio de las señales que el modelo puntúa mejor (y peor) dentro de cada año, frente a la media de todas las señales ese año."""
    row = E['row']
    r = np.where(row >= 0, row, 0)
    ok = np.isfinite(pred) & years_mask
    if ok.sum() < 200:
        return {'n': int(ok.sum())}
    yr = D.year[r]
    top = np.zeros(len(pred), bool)
    bot = np.zeros(len(pred), bool)
    for Y in np.unique(yr[ok]):
        m = ok & (yr == Y)
        lo, hi = np.quantile(pred[m], [frac, 1 - frac])
        top |= m & (pred >= hi)
        bot |= m & (pred <= lo)
    code = D.dcode[r]
    ic = float(np.corrcoef(np.argsort(np.argsort(pred[ok])), np.argsort(np.argsort(y[ok])))[0, 1])
    return {'n': int(ok.sum()), 'todas': float(y[ok].mean()), 'mejor': float(y[top].mean()), 'mejor_t': t_cluster(y[top], code[top], D.nd), 'mejor_neto': float(y[top].mean() - cost), 'mejor_n': int(top.sum()),
            'peor': float(y[bot].mean()), 'peor_t': t_cluster(y[bot], code[bot], D.nd), 'ic': ic, 'ic_t': t_cluster(_zs(pred[ok]) * _zs(y[ok]), code[ok], D.nd)}


def _zs(a: np.ndarray) -> np.ndarray:
    a = np.argsort(np.argsort(a)).astype(float)
    return (a - a.mean()) / (a.std() + 1e-12)


def meta(D: ml.Data, E: Dict[str, np.ndarray], test_years: List[int], dev_end: int = DEV_END, grid: Optional[Dict[str, Dict]] = None, verbose: bool = False, min_train: int = 3000) -> Dict:
    X, names = meta_features(D, E)
    row = E['row']
    r = np.where(row >= 0, row, 0)
    y = np.clip(E['dir'] * D.rel[r], -30, 30).astype(np.float32)
    year = D.year[r]
    dev_years = [y_ for y_ in test_years if y_ <= dev_end]
    fin_years = [y_ for y_ in test_years if y_ > dev_end]
    grid = grid or META_GRID
    sel = {}
    for gname, pr in grid.items():
        pred = _meta_predict(D, E, X, names, y, dev_years, pr, min_train)
        sel[gname] = _decile_stats(D, E, pred, y, np.isin(year, dev_years))
        if verbose:
            print(f'    configuración {gname}: IC desarrollo {sel[gname].get("ic", float("nan")):+.4f} · mejor 10 % {sel[gname].get("mejor", float("nan")):+.3f} %', flush=True)
    best = max(sel, key=lambda g: sel[g].get('ic', -9))
    pred = _meta_predict(D, E, X, names, y, test_years, grid[best], min_train)
    rnd = np.random.default_rng(0).random(len(y)).astype(np.float32)
    rnd = np.where(np.isfinite(pred), rnd, np.nan)
    spans = {'desarrollo': np.isin(year, dev_years), 'final': np.isin(year, fin_years), 'todo': np.isin(year, test_years)}
    out = {'configuracion': {'elegida': best, 'en_desarrollo': sel, 'params': grid[best]}, 'años': {'desarrollo': dev_years, 'final': fin_years}, 'variables': names, 'modelo': {}, 'azar': {}, 'por_sentido': {}}
    for sp, m in spans.items():
        out['modelo'][sp] = _decile_stats(D, E, pred, y, m)
        out['azar'][sp] = _decile_stats(D, E, rnd, y, m)
        out['por_sentido'][sp] = {dn: _decile_stats(D, E, pred, y, m & (E['dir'] == d)) for d, dn in ((1, 'alcistas'), (-1, 'bajistas'))}
    return out


# ── conjunto ──────────────────────────────────────────────────────────────
def analyze(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, reps: int = 100, test_years: Optional[List[int]] = None, verbose: bool = False,
            min_train: int = 3000, grid: Optional[Dict[str, Dict]] = None) -> Dict:
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    jobs = [(s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if len(b['c']) > 400]
    work: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n_, (s, w) in enumerate(ex.map(ml._work, jobs, chunksize=4)):
                work[s] = w
                if verbose and (n_ + 1) % 100 == 0:
                    print(f'    {n_ + 1}/{len(jobs)} valores · {time.time() - t0:.0f}s', flush=True)
    else:
        for jb in jobs:
            s, w = ml._work(jb)
            work[s] = w
    D = ml.Data(work, k=k)
    E = events(D)
    years = sorted(set(int(y) for y in np.unique(D.year)))
    if test_years is None:
        test_years = [y for y in years if y >= years[0] + 3 and int((D.year == y).sum()) > 20000]
    out = {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(D.y)), 'señales': int((E['row'] >= 0).sum()), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k,
                    'horizonte': ml.HZ, 'desarrollo_hasta': DEV_END, 'test_years': test_years, 'filtros': FILTERS},
           'experimento': experiment(D, E, reps=reps, verbose=verbose), 'meta_etiquetado': meta(D, E, test_years, verbose=verbose, min_train=min_train, grid=grid)}
    if verbose:
        print(f'  contexto y confluencia: {out["meta"]["señales"]:,} señales · {time.time() - t0:.0f}s', flush=True)
    return out


def report(ev: Dict) -> str:
    m, ex = ev['meta'], ev['experimento']
    p95 = ex['umbral']['p95']
    L = [f"CONTEXTO Y CONFLUENCIA · {m['universe']} acciones · {m['señales']:,} señales de patrón (las {m['k']} más negociadas) · {m['from']} → {m['to']} · exceso = rendimiento a {m['horizonte']} sesiones menos el del conjunto ese día, con el signo del sentido",
         f"Umbral del mejor de {ex['umbral']['combinaciones']} combinaciones: |t| ≥ {p95} (p95) / {ex['umbral']['p99']} (p99). Desarrollo ≤ {m['desarrollo_hasta']}, final después. ★ = supera el umbral y mantiene el signo en el final.",
         '']
    for g, rec in ex['grupos'].items():
        b = rec['base']
        if b.get('n', 0) < 30:
            continue
        L.append(f"=== {g} · sin filtrar: n {b['n']:,} · exceso {b['mean']:+.3f} % (t {b['t']:+.1f}) ===")
        rows = []
        for f, s in rec['filtros'].items():
            c, dv, fi, nc = s['cumple'], s['cumple_desarrollo'], s['cumple_final'], s['no_cumple']
            if c.get('n', 0) < 30 or 'mean' not in c:
                continue
            star = '★' if abs(c['t']) >= p95 and fi.get('mean') is not None and np.sign(fi['mean']) == np.sign(c['mean']) else ' '
            rows.append((abs(c['t']), f"  {star} {f:18s} cumple: n {c['n']:7,} exceso {c['mean']:+.3f} % (t {c['t']:+5.1f})  desarrollo {dv.get('mean', float('nan')):+.3f} %  final {fi.get('mean', float('nan')):+.3f} % (n {fi.get('n', 0):,})   no cumple: {nc.get('mean', float('nan')):+.3f} %"))
        for _, line in sorted(rows, reverse=True):
            L.append(line)
        L.append('')
    mm = ev['meta_etiquetado']
    L.append(f"=== META-ETIQUETADO (gradient boosting; configuración {mm['configuracion']['elegida']} elegida con {mm['años']['desarrollo']}; juzgada en {mm['años']['final']}) ===")
    head = f"{'':12s}{'n':>9s}{'todas':>9s} | {'mejor 10%':>9s}{'t':>7s}{'neto':>9s} | {'peor 10%':>9s}{'t':>7s} | {'IC':>8s}{'t':>7s}"
    for sp in ('final', 'desarrollo', 'todo'):
        s = mm['modelo'].get(sp, {})
        if 'mejor' not in s:
            continue
        a = mm['azar'].get(sp, {})
        L += ['', f'--- {sp} ---', head,
              f"{'modelo':12s}{s['n']:9,}{s['todas']:+8.3f}% | {s['mejor']:+8.3f}%{s['mejor_t']:+7.1f}{s['mejor_neto']:+8.3f}% | {s['peor']:+8.3f}%{s['peor_t']:+7.1f} | {s['ic']:+8.4f}{s['ic_t']:+7.1f}"]
        if 'mejor' in a:
            L.append(f"{'azar':12s}{a['n']:9,}{a['todas']:+8.3f}% | {a['mejor']:+8.3f}%{a['mejor_t']:+7.1f}{a['mejor_neto']:+8.3f}% | {a['peor']:+8.3f}%{a['peor_t']:+7.1f} | {a['ic']:+8.4f}{a['ic_t']:+7.1f}")
        for dn, v in mm['por_sentido'].get(sp, {}).items():
            if 'mejor' in v:
                L.append(f"{'  ' + dn:12s}{v['n']:9,}{v['todas']:+8.3f}% | {v['mejor']:+8.3f}%{v['mejor_t']:+7.1f}{v['mejor_neto']:+8.3f}% | {v['peor']:+8.3f}%{v['peor_t']:+7.1f} | {v['ic']:+8.4f}{v['ic_t']:+7.1f}")
    return '\n'.join(L)


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, reps: int = 100) -> Dict:
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    ev = analyze(bars, workers=workers, reps=reps, verbose=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'context.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'context_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
