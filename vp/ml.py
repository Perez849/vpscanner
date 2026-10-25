"""
Capa de aprendizaje automático, evaluada SOLO fuera de muestra y con las mismas reglas que el estudio de evidencia.

Pregunta: ¿puede un modelo que combine el contexto del precio, las velas (con su consenso) y las figuras ordenar las acciones por rendimiento futuro mejor que el azar?

  · Cada fila es una acción y una fecha (las 300 más negociadas en esa fecha). Objetivo: percentil, dentro de ese día, del rendimiento a 10 sesiones
    (apertura siguiente → cierre a 10 sesiones). Así se quita el mercado y se mide solo qué acción rinde más que las demás.
  · Variables, todas causales (usan datos hasta el cierre de la fecha): rendimientos a 1–120 sesiones, RSI, distancias a medias, posición en el rango anual, volatilidad,
    volumen, forma de la vela; cada una como percentil del día. Más los patrones: las formaciones de velas de hoy, el recuento de velas alcistas y bajistas (consenso),
    las figuras y rupturas de las últimas 5 sesiones.
  · Modelo: gradient boosting con parámetros fijados de antemano (sin ajustarlos mirando el resultado). Validación hacia delante: para predecir el año Y se entrena con
    todo lo anterior a enero de Y (menos 15 días para que no se cuele el rendimiento a 10 sesiones) y nunca se vuelve atrás.
  · Se comparan cuatro conjuntos: solo contexto, + velas, + figuras y rupturas, y todo. Si los patrones ayudan, el conjunto completo ha de rendir mejor que el de contexto solo.
  · Medidas por fecha: correlación de rangos (IC) entre la predicción y el rendimiento relativo, y rendimiento del 10 % mejor y del 10 % peor según el modelo.
    t de Newey–West (retardo 9) porque los rendimientos a 10 sesiones de fechas contiguas se solapan.
  · Referencias: los factores simples de siempre (reversión a 5 sesiones, impulso, RSI de 2) en las mismas fechas, como control positivo del método.
"""
from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import engine, ta
from .catalog import CATALOG, HQ
from .ta import shift

HZ = 10                    # horizonte del objetivo (sesiones)
COST = 0.15                # % de ida y vuelta por operación
K = 300                    # se opera solo con las K más negociadas de cada fecha
MEM_WINDOW = 60
RECENT = 5                 # una figura o ruptura cuenta durante 5 sesiones
PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=10.0, max_bins=64, random_state=0)   # fijados de antemano
GROUPS = ('ctx', 'ctx+velas', 'ctx+figuras', 'todo')

CTX = ['r1', 'r5', 'r10', 'r20', 'r60', 'r120', 'rsi2', 'rsi14', 'dist20', 'dist50', 'dist200', 'slope50', 'pos52', 'dd252', 'atrp', 'volrel', 'dvol', 'rvol',
       'gap', 'rng', 'body', 'clpos']


def _ctx(F: Dict[str, np.ndarray]) -> np.ndarray:
    """Variables de contexto por barra (n × len(CTX)); NaN donde aún no hay historia."""
    o, h, l, c, v = F['o'], F['h'], F['l'], F['c'], F['v']
    n = len(c)
    with np.errstate(invalid='ignore', divide='ignore'):
        lc = np.log(c)
        r = lambda k: lc - shift(lc, k)
        atr = F['atr']
        rng = h - l
        v20 = ta.sma(v, 20)
        dv = c * v
        r1 = r(1)
        rv = ta.rstd(np.nan_to_num(r1), 20)
        cols = {
            'r1': r1, 'r5': r(5), 'r10': r(10), 'r20': r(20), 'r60': r(60), 'r120': r(120), 'rsi2': F['rsi2'], 'rsi14': F['rsi14'],
            'dist20': c / F['sma20'] - 1, 'dist50': c / F['sma50'] - 1, 'dist200': c / F['sma200'] - 1, 'slope50': F['sma50'] / shift(F['sma50'], 10) - 1,
            'pos52': F['pos52'], 'dd252': c / ta.rmax(h, 252) - 1, 'atrp': F['atrp'], 'volrel': np.log(v / v20), 'dvol': np.log(ta.sma(dv, 60)),
            'rvol': rv / (F['atrp'] / 100.0), 'gap': o / shift(c, 1) - 1, 'rng': rng / atr, 'body': (c - o) / atr, 'clpos': np.where(rng > 0, (c - l) / rng, 0.5),
        }
    return np.column_stack([cols[k] for k in CTX]).astype(np.float32)


def _patterns(F: Dict[str, np.ndarray], events: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Variables de patrones por barra: formaciones de vela de hoy, figuras y rupturas de las últimas RECENT sesiones, recuentos de consenso y calidad."""
    n = len(F['c'])
    names = sorted(k for k in CATALOG)
    col = {k: j for j, k in enumerate(names)}
    M = np.zeros((n, len(names)), dtype=np.uint8)
    hq = np.zeros((n, 2), dtype=np.uint8)               # figura de calidad alta alcista / bajista
    for e in events:
        j = col.get(e['name'])
        if j is None:
            continue
        i = e['i']
        M[i, j] = 1
        if e['q'] is not None and e['q'] >= HQ:
            hq[i, 0 if CATALOG[e['name']].dir > 0 else 1] = 1
    out = np.zeros_like(M)
    cs = np.cumsum(np.vstack([np.zeros((1, M.shape[1]), np.int32), M]), axis=0, dtype=np.int32)
    for k, j in col.items():
        w = 1 if CATALOG[k].fam == 'velas' else RECENT
        i = np.arange(n)
        out[:, j] = (cs[i + 1, j] - cs[np.maximum(i + 1 - w, 0), j]) > 0
    csh = np.cumsum(np.vstack([np.zeros((1, 2), np.int32), hq]), axis=0, dtype=np.int32)
    i = np.arange(n)
    hqr = ((csh[i + 1] - csh[np.maximum(i + 1 - RECENT, 0)]) > 0).astype(np.uint8)
    fam = np.array([CATALOG[k].fam for k in names])
    dr = np.array([CATALOG[k].dir for k in names])
    cnt = lambda f, d: out[:, (fam == f) & (dr == d) & ~np.array([k.startswith('consenso') for k in names])].sum(axis=1) if ((fam == f) & (dr == d)).any() else np.zeros(n)
    extra = np.column_stack([cnt('velas', 1), cnt('velas', -1), cnt('figuras', 1), cnt('figuras', -1), cnt('rupturas', 1), cnt('rupturas', -1), hqr[:, 0], hqr[:, 1]]).astype(np.uint8)
    names2 = names + ['n_velas_up', 'n_velas_dn', 'n_fig_up', 'n_fig_dn', 'n_rup_up', 'n_rup_dn', 'hq_up', 'hq_dn']
    return np.hstack([out, extra]), names2


def _work(args: Tuple[str, Dict[str, np.ndarray]]) -> Tuple[str, Dict]:
    sym, b = args
    F = ta.prepare(b)
    events, _ = engine.scan(F)
    n = len(F['c'])
    o, c = F['o'], F['c']
    ret = np.full(n, np.nan)
    if n > HZ + 1:
        m = n - HZ                                               # señal en t: entrada en la apertura de t+1 y salida en el cierre de t+HZ
        ret[:m] = (c[HZ:HZ + m] / o[1:1 + m] - 1.0) * 100.0
    dv = np.full(n, np.nan)
    if n >= MEM_WINDOW:
        cs = np.cumsum(np.insert(c * F['v'], 0, 0.0))
        dv[MEM_WINDOW - 1:] = (cs[MEM_WINDOW:] - cs[:-MEM_WINDOW]) / MEM_WINDOW
    P, pnames = _patterns(F, events)
    return sym, {'t': (F['t'] // 86400).astype(np.int64), 'ret': ret, 'dv': dv, 'ctx': _ctx(F), 'pat': P, 'pnames': pnames}


class Data:
    """Panel con solo las filas de las K más negociadas de cada fecha y con historia suficiente."""

    def __init__(self, work: Dict[str, Dict], k: int = K, min_hist: int = 130):
        syms = list(work)
        lens = np.array([len(work[s]['t']) for s in syms])
        day = np.concatenate([work[s]['t'] for s in syms])
        days = np.unique(day)
        dmat = np.full((len(days), len(syms)), np.nan, dtype=np.float32)
        for j, s in enumerate(syms):
            dmat[np.searchsorted(days, work[s]['t']), j] = work[s]['dv']
        rank = np.argsort(np.argsort(-np.nan_to_num(dmat, nan=-1.0), axis=1), axis=1)
        memb = (rank < k) & np.isfinite(dmat)
        rows = []
        for j, s in enumerate(syms):
            w = work[s]
            di = np.searchsorted(days, w['t'])
            hist = np.arange(len(w['t'])) >= min_hist
            m = memb[di, j] & hist & np.isfinite(w['ret'])
            rows.append(np.flatnonzero(m))
        self.syms = syms
        self.pnames = work[syms[0]]['pnames']
        self.sym = np.concatenate([np.full(len(r), j, np.int32) for j, r in enumerate(rows)])
        self.day = np.concatenate([work[s]['t'][r] for s, r in zip(syms, rows)])
        self.ret = np.concatenate([work[s]['ret'][r] for s, r in zip(syms, rows)]).astype(np.float64)
        self.ctx = np.vstack([work[s]['ctx'][r] for s, r in zip(syms, rows)])
        self.pat = np.vstack([work[s]['pat'][r] for s, r in zip(syms, rows)])
        d_ = np.unique(self.day)
        self.dcode = np.searchsorted(d_, self.day)
        self.nd = len(d_)
        self.date = d_.astype('datetime64[D]')
        self.year = (self.day.astype('datetime64[D]').astype('datetime64[Y]').astype(int) + 1970)
        # percentil del día de cada variable de contexto y del rendimiento (el modelo solo ve percentiles: no hay niveles absolutos de mercado)
        self.X_ctx = self._rank_by_day(self.ctx)
        self.y = self._rank_by_day(self.ret[:, None])[:, 0] - 0.5
        cnt = np.bincount(self.dcode, minlength=self.nd)
        sm = np.bincount(self.dcode, weights=self.ret, minlength=self.nd)
        self.rel = self.ret - (sm / np.maximum(cnt, 1))[self.dcode]               # rendimiento relativo (bruto) al promedio del día

    def _rank_by_day(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        df = pd.DataFrame(X.astype(np.float64))
        df['d'] = self.dcode
        r = df.groupby('d').rank(pct=True)
        return r.to_numpy(dtype=np.float32)

    def matrix(self, group: str) -> Tuple[np.ndarray, List[str]]:
        pn = self.pnames
        vel = [j for j, k in enumerate(pn) if k in CATALOG and CATALOG[k].fam == 'velas'] + [pn.index('n_velas_up'), pn.index('n_velas_dn')]
        fig = [j for j, k in enumerate(pn) if (k in CATALOG and CATALOG[k].fam != 'velas')] + [pn.index(x) for x in ('n_fig_up', 'n_fig_dn', 'n_rup_up', 'n_rup_dn', 'hq_up', 'hq_dn')]
        idx = {'ctx': [], 'ctx+velas': vel, 'ctx+figuras': fig, 'todo': vel + fig}[group]
        X = np.hstack([self.X_ctx, self.pat[:, idx].astype(np.float32)]) if idx else self.X_ctx
        return X, CTX + [pn[j] for j in idx]


# ── evaluación ────────────────────────────────────────────────────────────
def nw_t(x: np.ndarray, lag: int = 9) -> float:
    """t de la media con errores estándar de Newey–West."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float('nan')
    m = x.mean()
    u = x - m
    s = (u * u).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - L / (lag + 1.0)) * (u[L:] * u[:-L]).sum() / n
    return float(m / np.sqrt(max(s, 1e-18) / n))


def daily_metrics(score: np.ndarray, dcode: np.ndarray, rel: np.ndarray, nd: int, cost: float = COST, q: float = 0.10) -> Dict[str, np.ndarray]:
    """Por fecha: IC de rangos entre la puntuación y el rendimiento relativo, y rendimiento relativo del q mejor y del q peor según la puntuación."""
    import pandas as pd
    df = pd.DataFrame({'d': dcode, 's': score, 'r': rel})
    df['ps'] = df.groupby('d')['s'].rank(pct=True)
    df['pr'] = df.groupby('d')['r'].rank(pct=True)
    g = df.groupby('d')
    ic = g.apply(lambda x: np.corrcoef(x['ps'], x['pr'])[0, 1] if len(x) > 20 and x['ps'].std() > 0 else np.nan, include_groups=False)
    top = df[df['ps'] >= 1 - q].groupby('d')['r'].mean()
    bot = df[df['ps'] <= q].groupby('d')['r'].mean()
    out = {k: np.full(nd, np.nan) for k in ('ic', 'top', 'bot')}
    out['ic'][ic.index.to_numpy()] = ic.to_numpy()
    out['top'][top.index.to_numpy()] = top.to_numpy()
    out['bot'][bot.index.to_numpy()] = bot.to_numpy()
    return out


def summarize(dm: Dict[str, np.ndarray], mask_days: np.ndarray, cost: float = COST) -> Dict:
    ic, top, bot = dm['ic'][mask_days], dm['top'][mask_days], dm['bot'][mask_days]
    ok = np.isfinite(ic) & np.isfinite(top) & np.isfinite(bot)
    ic, top, bot = ic[ok], top[ok], bot[ok]
    spread = top - bot
    return {'dias': int(ok.sum()), 'ic': float(ic.mean()), 'ic_t': nw_t(ic), 'ic_pos': float((ic > 0).mean() * 100),
            'top': float(top.mean()), 'top_t': nw_t(top), 'top_neto': float(top.mean() - cost), 'bot': float(bot.mean()),
            'spread': float(spread.mean()), 'spread_t': nw_t(spread), 'spread_neto': float(spread.mean() - 2 * cost)}


def walk_forward(D: Data, group: str, test_years: List[int], params: Optional[Dict] = None, train_stride: int = 2, verbose: bool = False, min_train: int = 50000) -> np.ndarray:
    """Puntuación fuera de muestra de cada fila de los años de prueba (NaN en el resto)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, _ = D.matrix(group)
    pred = np.full(len(D.y), np.nan, dtype=np.float32)
    for Y in test_years:
        cut = np.datetime64(f'{Y}-01-01') - np.timedelta64(15, 'D')
        tr = (D.date[D.dcode] < cut) & (D.dcode % train_stride == 0)
        te = D.year == Y
        if tr.sum() < min_train or te.sum() == 0:
            continue
        t0 = time.time()
        m = HistGradientBoostingRegressor(**(params or PARAMS))
        m.fit(X[tr], D.y[tr])
        pred[te] = m.predict(X[te])
        if verbose:
            print(f'    {group} · {Y}: entrena {int(tr.sum()):,} filas, predice {int(te.sum()):,} · {time.time() - t0:.0f}s', flush=True)
    return pred


def baselines(D: Data, mask_rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Factores simples con el signo «esperado» de la literatura, como control positivo del método (sobre las mismas filas)."""
    j = {k: i for i, k in enumerate(CTX)}
    return {'reversión 5 sesiones (−r5)': -D.X_ctx[:, j['r5']], 'impulso 12-1 meses (r120)': D.X_ctx[:, j['r120']], 'RSI(2) bajo (−rsi2)': -D.X_ctx[:, j['rsi2']],
            'cerca del máximo anual (pos52)': D.X_ctx[:, j['pos52']], 'baja volatilidad (−atrp)': -D.X_ctx[:, j['atrp']], 'azar': np.random.default_rng(0).random(len(D.y))}


def analyze(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, test_years: Optional[List[int]] = None, params: Optional[Dict] = None, verbose: bool = False,
            k: int = K, groups: Tuple[str, ...] = GROUPS, train_stride: int = 2, min_train: int = 50000, min_rows: int = 20000) -> Dict:
    t0 = time.time()
    jobs = [(s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if len(b['c']) > 400]
    work: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n_, (s, w) in enumerate(ex.map(_work, jobs, chunksize=4)):
                work[s] = w
                if verbose and (n_ + 1) % 100 == 0:
                    print(f'    {n_ + 1}/{len(jobs)} valores · {time.time() - t0:.0f}s', flush=True)
    else:
        for jb in jobs:
            s, w = _work(jb)
            work[s] = w
    D = Data(work, k=k)
    years = sorted(set(int(y) for y in np.unique(D.year)))
    if test_years is None:
        test_years = [y for y in years if y >= years[0] + 3 and int((D.year == y).sum()) > min_rows]
    out: Dict = {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(D.y)), 'fechas': int(D.nd), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k,
                          'horizonte': HZ, 'coste': COST, 'test_years': test_years, 'params': params or PARAMS, 'variables': {'contexto': len(CTX), 'patrones': len(D.pnames)}},
                 'modelos': {}, 'base': {}, 'por_año': {}}
    test_rows = np.isin(D.year, test_years)
    day_mask = np.zeros(D.nd, bool)
    day_mask[np.unique(D.dcode[test_rows])] = True
    for name, score in baselines(D, test_rows).items():
        sc = np.where(test_rows, score, np.nan)
        out['base'][name] = summarize(daily_metrics(np.nan_to_num(sc, nan=0.0)[test_rows], D.dcode[test_rows], D.rel[test_rows], D.nd), day_mask)
    for g in groups:
        pred = walk_forward(D, g, test_years, params, train_stride=train_stride, verbose=verbose, min_train=min_train)
        ok = np.isfinite(pred)
        dm = daily_metrics(pred[ok], D.dcode[ok], D.rel[ok], D.nd)
        out['modelos'][g] = summarize(dm, day_mask)
        out['por_año'][g] = {}
        for Y in test_years:
            mk = np.zeros(D.nd, bool)
            mk[np.unique(D.dcode[D.year == Y])] = True
            out['por_año'][g][str(Y)] = summarize(dm, mk)
    if verbose:
        print(f'  aprendizaje automático: {len(D.y):,} filas · {time.time() - t0:.0f}s', flush=True)
    return out


def report(ev: Dict) -> str:
    m = ev['meta']
    L = [f"APRENDIZAJE AUTOMÁTICO · {m['universe']} acciones · {m['filas']:,} filas (las {m['k']} más negociadas) · {m['from']} → {m['to']} · objetivo: rendimiento relativo a {m['horizonte']} sesiones · "
         f"validación hacia delante en {', '.join(map(str, m['test_years']))} · {m['variables']['contexto']} variables de contexto y {m['variables']['patrones']} de patrones",
         'IC = correlación de rangos diaria entre la puntuación y el rendimiento relativo; «mejor/peor 10 %» = rendimiento relativo medio (bruto) de las acciones mejor y peor puntuadas; t de Newey–West.',
         f"{'':34s}{'IC':>8s}{'t':>7s}{'+días':>7s} | {'mejor 10%':>10s}{'t':>7s}{'neto':>8s} | {'peor 10%':>9s} | {'spread':>8s}{'t':>7s}{'neto':>8s}"]
    row = lambda n, s: f"{n:34s}{s['ic']:+8.4f}{s['ic_t']:+7.1f}{s['ic_pos']:6.0f}% | {s['top']:+9.3f}%{s['top_t']:+7.1f}{s['top_neto']:+7.3f}% | {s['bot']:+8.3f}% | {s['spread']:+7.3f}%{s['spread_t']:+7.1f}{s['spread_neto']:+7.3f}%"
    L.append('--- referencias (control positivo del método) ---')
    for n, s in ev['base'].items():
        L.append(row(n, s))
    L.append('--- modelos (fuera de muestra) ---')
    for n, s in ev['modelos'].items():
        L.append(row(n, s))
    L.append('--- IC por año y modelo ---')
    for n, ys in ev['por_año'].items():
        L.append(f'{n:14s}' + ' '.join(f"{y}:{s['ic']:+.3f}" for y, s in ys.items()))
    L.append('--- mejor 10 % (rendimiento relativo bruto, %) por año y modelo ---')
    for n, ys in ev['por_año'].items():
        L.append(f'{n:14s}' + ' '.join(f"{y}:{s['top']:+.2f}" for y, s in ys.items()))
    return '\n'.join(L)


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2) -> Dict:
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    ev = analyze(bars, workers=workers, verbose=True, train_stride=3)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'ml.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'ml_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
