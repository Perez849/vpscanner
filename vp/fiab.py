"""
Modelo de fiabilidad (parte 2): ¿se puede saber DE ANTEMANO qué operaciones tienen más probabilidad de acabar bien con la escalera del operador?

  · Fila = acción y fecha (las 300 más negociadas ese día). Objetivo: 1 si la escalera en largo (stop inicial 2,5 ATR; al cerrar a +4 % el stop sube a +4 % y se deja correr; al cerrar
    a +10 % sube a +10 % y persigue a 2 ATR; 60 sesiones) acabó con ganancia neta, 0 si no. Solo filas con la operación ya cerrada.
  · Variables (todas con datos hasta el cierre de la fecha): contexto del valor como percentil del día (rendimientos a 1–250 sesiones, RSI, distancias a medias, posición anual,
    volatilidad, volumen, hueco, forma de la vela), recuento de patrones recientes (velas, figuras y rupturas alcistas y bajistas, calidad alta), amplitud del mercado y su
    rendimiento a 20 sesiones. El modelo solo ve percentiles: no hay niveles absolutos que cambien de régimen.
  · Modelo: gradient boosting (clasificador) con parámetros fijados de antemano. Walk-forward por años: para predecir el año Y se entrena con las filas cuya operación estaba cerrada
    antes de empezar Y (fecha + 95 días naturales < 1 de enero), nunca se vuelve atrás.
  · Evaluación fuera de muestra: AUC; calibración por tramos de probabilidad predicha (si el modelo dice 70 %, ¿gana el 70 %?); por decil de probabilidad dentro de cada día: ganancia
    realizada, ganancia relativa al día (fiabilidad), neto y exceso; t con errores agrupados en bloques de 60 sesiones. Y lo mismo dentro de las filas de la regla candidata (C6).
  · Control «solo volatilidad»: el mismo modelo con una sola variable (percentil de ATR/precio). Si el modelo completo no gana a este control, la «fiabilidad» es solo que el valor se mueve.
"""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ml, search
from .ml import CTX

PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=10.0, max_bins=64, random_state=0)   # fijados de antemano
EXIT = 'esc'               # la escalera del operador con stop inicial a 2,5 ATR
PURGE_DAYS = 95            # días naturales entre la última fila de entrenamiento y el año de prueba (60 sesiones de la operación más margen)
BLOCK = 60                 # sesiones por grupo de error (la operación dura hasta 60)
BINS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)
PAT_COLS = ('n_velas_up', 'n_velas_dn', 'n_fig_up', 'n_fig_dn', 'n_rup_up', 'n_rup_dn', 'hq_up', 'hq_dn')
FEATS = list(CTX) + list(PAT_COLS) + ['amplitud', 'mercado_r20', 'atrp_bruto']


def features(D: ml.Data) -> np.ndarray:
    """Matriz de variables por fila (float32): percentiles de contexto, recuento de patrones, amplitud del mercado y volatilidad bruta."""
    pn = {k: j for j, k in enumerate(D.pnames)}
    pat = np.column_stack([D.pat[:, pn[k]] for k in PAT_COLS]).astype(np.float32)
    extra = np.column_stack([D.breadth_d[D.dcode], D.mkt_r20_d[D.dcode], D.ctx[:, CTX.index('atrp')]]).astype(np.float32)
    return np.hstack([D.X_ctx.astype(np.float32), pat, extra])


def target(P: search.Panel, exit_name: str = EXIT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, válido, neto) de la escalera en largo: y = 1 si la operación acabó con ganancia neta."""
    e = search.EXN.index(exit_name)
    net = P.net[:, 0, e]
    ok = np.isfinite(net)
    return (net > 0).astype(np.int8), ok, net


def _fit(X: np.ndarray, y: np.ndarray, params: Dict):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(**params)
    m.fit(X, y)
    return m


def walk_forward(D: ml.Data, X: np.ndarray, y: np.ndarray, ok: np.ndarray, test_years: List[int], params: Optional[Dict] = None, verbose: bool = False, min_train: int = 20000) -> np.ndarray:
    """Probabilidad fuera de muestra de cada fila de los años de prueba (NaN en el resto)."""
    pred = np.full(len(y), np.nan, dtype=np.float32)
    dates = D.date[D.dcode]
    for Y in test_years:
        cut = np.datetime64(f'{Y}-01-01') - np.timedelta64(PURGE_DAYS, 'D')
        tr = ok & (dates < cut)
        te = (D.year == Y)
        if tr.sum() < min_train or not te.any():
            continue
        t0 = time.time()
        m = _fit(X[tr], y[tr], params or PARAMS)
        pred[te] = m.predict_proba(X[te])[:, 1]
        if verbose:
            print(f'    fiabilidad · {Y}: entrenado con {int(tr.sum()):,} filas cerradas antes del {str(cut)}, predice {int(te.sum()):,} · {time.time() - t0:.0f}s', flush=True)
    return pred


def fit_final(D: ml.Data, P: search.Panel, params: Optional[Dict] = None):
    """Modelo entrenado con las filas cerradas al menos PURGE_DAYS antes del último día (para puntuar las de hoy): en los últimos días solo están cerradas las operaciones que salieron pronto."""
    X = features(D)
    y, ok, _ = target(P)
    dates = D.date[D.dcode]
    tr = ok & (dates < dates.max() - np.timedelta64(PURGE_DAYS, 'D'))
    return _fit(X[tr], y[tr], params or PARAMS)


def _t_blocks(x: np.ndarray, grp: np.ndarray) -> float:
    n = len(x)
    if n < 30:
        return float('nan')
    m = x.mean()
    s = np.bincount(grp, weights=x - m)
    se = np.sqrt((s ** 2).sum()) / n
    return float(m / se) if se > 0 else float('nan')


def _deciles_by_day(pred: np.ndarray, dcode: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Decil (0–9) de la probabilidad predicha dentro de cada día (solo filas de `mask`); -1 fuera."""
    import pandas as pd
    out = np.full(len(pred), -1, np.int8)
    ix = np.flatnonzero(mask)
    df = pd.DataFrame({'d': dcode[ix], 'p': pred[ix]})
    r = df.groupby('d')['p'].rank(pct=True, method='first').to_numpy()
    out[ix] = np.minimum((r * 10).astype(int), 9)
    return out


def evaluate(D: ml.Data, P: search.Panel, pred: np.ndarray, y: np.ndarray, ok: np.ndarray, net: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    """Métricas fuera de muestra sobre las filas con predicción y resultado (y, opcionalmente, dentro de `mask`)."""
    from sklearn.metrics import roc_auc_score
    e = search.EXN.index(EXIT)
    rel, relw = P.rel[:, 0, e], P.relw[:, 0, e]
    m = ok & np.isfinite(pred) & np.isfinite(relw) & (mask if mask is not None else True)
    if m.sum() < 100 or len(np.unique(y[m])) < 2:
        return {'n': int(m.sum())}
    grp = D.dcode // BLOCK
    base = float(y[m].mean())
    out = {'n': int(m.sum()), 'auc': round(float(roc_auc_score(y[m], pred[m])), 4), 'base': round(100 * base, 2), 'pred_media': round(100 * float(pred[m].mean()), 2)}
    # calibración por tramos de probabilidad predicha
    cal = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        b = m & (pred >= lo) & (pred < hi)
        if b.sum() >= 200:
            cal.append({'tramo': f'{int(lo * 100)}–{int(min(hi, 1) * 100)} %', 'n': int(b.sum()), 'predicho': round(100 * float(pred[b].mean()), 1), 'real': round(100 * float(y[b].mean()), 1),
                        'fiab': round(100 * float(relw[b].mean()), 1), 'neto': round(float(net[b].mean()), 3), 'exceso': round(float(rel[b].mean()), 3)})
    out['calibracion'] = cal
    # deciles dentro del día
    dec = _deciles_by_day(pred, D.dcode, m)
    rows = []
    for q in range(10):
        b = dec == q
        if b.sum() < 100:
            continue
        rows.append({'decil': q + 1, 'n': int(b.sum()), 'predicho': round(100 * float(pred[b].mean()), 1), 'real': round(100 * float(y[b].mean()), 1), 'fiab': round(100 * float(relw[b].mean()), 1),
                     'fiab_t': round(_t_blocks(relw[b], grp[b]), 2), 'neto': round(float(net[b].mean()), 3), 'exceso': round(float(rel[b].mean()), 3), 'exceso_t': round(_t_blocks(rel[b], grp[b]), 2)})
    out['deciles'] = rows
    top, bot = dec == 9, dec == 0
    if top.sum() >= 100 and bot.sum() >= 100:
        out['top'] = {'n': int(top.sum()), 'real': round(100 * float(y[top].mean()), 1), 'fiab': round(100 * float(relw[top].mean()), 1), 'fiab_t': round(_t_blocks(relw[top], grp[top]), 2),
                      'neto': round(float(net[top].mean()), 3), 'exceso': round(float(rel[top].mean()), 3), 'exceso_t': round(_t_blocks(rel[top], grp[top]), 2)}
        out['bot'] = {'n': int(bot.sum()), 'real': round(100 * float(y[bot].mean()), 1), 'fiab': round(100 * float(relw[bot].mean()), 1), 'neto': round(float(net[bot].mean()), 3), 'exceso': round(float(rel[bot].mean()), 3)}
        out['top_menos_bot'] = round(float(100 * (y[top].mean() - y[bot].mean())), 1)
    return out


def analyze(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, test_years: Optional[List[int]] = None, verbose: bool = False, join: Optional[Dict[str, str]] = None,
            params: Optional[Dict] = None, min_train: int = 20000, rule=None) -> Dict:
    t0 = time.time()
    D, P, dropped = search.build(bars, workers, k, verbose, search.MIN_DAY, search.MIN_COMBO, join, search.COST, library=False)
    X = features(D)
    y, ok, net = target(P)
    years = sorted(int(v) for v in np.unique(D.year))
    if test_years is None:
        test_years = [v for v in years if v >= years[0] + 3 and int((D.year == v).sum()) > 20000]
    pred = walk_forward(D, X, y, ok, test_years, params, verbose, min_train)
    jv = CTX.index('atrp')
    pred_vol = walk_forward(D, X[:, [jv]], y, ok, test_years, params, False, min_train)
    rule = rule or {'regla': ['bajo_200', 'impulso12_alto'], 'nombre': 'C6'}
    cond = search.conditions(D)
    mr = np.ones(len(y), bool)
    for c_ in rule['regla']:
        mr &= cond[c_]
    out = {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(y)), 'cerradas': int(ok.sum()), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k, 'salida': EXIT,
                    'texto_salida': search.EXIT_TXT[EXIT], 'test_years': test_years, 'variables': FEATS, 'params': params or PARAMS, 'purga_dias': PURGE_DAYS, 'miembros': 'en la fecha' if join else 'actuales',
                    'barras_quitadas': dropped, 'regla': rule},
           'modelo': {'todo': evaluate(D, P, pred, y, ok, net), 'por_año': {int(Y): evaluate(D, P, pred, y, ok, net, D.year == Y) for Y in test_years}, 'en_regla': evaluate(D, P, pred, y, ok, net, mr)},
           'solo_volatilidad': {'todo': evaluate(D, P, pred_vol, y, ok, net), 'en_regla': evaluate(D, P, pred_vol, y, ok, net, mr)}}
    if verbose:
        print(f'  modelo de fiabilidad: {int(ok.sum()):,} filas cerradas · AUC {out["modelo"]["todo"].get("auc")} (solo volatilidad {out["solo_volatilidad"]["todo"].get("auc")}) · {time.time() - t0:.0f}s', flush=True)
    return out


def report(ev: Dict) -> str:
    m = ev['meta']
    L = [f"MODELO DE FIABILIDAD · {m['universe']} acciones · {m['filas']:,} filas (las {m['k']} más negociadas), {m['cerradas']:,} con la operación cerrada · {m['from']} → {m['to']} · salida: {m['texto_salida']}",
         f"Objetivo: la operación acabó con ganancia neta. Walk-forward por años ({m['test_years'][0]}–{m['test_years'][-1]}): se entrena con filas cerradas al menos {m['purga_dias']} días antes del año y se predice el año. "
         f"Fiabilidad = ganó menos la fracción de filas del día que ganó con la misma salida (puntos). t con bloques de {BLOCK} sesiones. Universo: {'miembros actuales desde su incorporación al S&P 500' if m['miembros'] == 'en la fecha' else 'miembros actuales'}.", '']
    def block(title, r):
        if 'auc' not in r:
            L.append(f'{title}: n {r.get("n", 0)} (insuficiente)'); return
        L.append(f"{title}: n {r['n']:,} · AUC {r['auc']:.3f} · ganan {r['base']:.1f} % (predicho medio {r['pred_media']:.1f} %)")
        if r.get('top'):
            t, b = r['top'], r['bot']
            L.append(f"   decil más fiable de cada día: ganan {t['real']:.1f} % (fiabilidad {t['fiab']:+.1f} puntos, t {t['fiab_t']:+.1f}) · neto {t['neto']:+.3f} % · exceso {t['exceso']:+.3f} % (t {t['exceso_t']:+.1f}) | decil menos fiable: ganan {b['real']:.1f} %, neto {b['neto']:+.3f} % · diferencia {r['top_menos_bot']:+.1f} puntos")
        if r.get('calibracion'):
            L.append('   calibración (predicho → real): ' + ' · '.join(f"{c['tramo']}: {c['predicho']:.0f} → {c['real']:.0f} % (n {c['n']:,}, neto {c['neto']:+.2f} %)" for c in r['calibracion']))
    block('MODELO COMPLETO · todas las filas', ev['modelo']['todo'])
    block(f"MODELO COMPLETO · dentro de la regla {ev['meta']['regla']['nombre']} ({' Y '.join(search.DESC[k] for k in ev['meta']['regla']['regla'])})", ev['modelo']['en_regla'])
    block('CONTROL SOLO VOLATILIDAD · todas las filas', ev['solo_volatilidad']['todo'])
    block(f"CONTROL SOLO VOLATILIDAD · dentro de la regla", ev['solo_volatilidad']['en_regla'])
    L.append('')
    L.append(f"{'año':>5s} {'n':>8s} {'AUC':>6s} {'ganan':>6s} {'decil+ gana':>11s} {'fiab':>6s} {'neto':>7s} {'exceso':>7s} {'decil- gana':>11s}")
    for Y, r in ev['modelo']['por_año'].items():
        if 'auc' not in r:
            continue
        t, b = r.get('top', {}), r.get('bot', {})
        L.append(f"{Y:>5} {r['n']:8,} {r['auc']:6.3f} {r['base']:6.1f} {t.get('real', float('nan')):11.1f} {t.get('fiab', float('nan')):+6.1f} {t.get('neto', float('nan')):+7.2f} {t.get('exceso', float('nan')):+7.2f} {b.get('real', float('nan')):11.1f}")
    if ev['modelo']['todo'].get('deciles'):
        L += ['', 'Deciles de probabilidad dentro de cada día (modelo completo, todas las filas):', f"{'decil':>5s} {'n':>8s} {'pred':>6s} {'real':>6s} {'fiab':>6s} {'t':>6s} {'neto':>7s} {'exceso':>7s} {'t':>6s}"]
        for q in ev['modelo']['todo']['deciles']:
            L.append(f"{q['decil']:5d} {q['n']:8,} {q['predicho']:6.1f} {q['real']:6.1f} {q['fiab']:+6.1f} {q['fiab_t']:+6.1f} {q['neto']:+7.2f} {q['exceso']:+7.2f} {q['exceso_t']:+6.1f}")
    return '\n'.join(L)


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, members: str = 'asof') -> Dict:
    from . import data, universe
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    join = None
    if members == 'asof':
        try:
            join = universe.sp500_join_dates()
        except Exception as exc:
            print(f'  no se pudo leer la tabla de miembros del S&P 500 ({type(exc).__name__}: {exc}): se usan los miembros actuales', flush=True)
    ev = analyze(bars, workers=workers, verbose=True, join=join)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'fiab.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'fiab_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
