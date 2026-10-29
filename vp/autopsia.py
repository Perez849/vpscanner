"""Autopsia de las operaciones perdedoras de la regla de impulso (parte 2): ¿por qué pierden, hay un patrón y se pueden mitigar?

Se parte de las filas de la regla base (impulso 12-1 alto, bajo su media de 200, volatilidad alta) con la escalera del operador, tal como las mide la cartera simulada. Tres partes:

  1. AUTOPSIA (describe, no contrasta): cómo terminan las perdedoras (stop inicial, hueco de apertura, rapidez), qué camino siguieron (cuánto subieron antes de caer), cuándo (año, mes, día de la
     semana, estado del mercado, días con muchas señales a la vez, cercanía de resultados estimada con volumen y huecos), y en qué estado entraron cada variable de contexto.
  2. PATRONES CON VALIDACIÓN: con tantas variables, algo siempre «explica» las pérdidas. Por eso cada patrón se juzga fuera de muestra: (a) los más fuertes con los años ≤ 2022 y su comportamiento
     en 2023–2026; (b) selección hacia delante año a año de filtros de entrada (cada año se elige el filtro con lo anterior, menos 95 días, y se aplica al año), frente a excluir el mismo número de
     filas al azar; (c) un árbol de decisión pequeño, también hacia delante. Solo usa variables conocidas al cerrar la señal.
  3. MITIGACIÓN: (a) los filtros hacia delante aplicados a la cartera simulada (plan de 5 y todas las señales) frente a la cartera sin filtrar y frente a filtros al azar; (b) variantes de salida
     fijadas de antemano (asegurar antes, stop más ajustado o más ancho).
Resultado: data/autopsia.json y data/autopsia_report.txt.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import cartera, ml, search, ta
from .ml import CTX
from .ta import shift

EXIT = 'esc'
RULE = ('bajo_200', 'impulso12_alto', 'vol_alta')
EVT_VOL = 2.0                  # día de «resultados»: volumen ≥ 2 veces su media de 20 y movimiento (hueco o rendimiento) ≥ EVT_MOVE
EVT_MOVE = 0.04
DEV_END = 2022
PURGE_DAYS = 95                # entre el último día de entrenamiento de un filtro y el año al que se aplica
BLOCK = 10                     # sesiones por bloque para los errores agrupados
MAX_EXCL = 0.25                # un filtro no puede excluir más del 25 % de las filas que quedan
MIN_EXCL = 100
T_MIN = 2.0
VARIANTS = {                   # variantes de salida fijadas de antemano (sobre la escalera: stop 2,5 ATR; +4 % plano; +10 % persigue a 2 ATR)
    'esc': dict(stop_atr=2.5, ladder=search.LADDER),
    'lock2': dict(stop_atr=2.5, ladder=((2.0, None), (10.0, 2.0))),
    'lock3': dict(stop_atr=2.5, ladder=((3.0, None), (10.0, 2.0))),
    'stop2': dict(stop_atr=2.0, ladder=search.LADDER),
    'stop3': dict(stop_atr=3.0, ladder=search.LADDER),
    'stop4': dict(stop_atr=4.0, ladder=search.LADDER),
}
VAR_TXT = {'esc': 'tu escalera: stop 2,5 ATR · +4 % plano · +10 % persigue a 2 ATR', 'lock2': 'asegurar antes: +2 % plano · +10 % persigue', 'lock3': 'asegurar antes: +3 % plano · +10 % persigue',
           'stop2': 'stop más ajustado: 2,0 ATR', 'stop3': 'stop más ancho: 3,0 ATR', 'stop4': 'stop más ancho: 4,0 ATR'}
EXTRA_FEATS = ['entry_gap', 'ses_event', 'nsig', 'breadth', 'mkt_r20', 'nbear', 'nbull', 'prev_loss', 'prev_ses', 'stop_pct']
FEATS = list(CTX) + EXTRA_FEATS
FEAT_TXT = {'r1': 'rendimiento de la sesión', 'r5': 'rendimiento a 5 sesiones', 'r10': 'rendimiento a 10 sesiones', 'r20': 'rendimiento a 20 sesiones', 'r60': 'rendimiento a 60 sesiones', 'r120': 'rendimiento a 120 sesiones',
            'r250': 'rendimiento a 250 sesiones', 'mom12_1': 'impulso 12-1 meses', 'rsi2': 'RSI(2)', 'rsi14': 'RSI(14)', 'dist20': 'distancia a la media de 20', 'dist50': 'distancia a la media de 50',
            'dist200': 'distancia a la media de 200', 'slope50': 'pendiente de la media de 50', 'pos52': 'posición en el rango de 52 semanas', 'dd252': 'caída desde el máximo de 252 sesiones',
            'atrp': 'volatilidad (ATR/precio)', 'volrel': 'volumen relativo (log)', 'dvol': 'dinero negociado (log)', 'rvol': 'volatilidad realizada / ATR', 'gap': 'hueco de apertura de la sesión de la señal',
            'rng': 'rango de la sesión en ATR', 'body': 'cuerpo de la vela en ATR', 'clpos': 'dónde cerró dentro del rango', 'entry_gap': 'hueco de la apertura de entrada (siguiente sesión)',
            'ses_event': 'sesiones desde el último día de «resultados» estimado', 'nsig': 'señales de la regla ese mismo día', 'breadth': 'amplitud del mercado (% sobre su media de 200)',
            'mkt_r20': 'rendimiento medio del mercado a 20 sesiones', 'nbear': 'patrones bajistas vigentes', 'nbull': 'patrones alcistas vigentes', 'prev_loss': 'la última operación cerrada del valor perdió',
            'prev_ses': 'sesiones desde que cerró la última operación del valor', 'stop_pct': 'distancia del stop inicial (% del precio)'}


# ── trabajo por valor ─────────────────────────────────────────────────────
def _work(args: Tuple[str, Dict[str, np.ndarray]]) -> Tuple[str, Dict]:
    """Por valor: lo de la cartera (resultado de cada salida del menú), el camino de la operación con la escalera, los eventos tipo «resultados» y las variantes de salida."""
    sym, b = args
    _, w = cartera._sym_work((sym, b))
    F = ta.prepare(b)
    o, h, l, c, v, atr = F['o'], F['h'], F['l'], F['c'], F['v'], F['atr']
    n = len(c)
    net, code, nb, ex = search.sim_all(o, h, l, c, atr, 1, 2.5, None, None, 60, search.COST, None, search.LADDER, with_bars=True, extras=True)
    idx = np.arange(n)
    with np.errstate(invalid='ignore', divide='ignore'):
        vr = v / shift(ta.sma(v, 20), 1)
        ret = c / shift(c, 1) - 1.0
        gap = o / shift(c, 1) - 1.0
    evt = (vr >= EVT_VOL) & ((np.abs(ret) >= EVT_MOVE) | (np.abs(gap) >= EVT_MOVE))      # día tipo «resultados»: volumen alto y movimiento grande
    cs = np.cumsum(evt)
    jj = np.minimum(idx + nb.astype(np.int64), n - 1)                          # sesión de salida
    last = np.maximum.accumulate(np.where(evt, idx, -1))
    x = dict(ex)
    x['nb'] = nb.astype(np.float64)
    x['ev_during'] = np.where(nb > 0, (cs[jj] - cs[idx]) > 0, False).astype(np.float64)       # hubo un día tipo «resultados» entre la entrada y la salida (se sabe después)
    x['evt_exit'] = np.where(nb > 0, evt[jj], False).astype(np.float64)                         # y la salida fue justo en uno de esos días
    x['ses_event'] = np.where(last >= 0, idx - last, np.nan)                                    # sesiones desde el último (se sabe al cerrar la señal)
    with np.errstate(invalid='ignore', divide='ignore'):
        x['exgap'] = np.where(nb > 0, (o[jj] / c[np.maximum(jj - 1, 0)] - 1.0) * 100.0, np.nan)
        x['exvol'] = np.where(nb > 0, vr[jj], np.nan)
        x['exret'] = np.where(nb > 0, ret[jj] * 100.0, np.nan)
        x['entry_gap'] = np.concatenate([(o[1:] / c[:-1] - 1.0) * 100.0, [np.nan]])
    var = {}
    for name, kw in VARIANTS.items():
        nt, _ = search.sim_all(o, h, l, c, atr, 1, kw['stop_atr'], None, None, 60, search.COST, None, kw['ladder'])
        var[name] = nt.astype(np.float32)
    w['x'], w['var'] = x, var
    return sym, w


# ── tabla de operaciones ──────────────────────────────────────────────────
def build_table(D: ml.Data, P: search.Panel, W: Dict[str, Dict], rule: Tuple[str, ...] = RULE):
    """Una fila por operación de la regla (con resultado conocido): resultado de la escalera, camino, variables de entrada y variables solo conocidas después (prefijo post_)."""
    import pandas as pd
    cond = search.conditions(D)
    m = np.ones(len(D.sym), bool)
    for k in rule:
        m &= cond[k]
    e = search.EXN.index(EXIT)
    net = P.net[:, 0, e].astype(np.float64)
    ix = np.flatnonzero(m & np.isfinite(net))
    nrow = len(ix)
    sym, bar = D.sym[ix], D.bar[ix]
    fields = ('nb', 'mfe_c', 'mfe_h', 'mae', 'gapx', 'ev_during', 'evt_exit', 'ses_event', 'exgap', 'exvol', 'exret', 'entry_gap')
    X = {f: np.full(nrow, np.nan) for f in fields}
    V = {name: np.full(nrow, np.nan) for name in VARIANTS}
    for j in np.unique(sym):
        sel = np.flatnonzero(sym == j)
        s_ = D.syms[int(j)]
        for f in fields:
            X[f][sel] = W[s_]['x'][f][bar[sel]]
        for name in VARIANTS:
            V[name][sel] = W[s_]['var'][name][bar[sel]]
    pn = {k: j for j, k in enumerate(D.pnames)}
    pat = lambda nm: D.pat[ix, pn[nm]].astype(np.float64)
    day = D.dcode[ix]
    nsig = np.bincount(D.dcode[m], minlength=D.nd)[day].astype(np.float64)
    cols = {'sym': np.array(D.syms)[sym], 'bar': bar.astype(np.int64), 'dcode': day, 'year': D.year[ix], 'date': D.date[day].astype(str), 'net': net[ix], 'code': P.code[ix, 0, e].astype(np.int64),
            'nsig': nsig, 'breadth': D.breadth_d[day] * 100.0, 'mkt_r20': D.mkt_r20_d[day] * 100.0, 'nbear': pat('n_velas_dn') + pat('n_fig_dn') + pat('n_rup_dn'),
            'nbull': pat('n_velas_up') + pat('n_fig_up') + pat('n_rup_up')}
    for k_, name in enumerate(CTX):
        cols[name] = D.ctx[ix, k_].astype(np.float64)
    cols['stop_pct'] = search_stop_pct(cols['atrp'])
    for f in ('entry_gap', 'ses_event'):
        cols[f] = X[f]
    for f in ('nb', 'mfe_c', 'mfe_h', 'mae', 'gapx', 'ev_during', 'evt_exit', 'exgap', 'exvol', 'exret'):
        cols['post_' + f] = X[f]
    for name in VARIANTS:
        cols['net_' + name] = V[name]
    T = pd.DataFrame(cols)
    T['loss'] = (T['net'] <= 0).astype(np.int8)
    # la última operación ya cerrada del mismo valor al entrar (¿venía de perder?)
    prev_loss = np.full(nrow, np.nan)
    prev_ses = np.full(nrow, np.nan)
    for s_, g in T.groupby('sym'):
        i_ = g.index.to_numpy()
        b_, e_ = g['bar'].to_numpy(), g['bar'].to_numpy() + g['post_nb'].to_numpy()
        o_ = np.argsort(e_, kind='stable')
        es, ns = e_[o_], g['net'].to_numpy()[o_]
        pos = np.searchsorted(es, b_, side='left') - 1                             # operaciones cerradas estrictamente antes de entrar
        ok_ = pos >= 0
        prev_loss[i_[ok_]] = (ns[pos[ok_]] <= 0).astype(float)
        prev_ses[i_[ok_]] = b_[ok_] - es[pos[ok_]]
    T['prev_loss'], T['prev_ses'] = prev_loss, prev_ses
    return T


def search_stop_pct(atrp: np.ndarray) -> np.ndarray:
    return 2.5 * atrp


# ── estadística ───────────────────────────────────────────────────────────
def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Área bajo la curva ROC de `score` para `y` (1 = pierde); 0,5 = sin información. NaN en score se ignora."""
    import pandas as pd
    ok = np.isfinite(score)
    s, yy = score[ok], y[ok]
    n1, n0 = int(yy.sum()), int((1 - yy).sum())
    if n1 < 20 or n0 < 20:
        return float('nan')
    r = pd.Series(s).rank().to_numpy()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def t_diff(net: np.ndarray, excl: np.ndarray, blk: np.ndarray) -> Tuple[float, float]:
    """Media de los que se quedan menos media de los excluidos, y su t con errores agrupados en bloques."""
    k = ~excl
    nk, ne = int(k.sum()), int(excl.sum())
    if nk < 30 or ne < 30:
        return float('nan'), float('nan')
    mk, me = net[k].mean(), net[excl].mean()
    infl = np.where(k, (net - mk) / nk, -(net - me) / ne)
    s = np.bincount(blk, weights=infl, minlength=int(blk.max()) + 1)
    se = float(np.sqrt((s ** 2).sum()))
    return float(mk - me), float((mk - me) / se) if se > 0 else float('nan')


def t_mean(x: np.ndarray, blk: np.ndarray) -> Tuple[float, float]:
    n = len(x)
    if n < 30:
        return float('nan'), float('nan')
    m = x.mean()
    s = np.bincount(blk, weights=x - m, minlength=int(blk.max()) + 1)
    se = float(np.sqrt((s ** 2).sum())) / n
    return float(m), float(m / se) if se > 0 else float('nan')


def _r(x, d=2):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


# ── 1. autopsia ───────────────────────────────────────────────────────────
def autopsy(T) -> Dict:
    net = T['net'].to_numpy()
    loss = T['loss'].to_numpy().astype(bool)
    win = ~loss
    n = len(T)
    L, Wn = T[loss], T[win]
    tot_loss = float(-L['net'].sum())
    out: Dict = {'n': int(n), 'ganan': _r(100 * win.mean(), 1), 'neto_medio': _r(net.mean(), 3), 'gan_media': _r(Wn['net'].mean()), 'perd_media': _r(L['net'].mean()), 'perdidas': int(loss.sum())}
    names = {0: 'salta el stop inicial', 1: 'aseguró +4 % y salió con pérdida (hueco)', 2: 'aseguró +10 % y salió con pérdida (hueco)', 3: 'cierre por tiempo sin asegurar'}
    out['por_tipo'] = [{'tipo': names.get(int(c_), str(c_)), 'n': int(len(g)), 'pct_perdedoras': _r(100 * len(g) / max(1, len(L)), 1), 'perdida_media': _r(g['net'].mean()), 'pct_del_total_perdido': _r(100 * -g['net'].sum() / tot_loss, 1)}
                       for c_, g in L.groupby('code')]
    st = L[L['code'] == 0]
    gx = st['post_gapx'] > 0
    out['stop_inicial'] = {'n': int(len(st)), 'por_hueco': _r(100 * gx.mean(), 1), 'perdida_media_hueco': _r(st.loc[gx, 'net'].mean()), 'perdida_media_intradia': _r(st.loc[~gx, 'net'].mean()),
                           'en_dia_resultados': _r(100 * (st['post_evt_exit'] > 0).mean(), 1), 'con_resultados_en_la_operacion': _r(100 * (st['post_ev_during'] > 0).mean(), 1),
                           'ganadoras_con_resultados_en_la_operacion': _r(100 * (Wn['post_ev_during'] > 0).mean(), 1), 'distancia_stop_media_pct': _r(st['stop_pct'].mean(), 1)}
    bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 60)]
    out['rapidez'] = [{'sesiones': f'{a}-{b_}', 'pct_stops': _r(100 * ((st['post_nb'] >= a) & (st['post_nb'] <= b_)).mean(), 1)} for a, b_ in bins]
    out['camino'] = {'stop_mfe_cierre_mediana': _r(st['post_mfe_c'].median()), 'stop_mfe_max_mediana': _r(st['post_mfe_h'].median()),
                     'pct_toco_2': _r(100 * (st['post_mfe_h'] >= 2).mean(), 1), 'pct_toco_4': _r(100 * (st['post_mfe_h'] >= 4).mean(), 1), 'pct_toco_8': _r(100 * (st['post_mfe_h'] >= 8).mean(), 1),
                     'pct_cerro_2': _r(100 * (st['post_mfe_c'] >= 2).mean(), 1), 'pct_nunca_arriba': _r(100 * (st['post_mfe_h'] < 1).mean(), 1),
                     'ganadoras_caida_maxima_mediana': _r(Wn['post_mae'].median()), 'ganadoras_con_caida_mayor_5': _r(100 * (Wn['post_mae'] <= -5).mean(), 1), 'ganadoras_con_caida_mayor_10': _r(100 * (Wn['post_mae'] <= -10).mean(), 1)}
    blk = (T['dcode'].to_numpy() // BLOCK).astype(np.int64)

    def grp(col, edges, labels):
        rows = []
        v = T[col].to_numpy()
        for lab, (lo, hi) in zip(labels, zip(edges[:-1], edges[1:])):
            m = np.isfinite(v) & (v >= lo) & (v < hi)
            if m.sum() >= 30:
                mm, tt = t_mean(net[m], blk[m])
                rows.append({'grupo': lab, 'n': int(m.sum()), 'pierden': _r(100 * loss[m].mean(), 1), 'neto_medio': _r(mm, 2), 't': _r(tt, 1), 'perd_media': _r(net[m & loss].mean())})
        return rows
    out['por_año'] = [{'grupo': str(int(y)), 'n': int(len(g)), 'pierden': _r(100 * g['loss'].mean(), 1), 'neto_medio': _r(g['net'].mean(), 2)} for y, g in T.groupby('year')]
    mes = T['date'].str.slice(5, 7).astype(int).to_numpy()
    T2 = T.assign(mes=mes)
    out['por_mes'] = [{'grupo': str(int(m_)), 'n': int(len(g)), 'pierden': _r(100 * g['loss'].mean(), 1), 'neto_medio': _r(g['net'].mean(), 2)} for m_, g in T2.groupby('mes')]
    import pandas as pd
    dow = pd.to_datetime(T['date']).dt.dayofweek.to_numpy()
    T2['dow'] = dow
    out['por_dia_semana'] = [{'grupo': ['lun', 'mar', 'mié', 'jue', 'vie', 'sáb', 'dom'][int(d_)], 'n': int(len(g)), 'pierden': _r(100 * g['loss'].mean(), 1), 'neto_medio': _r(g['net'].mean(), 2)} for d_, g in T2.groupby('dow')]
    out['por_amplitud'] = grp('breadth', [-1, 30, 45, 60, 75, 101], ['< 30 %', '30-45 %', '45-60 %', '60-75 %', '> 75 %'])
    out['por_señales_del_dia'] = grp('nsig', [0, 2, 4, 8, 16, 10 ** 6], ['1', '2-3', '4-7', '8-15', '16 o más'])
    out['por_mercado_20'] = grp('mkt_r20', [-100, -5, -2, 0, 2, 5, 100], ['< −5 %', '−5 a −2 %', '−2 a 0 %', '0 a 2 %', '2 a 5 %', '> 5 %'])
    out['por_hueco_entrada'] = grp('entry_gap', [-100, -3, -1, 0, 1, 3, 100], ['< −3 %', '−3 a −1 %', '−1 a 0 %', '0 a 1 %', '1 a 3 %', '> 3 %'])
    out['por_resultados_est'] = grp('ses_event', [0, 5, 20, 40, 55, 70, 10 ** 6], ['0-4', '5-19', '20-39', '40-54', '55-69', '70 o más'])
    out['por_volatilidad'] = grp('atrp', [0, 4, 5, 6, 8, 100], ['< 4 %', '4-5 %', '5-6 %', '6-8 %', '> 8 %'])
    pl = T['prev_loss'].to_numpy()
    se = T['ses_event'].to_numpy()
    nxt = np.where(np.isfinite(se), 63.0 - (se % 63.0), np.nan)                 # sesiones estimadas hasta los próximos resultados (si vuelven cada 63)
    out['por_resultados_proximos'] = grp_arr(v_=nxt, edges=[0, 7, 14, 26, 46, 64], labels=['1-6', '7-13', '14-25', '26-45', '46-63'], net=net, loss=loss, blk=blk)
    est = np.isfinite(nxt) & (nxt <= 13)
    evx = T['post_ev_during'].to_numpy() > 0
    out['proxy_resultados'] = {'con_proxy': {'n': int(est.sum()), 'con_resultados_en_la_operacion': _r(100 * evx[est].mean(), 1), 'pierden': _r(100 * loss[est].mean(), 1), 'neto_medio': _r(net[est].mean(), 2)},
                               'sin_proxy': {'n': int((~est & np.isfinite(nxt)).sum()), 'con_resultados_en_la_operacion': _r(100 * evx[~est & np.isfinite(nxt)].mean(), 1), 'pierden': _r(100 * loss[~est & np.isfinite(nxt)].mean(), 1),
                                             'neto_medio': _r(net[~est & np.isfinite(nxt)].mean(), 2)}}
    out['regimen_por_año'] = regimen_por_año(T)
    out['tras_perdedora_del_valor'] = [{'grupo': t_, 'n': int(m.sum()), 'pierden': _r(100 * loss[m].mean(), 1), 'neto_medio': _r(net[m].mean(), 2)} for t_, m in
                                       (('la anterior ganó', pl == 0), ('la anterior perdió', pl == 1), ('sin anterior cerrada', ~np.isfinite(pl))) if m.sum() >= 30]
    ev = T['post_ev_during'].to_numpy() > 0
    out['con_resultados_en_la_operacion'] = [{'grupo': t_, 'n': int(m.sum()), 'pierden': _r(100 * loss[m].mean(), 1), 'neto_medio': _r(net[m].mean(), 2), 'perd_media': _r(net[m & loss].mean())} for t_, m in
                                             (('sin día de resultados entre entrada y salida', ~ev), ('con día de resultados entre entrada y salida', ev)) if m.sum() >= 30]
    return out


def grp_arr(v_: np.ndarray, edges, labels, net, loss, blk) -> List[Dict]:
    rows = []
    for lab, (lo, hi) in zip(labels, zip(edges[:-1], edges[1:])):
        m = np.isfinite(v_) & (v_ >= lo) & (v_ < hi)
        if m.sum() >= 30:
            mm, tt = t_mean(net[m], blk[m])
            rows.append({'grupo': lab, 'n': int(m.sum()), 'pierden': _r(100 * loss[m].mean(), 1), 'neto_medio': _r(mm, 2), 't': _r(tt, 1), 'perd_media': _r(net[m & loss].mean())})
    return rows


REGIMENES = (('amplitud_30', 'amplitud del mercado < 30 %', lambda T: T['breadth'].to_numpy() < 30), ('amplitud_45', 'amplitud del mercado < 45 %', lambda T: T['breadth'].to_numpy() < 45),
             ('mercado_m5', 'mercado a 20 sesiones < −5 %', lambda T: T['mkt_r20'].to_numpy() < -5), ('señales_16', '16 o más señales el mismo día', lambda T: T['nsig'].to_numpy() >= 16),
             ('pánico', 'cualquiera de las tres anteriores (amplitud < 30 %, mercado < −5 %, 16+ señales)', lambda T: (T['breadth'].to_numpy() < 30) | (T['mkt_r20'].to_numpy() < -5) | (T['nsig'].to_numpy() >= 16)))


def regimen_por_año(T) -> List[Dict]:
    """El régimen de «pánico» año a año: cuántas operaciones, cuántos episodios distintos (rachas de días separadas por más de 10 sesiones) y cómo les fue; un patrón que vive de uno o dos episodios no es un patrón."""
    net = T['net'].to_numpy()
    loss = T['loss'].to_numpy().astype(bool)
    year = T['year'].to_numpy()
    dc = T['dcode'].to_numpy()
    out = []
    for key, txt, f in REGIMENES:
        m = f(T)
        days = np.unique(dc[m])
        epis = int(1 + (np.diff(days) > 10).sum()) if len(days) else 0
        rows = []
        for y in sorted(np.unique(year)):
            a = m & (year == y)
            b_ = ~m & (year == y)
            rows.append({'año': int(y), 'n': int(a.sum()), 'pierden': _r(100 * loss[a].mean(), 1) if a.sum() >= 15 else None, 'neto': _r(net[a].mean(), 2) if a.sum() >= 15 else None,
                         'n_resto': int(b_.sum()), 'pierden_resto': _r(100 * loss[b_].mean(), 1) if b_.sum() >= 15 else None, 'neto_resto': _r(net[b_].mean(), 2) if b_.sum() >= 15 else None})
        out.append({'regimen': key, 'txt': txt, 'n': int(m.sum()), 'episodios': epis, 'dias': int(len(days)), 'pierden': _r(100 * loss[m].mean(), 1), 'neto': _r(net[m].mean(), 2),
                    'pierden_resto': _r(100 * loss[~m].mean(), 1), 'neto_resto': _r(net[~m].mean(), 2), 'por_año': rows})
    return out


# ── 2. patrones con validación ────────────────────────────────────────────
def univariate(T, split: int) -> Dict:
    """Por variable de entrada: AUC de «pierde» en desarrollo (año < split) y en la prueba (≥ split), y la tabla por quintiles. Un patrón real tiene el mismo signo en las dos."""
    loss = T['loss'].to_numpy()
    dev = (T['year'] < split).to_numpy()
    rows = []
    for f in FEATS:
        x = T[f].to_numpy().astype(np.float64)
        a_d, a_t = auc(x[dev], loss[dev]), auc(x[~dev], loss[~dev])
        rows.append({'var': f, 'txt': FEAT_TXT.get(f, f), 'auc_dev': _r(a_d, 3), 'auc_test': _r(a_t, 3), 'misma_direccion': bool(np.isfinite(a_d) and np.isfinite(a_t) and (a_d - 0.5) * (a_t - 0.5) > 0)})
    rows.sort(key=lambda r: -abs((r['auc_dev'] or 0.5) - 0.5))
    top = []
    for r in rows[:12]:
        x = T[r['var']].to_numpy().astype(np.float64)
        ok = np.isfinite(x)
        if ok.sum() < 200 or len(np.unique(x[ok])) < 3:
            continue
        qs = np.unique(np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8]))
        b_ = np.digitize(x, qs)
        tabla = [{'q': int(q_) + 1, 'n': int(((b_ == q_) & ok).sum()), 'pierden': _r(100 * loss[(b_ == q_) & ok].mean(), 1), 'neto_medio': _r(T['net'].to_numpy()[(b_ == q_) & ok].mean(), 2),
                  'desde': _r(x[(b_ == q_) & ok].min(), 3), 'hasta': _r(x[(b_ == q_) & ok].max(), 3)} for q_ in range(len(qs) + 1) if ((b_ == q_) & ok).sum() >= 30]
        top.append({**r, 'quintiles': tabla})
    return {'split': split, 'ranking': rows, 'top': top}


def _cands(T, train: np.ndarray) -> List[Tuple[str, str, float]]:
    """Filtros candidatos con el corte sacado solo del entrenamiento: cola baja o alta (10 y 20 %) de cada variable, y «la anterior perdió»."""
    out = []
    for f in FEATS:
        x = T[f].to_numpy().astype(np.float64)[train]
        x = x[np.isfinite(x)]
        if len(x) < 300 or len(np.unique(x)) < 3:
            continue
        if f == 'prev_loss':
            out.append((f, 'eq', 1.0))
            continue
        for q in (0.10, 0.20):
            out.append((f, 'lo', float(np.quantile(x, q))))
            out.append((f, 'hi', float(np.quantile(x, 1 - q))))
    return out


def _mask(T, f: str, side: str, cut: float) -> np.ndarray:
    x = T[f].to_numpy().astype(np.float64)
    with np.errstate(invalid='ignore'):
        return (x <= cut) if side == 'lo' else (x >= cut) if side == 'hi' else (x == cut)


def _ftxt(f: str, side: str, cut: float) -> str:
    base = FEAT_TXT.get(f, f)
    return f'{base} (la anterior perdió)' if side == 'eq' else f'{base} {"≤" if side == "lo" else "≥"} {cut:.3g}'


def greedy_filters(T, train: np.ndarray, depth: int = 3) -> List[Tuple[str, str, float, float, int]]:
    """Hasta `depth` filtros en cadena: cada paso elige, entre las filas que quedan del entrenamiento, el de mayor t de «los que se quedan menos los excluidos» (con errores agrupados), si t ≥ T_MIN,
    excluye entre MIN_EXCL filas y MAX_EXCL de las que quedan y los excluidos rinden menos que los que se quedan."""
    net = T['net'].to_numpy()
    blk = (T['dcode'].to_numpy() // BLOCK).astype(np.int64)
    cands = _cands(T, train)
    keep = train.copy()
    chosen = []
    for _ in range(depth):
        best = None
        for f, side, cut in cands:
            ex = _mask(T, f, side, cut) & keep
            ne = int(ex.sum())
            if ne < MIN_EXCL or ne > MAX_EXCL * keep.sum():
                continue
            sub = keep
            d, t = t_diff(net[sub], ex[sub], blk[sub])
            if np.isfinite(t) and d > 0 and t >= T_MIN and (best is None or t > best[0]):
                best = (t, f, side, cut, ne, d)
        if best is None:
            break
        _, f, side, cut, ne, d = best
        chosen.append((f, side, cut, best[0], ne))
        keep = keep & ~_mask(T, f, side, cut)
    return chosen


def filters_oos(T, test_years: List[int], reps: int = 300, seed: int = 5, depth: int = 3) -> Dict:
    """Selección hacia delante: cada año de prueba se eligen los filtros con las filas anteriores (menos PURGE_DAYS) y se aplican a ese año. Se compara con excluir el mismo número de filas al azar."""
    import pandas as pd
    net = T['net'].to_numpy()
    loss = T['loss'].to_numpy().astype(bool)
    year = T['year'].to_numpy()
    dates = pd.to_datetime(T['date']).to_numpy()
    blk = (T['dcode'].to_numpy() // BLOCK).astype(np.int64)
    excl = np.zeros(len(T), bool)
    per_year = []
    for Y in test_years:
        cut = np.datetime64(f'{Y}-01-01') - np.timedelta64(PURGE_DAYS, 'D')
        train = dates < cut
        te = year == Y
        if train.sum() < 1500 or te.sum() < 100:
            continue
        ch = greedy_filters(T, train, depth)
        ex = np.zeros(len(T), bool)
        for f, side, c_, _, _ in ch:
            ex |= _mask(T, f, side, c_)
        ex &= te
        excl |= ex
        per_year.append({'año': int(Y), 'n': int(te.sum()), 'excluidas': int(ex.sum()), 'neto_todas': _r(net[te].mean()), 'neto_quedan': _r(net[te & ~ex].mean()) if (te & ~ex).any() else None,
                         'neto_excluidas': _r(net[ex].mean()) if ex.any() else None, 'filtros': [{'txt': _ftxt(f, s_, c_), 't_train': _r(t_, 1), 'n_train': int(n_)} for f, s_, c_, t_, n_ in ch]})
    ev = np.isin(year, [p['año'] for p in per_year])
    out: Dict = {'por_año': per_year}
    if ev.sum() < 300 or not excl.any():
        out['resumen'] = None
        return out
    d, t = t_diff(net[ev], excl[ev], blk[ev])
    rng = np.random.default_rng(seed)
    obs = net[ev & ~excl].mean()
    sims = np.zeros(reps)
    idxs = [(np.flatnonzero(year == Y), int((excl & (year == Y)).sum())) for Y in [p['año'] for p in per_year]]
    for r_ in range(reps):
        exr = np.zeros(len(T), bool)
        for ii, k_ in idxs:
            exr[rng.choice(ii, k_, replace=False)] = True
        sims[r_] = net[ev & ~exr].mean()
    out['resumen'] = {'n': int(ev.sum()), 'excluidas': int(excl.sum()), 'pct_excluidas': _r(100 * excl.sum() / ev.sum(), 1), 'neto_todas': _r(net[ev].mean(), 3), 'neto_quedan': _r(obs, 3), 'neto_excluidas': _r(net[excl].mean(), 3),
                      'ganan_todas': _r(100 * (~loss[ev]).mean(), 1), 'ganan_quedan': _r(100 * (~loss[ev & ~excl]).mean(), 1), 'perdedoras_evitadas_pct': _r(100 * (excl & loss).sum() / loss[ev].sum(), 1),
                      'ganadoras_perdidas_pct': _r(100 * (excl & ~loss).sum() / (~loss[ev]).sum(), 1), 'perd_media_quedan': _r(net[ev & ~excl & loss].mean(), 2), 'diferencia': _r(d, 3), 't': _r(t, 2),
                      'azar_neto_quedan_mediana': _r(np.median(sims), 3), 'azar_p95': _r(np.quantile(sims, 0.95), 3), 'p_azar': _r(float((sims >= obs).mean()), 3), 'reps': reps}
    out['excl'] = excl
    return out


def tree_oos(T, split: int, test_years: List[int], depth: int = 3, min_leaf: int = 250) -> Dict:
    """Un árbol de decisión pequeño sobre el resultado neto: (a) ajustado con los años < split, sus hojas y cómo rinden después; (b) hacia delante año a año: se excluyen las hojas cuyo rendimiento
    en el entrenamiento es menos de la mitad del medio, y se mide el efecto fuera de muestra."""
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor, export_text
    X = T[FEATS].to_numpy().astype(np.float64)
    net = T['net'].to_numpy()
    loss = T['loss'].to_numpy().astype(bool)
    year = T['year'].to_numpy()
    dates = pd.to_datetime(T['date']).to_numpy()
    blk = (T['dcode'].to_numpy() // BLOCK).astype(np.int64)

    def fit(train):
        med = np.nanmedian(X[train], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isfinite(X), X, med[None, :])
        m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=0).fit(Xf[train], net[train])
        return m, Xf
    out: Dict = {}
    dev = year < split
    if dev.sum() >= 1500 and (~dev).sum() >= 300:
        m, Xf = fit(dev)
        leaf = m.apply(Xf)
        rows = []
        for l_ in np.unique(leaf):
            a, b_ = leaf == l_, None
            rows.append({'hoja': int(l_), 'n_dev': int((a & dev).sum()), 'neto_dev': _r(net[a & dev].mean(), 2), 'pierden_dev': _r(100 * loss[a & dev].mean(), 1), 'n_test': int((a & ~dev).sum()),
                         'neto_test': _r(net[a & ~dev].mean(), 2) if (a & ~dev).sum() >= 20 else None, 'pierden_test': _r(100 * loss[a & ~dev].mean(), 1) if (a & ~dev).sum() >= 20 else None})
        out['reglas'] = export_text(m, feature_names=FEATS, max_depth=depth)
        out['hojas'] = sorted(rows, key=lambda r: r['neto_dev'])
    excl = np.zeros(len(T), bool)
    used = []
    for Y in test_years:
        cut = np.datetime64(f'{Y}-01-01') - np.timedelta64(PURGE_DAYS, 'D')
        train, te = dates < cut, year == Y
        if train.sum() < 1500 or te.sum() < 100:
            continue
        m, Xf = fit(train)
        leaf = m.apply(Xf)
        base = net[train].mean()
        bad = [l_ for l_ in np.unique(leaf[train]) if net[train & (leaf == l_)].mean() < 0.5 * base and (train & (leaf == l_)).sum() >= min_leaf]
        ex = np.isin(leaf, bad) & te
        excl |= ex
        used.append(int(Y))
    ev = np.isin(year, used)
    if ev.sum() >= 300 and excl.any():
        d, t = t_diff(net[ev], excl[ev], blk[ev])
        out['hacia_delante'] = {'n': int(ev.sum()), 'excluidas': int(excl.sum()), 'pct_excluidas': _r(100 * excl.sum() / ev.sum(), 1), 'neto_todas': _r(net[ev].mean(), 3), 'neto_quedan': _r(net[ev & ~excl].mean(), 3),
                                'neto_excluidas': _r(net[excl].mean(), 3), 'perdedoras_evitadas_pct': _r(100 * (excl & loss).sum() / loss[ev].sum(), 1), 'ganadoras_perdidas_pct': _r(100 * (excl & ~loss).sum() / (~loss[ev]).sum(), 1),
                                'diferencia': _r(d, 3), 't': _r(t, 2)}
    out['excl'] = excl
    return out


def dev_filters(T, split: int, top: int = 12) -> Dict:
    """Los patrones más fuertes en desarrollo (año < split), uno a uno, y qué pasó con ellos después: es el contraste honesto de «el patrón que explica las pérdidas»."""
    net = T['net'].to_numpy()
    loss = T['loss'].to_numpy().astype(bool)
    year = T['year'].to_numpy()
    blk = (T['dcode'].to_numpy() // BLOCK).astype(np.int64)
    dev = year < split
    rows = []
    for f, side, cut in _cands(T, dev):
        ex = _mask(T, f, side, cut)
        nd, nt = int((ex & dev).sum()), int((ex & ~dev).sum())
        if nd < MIN_EXCL or nd > MAX_EXCL * dev.sum() or nt < 30:
            continue
        d, t = t_diff(net[dev], (ex & dev)[dev], blk[dev])
        if not np.isfinite(t) or d <= 0:
            continue
        dt, tt = t_diff(net[~dev], (ex & ~dev)[~dev], blk[~dev])
        rows.append({'txt': _ftxt(f, side, cut), 'n_dev': nd, 'neto_excl_dev': _r(net[ex & dev].mean(), 2), 'pierden_excl_dev': _r(100 * loss[ex & dev].mean(), 1), 'neto_resto_dev': _r(net[~ex & dev].mean(), 2), 't_dev': _r(t, 1),
                     'n_test': nt, 'neto_excl_test': _r(net[ex & ~dev].mean(), 2), 'pierden_excl_test': _r(100 * loss[ex & ~dev].mean(), 1), 'neto_resto_test': _r(net[~ex & ~dev].mean(), 2), 't_test': _r(tt, 1),
                     'se_mantiene': bool(np.isfinite(tt) and dt > 0 and tt >= 1.0)})
    rows.sort(key=lambda r: -r['t_dev'])
    return {'split': split, 'candidatos': len(rows), 'top': rows[:top], 'se_mantienen': int(sum(r['se_mantiene'] for r in rows[:top]))}


# ── 3. mitigación ─────────────────────────────────────────────────────────
def exit_variants(T, split: int) -> Dict:
    """Las variantes de salida fijadas de antemano sobre las mismas filas: neto, % que ganan, ganancia y pérdida medias, y t de la diferencia con la escalera (por operación)."""
    blk = (T['dcode'].to_numpy() // BLOCK).astype(np.int64)
    dev = (T['year'] < split).to_numpy()
    base = T['net_esc'].to_numpy()
    rows = []
    for name in VARIANTS:
        x = T['net_' + name].to_numpy()
        ok = np.isfinite(x) & np.isfinite(base)
        dm, dt = t_mean((x - base)[ok], blk[ok])
        rows.append({'salida': name, 'txt': VAR_TXT[name], 'n': int(ok.sum()), 'neto': _r(x[ok].mean(), 3), 'ganan': _r(100 * (x[ok] > 0).mean(), 1), 'gan_media': _r(x[ok][x[ok] > 0].mean(), 2), 'perd_media': _r(x[ok][x[ok] <= 0].mean(), 2),
                     'dif_con_escalera': _r(dm, 3), 't_dif': _r(dt, 1), 'neto_dev': _r(x[ok & dev].mean(), 3), 'neto_test': _r(x[ok & ~dev].mean(), 3), 'dif_dev': _r((x - base)[ok & dev].mean(), 3), 'dif_test': _r((x - base)[ok & ~dev].mean(), 3)})
    return {'split': split, 'variantes': rows}


def symbols(T, split: int) -> Dict:
    """¿Hay valores que pierden siempre? Los que más operaciones y pérdidas acumulan, y si la tasa de pérdida de un valor en desarrollo predice la de la prueba."""
    loss = T['loss'].to_numpy()
    dev = (T['year'] < split).to_numpy()
    g = T.groupby('sym').agg(n=('net', 'size'), pierden=('loss', 'mean'), neto=('net', 'mean'), perdida_total=('net', lambda s: float(s[s <= 0].sum()))).reset_index()
    top = g.sort_values('perdida_total').head(12)
    gd = T[dev].groupby('sym')['loss'].agg(['mean', 'size'])
    gt = T[~dev].groupby('sym')['loss'].agg(['mean', 'size'])
    j = gd.join(gt, lsuffix='_d', rsuffix='_t', how='inner')
    j = j[(j['size_d'] >= 8) & (j['size_t'] >= 8)]
    corr = float(np.corrcoef(j['mean_d'], j['mean_t'])[0, 1]) if len(j) >= 20 else None
    tot = float(-T['net'][T['loss'] == 1].sum())
    return {'top_perdedores': [{'sym': r.sym, 'n': int(r.n), 'pierden': _r(100 * r.pierden, 1), 'neto_medio': _r(r.neto, 2), 'pct_del_total_perdido': _r(100 * -r.perdida_total / tot, 1)} for r in top.itertuples()],
            'valores_con_historial': int(len(g)), 'corr_tasa_dev_test': _r(corr, 2) if corr is not None else None, 'valores_comparables': int(len(j))}


def portfolio(D, Wd, T, excl_greedy: np.ndarray, excl_tree: np.ndarray, test_years: List[int], reps: int = 20, seed: int = 9) -> Dict:
    """Los filtros hacia delante aplicados a la cartera simulada (plan de 5 y todas las señales), solo en los años de prueba, frente a la cartera sin filtrar y a filtros al azar con el mismo número de filas.
    Orden de entrada: por volatilidad (el modelo de fiabilidad no entra aquí, para que sea comparable año a año)."""
    big = 10 ** 6
    rng = np.random.default_rng(seed)
    mask = Wd.rule_mask(list(RULE[:2]))
    years_d = np.zeros(Wd.nd, int)
    years_d[D.dcode] = D.year
    d0 = int(np.flatnonzero(years_d == min(test_years))[0])
    rowid = {}                                                                  # (símbolo, barra) → fila del panel
    for r_, (s_, b_) in enumerate(zip(D.sym, D.bar)):
        rowid[(int(s_), int(b_))] = r_
    symidx = {s_: j for j, s_ in enumerate(D.syms)}
    prow = np.array([rowid[(symidx[s_], int(b_))] for s_, b_ in zip(T['sym'], T['bar'])])

    def keep(excl: np.ndarray) -> np.ndarray:
        k = np.ones(len(D.sym), bool)
        k[prow[excl]] = False
        return k
    out: Dict = {'desde': str(D.date[int(np.flatnonzero(D.dcode == d0)[0])]) if (D.dcode == d0).any() else None}
    cases = (('plan', {}), ('todas', {'max_pos': big, 'corr_min': 2.0}))
    sims = {}
    for nm, kw in cases:
        base = cartera.simulate(Wd, mask, EXIT, 'vol', rng, d0=d0, **kw)
        yrs = years_d[base['d0']:base['dend'] + 1]
        sims[nm] = {'sin_filtro': cartera.metrics(base['eq'], yrs, base['trades'], base['expo'])}
        for fnm, ex in (('filtros', excl_greedy), ('arbol', excl_tree)):
            if not ex.any():
                continue
            kp = keep(ex)
            s2 = cartera.simulate(Wd, mask & kp, EXIT, 'vol', rng, d0=d0, **kw)
            sims[nm][fnm] = cartera.metrics(s2['eq'], yrs, s2['trades'], s2['expo'])
        if excl_greedy.any():                                                   # control: excluir al azar el mismo número de filas por año de prueba
            yr_rows = {Y: np.flatnonzero((T['year'].to_numpy() == Y)) for Y in test_years}
            per = {Y: int((excl_greedy & (T['year'].to_numpy() == Y)).sum()) for Y in test_years}
            an = []
            for _ in range(reps):
                exr = np.zeros(len(T), bool)
                for Y, ii in yr_rows.items():
                    if per[Y] and len(ii) > per[Y]:
                        exr[rng.choice(ii, per[Y], replace=False)] = True
                s3 = cartera.simulate(Wd, mask & keep(exr), EXIT, 'vol', rng, d0=d0, **kw)
                an.append(cartera.metrics(s3['eq'], yrs, s3['trades'], s3['expo']))
            obs = sims[nm]['filtros']['anual']
            sims[nm]['azar'] = {'anual': {'mediana': _r(np.median([a['anual'] for a in an]), 2), 'p05': _r(np.quantile([a['anual'] for a in an], 0.05), 2), 'p95': _r(np.quantile([a['anual'] for a in an], 0.95), 2),
                                          'p': _r(float(np.mean([a['anual'] >= obs for a in an])), 3), 'n': reps},
                                'caida_max': {'mediana': _r(np.median([a['caida_max'] for a in an]), 2)}}
    out['carteras'] = sims
    # (a) stops iniciales más anchos con la escalera: mismo tamaño de posición (el de 2,5 ATR) y mismo riesgo (posición más pequeña cuanto más ancho el stop)
    ancho = {}
    for nm, kw in cases:
        ancho[nm] = {}
        for ex_, atr_ in (('esc', 2.5), ('esc3', 3.0), ('esc4', 4.0), ('esc5', 5.0)):
            s5 = cartera.simulate(Wd, mask, ex_, 'vol', rng, d0=d0, **kw)
            ancho[nm][ex_] = {'mismo_tamaño': cartera.metrics(s5['eq'], yrs, s5['trades'], s5['expo'])}
            if atr_ != 2.5:
                s6 = cartera.simulate(Wd, mask, ex_, 'vol', rng, d0=d0, risk=cartera.RISK_PCT * 2.5 / atr_, **kw)
                ancho[nm][ex_]['mismo_riesgo'] = cartera.metrics(s6['eq'], yrs, s6['trades'], s6['expo'])
    out['stops_anchos'] = ancho
    # (b) régimen de pánico: operar solo en él (o fuera de él), en la ventana de prueba; mirado DESPUÉS de ver la tabla, así que no es evidencia, solo cuánto cuenta en una cuenta
    bre = D.breadth_d[D.dcode] * 100.0
    mkt = D.mkt_r20_d[D.dcode] * 100.0
    cond = search.conditions(D)
    rm = np.ones(len(D.sym), bool)
    for k_ in RULE:
        rm &= cond[k_]
    nsig_row = np.bincount(D.dcode[rm], minlength=D.nd)[D.dcode]
    panico = (bre < 30) | (mkt < -5) | (nsig_row >= 16)
    reg = {}
    for nm, kw in cases:
        reg[nm] = {}
        for key, rmask in (('solo_panico', panico), ('sin_panico', ~panico)):
            s7 = cartera.simulate(Wd, mask & rmask, EXIT, 'vol', rng, d0=d0, **kw)
            reg[nm][key] = cartera.metrics(s7['eq'], yrs, s7['trades'], s7['expo'])
    out['regimen'] = reg
    # (c) arriesgar según el régimen: más en pánico, menos en días normales (pesos fijados antes de simular; mismo plan de 5 y todas las señales; el régimen se definió mirando la tabla)
    sizing = {}
    pesos = (('base', 'riesgo 1 % siempre', 1.0, 1.0), ('x2_x1', 'pánico ×2, resto ×1', 2.0, 1.0), ('x2_x05', 'pánico ×2, resto ×0,5', 2.0, 0.5), ('x3_x05', 'pánico ×3, resto ×0,5', 3.0, 0.5), ('x1_x05', 'pánico ×1, resto ×0,5', 1.0, 0.5))
    for nm, kw in cases:
        sizing[nm] = {}
        for key, txt, a_, b_ in pesos:
            rr = np.where(panico, a_, b_)
            s8 = cartera.simulate(Wd, mask, EXIT, 'vol', rng, d0=d0, risk_rows=rr, **kw)
            sizing[nm][key] = {'txt': txt, **cartera.metrics(s8['eq'], yrs, s8['trades'], s8['expo'])}
    out['sizing'] = sizing
    return out


# ── conjunto ──────────────────────────────────────────────────────────────
def analyze(bars: Dict[str, Dict[str, np.ndarray]], workers: int = 1, k: int = ml.K, verbose: bool = False, join: Optional[Dict[str, str]] = None, rule: Tuple[str, ...] = RULE,
            test_years: Optional[List[int]] = None, reps: int = 300, port: bool = True, seed: int = 5) -> Dict:
    t0 = time.time()
    D, P, dropped = search.build(bars, workers, k, verbose, search.MIN_DAY, search.MIN_COMBO, join, search.COST, library=False)
    jobs = [(s, {a: b[a] for a in ('t', 'o', 'h', 'l', 'c', 'v')}) for s, b in bars.items() if s in D.syms]
    W: Dict[str, Dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s, w in ex.map(_work, jobs, chunksize=8):
                W[s] = w
    else:
        for jb in jobs:
            s, w = _work(jb)
            W[s] = w
    T = build_table(D, P, W, rule)
    if verbose:
        print(f'  tabla: {len(T):,} operaciones de la regla · {T["loss"].sum():,} perdedoras · {time.time() - t0:.0f}s', flush=True)
    years = sorted(int(y) for y in T['year'].unique())
    if test_years is None:
        test_years = [y for y in years if y >= years[0] + 3 and int((T['year'] == y).sum()) >= 150]
    split = DEV_END + 1 if int((T['year'] > DEV_END).sum()) >= 500 and int((T['year'] <= DEV_END).sum()) >= 1500 else int(np.median(T['year']))
    out: Dict = {'meta': {'generated': int(time.time()), 'universe': len(D.syms), 'filas': int(len(D.sym)), 'from': str(D.date[0]), 'to': str(D.date[-1]), 'k': k, 'coste': search.COST, 'regla': list(rule), 'salida': EXIT,
                          'miembros': 'en la fecha' if join else 'actuales', 'barras_quitadas': dropped, 'split': split, 'test_years': test_years, 'purga_dias': PURGE_DAYS, 'reps': reps}}
    out['autopsia'] = autopsy(T)
    out['univariado'] = univariate(T, split)
    out['patrones_dev'] = dev_filters(T, split)
    fo = filters_oos(T, test_years, reps, seed)
    to = tree_oos(T, split, test_years)
    out['filtros_hacia_delante'] = {k_: v for k_, v in fo.items() if k_ != 'excl'}
    out['arbol'] = {k_: v for k_, v in to.items() if k_ != 'excl'}
    out['salidas'] = exit_variants(T, split)
    out['valores'] = symbols(T, split)
    if verbose:
        print(f'  autopsia, patrones y filtros hacia delante listos · {time.time() - t0:.0f}s', flush=True)
    if port and 'excl' in fo and 'excl' in to:
        Wd = cartera.World(D, W)
        out['cartera'] = portfolio(D, Wd, T, fo['excl'], to['excl'], test_years, reps=20)
    out['_tabla_filas'] = int(len(T))
    return out


# ── informe ───────────────────────────────────────────────────────────────
def _f(x, fmt: str = '') -> str:
    """Formatea un número que puede faltar (sin datos suficientes)."""
    return 'n/d' if x is None else format(x, fmt)


def report(ev: Dict) -> str:
    m, a = ev['meta'], ev['autopsia']
    L = [f"AUTOPSIA DE LAS PERDEDORAS · regla {' + '.join(m['regla'])} · salida: escalera · {m['universe']} acciones · {m['from']} → {m['to']} · coste {m['coste']} %",
         f"Operaciones: {a['n']:,} · ganan {a['ganan']} % (ganancia media {a['gan_media']:+.2f} %) · pierden {a['perdidas']:,} (pérdida media {a['perd_media']:+.2f} %) · neto medio {a['neto_medio']:+.3f} %", '']
    L.append('CÓMO PIERDEN')
    for r in a['por_tipo']:
        L.append(f"   {r['tipo']:46s} {r['n']:6,d} ({r['pct_perdedoras']:5.1f} % de las perdedoras) · pérdida media {r['perdida_media']:+.2f} % · {r['pct_del_total_perdido']:.1f} % de lo perdido")
    s = a['stop_inicial']
    L += [f"   Stop inicial: {s['n']:,} · {s['por_hueco']} % por hueco de apertura (pérdida media {s['perdida_media_hueco']:+.2f} % frente a {s['perdida_media_intradia']:+.2f} % intradía) · distancia media del stop {s['distancia_stop_media_pct']} % del precio",
          f"   Con un día tipo «resultados» entre la entrada y la salida: {s['con_resultados_en_la_operacion']} % de las que saltaron el stop frente a {s['ganadoras_con_resultados_en_la_operacion']} % de las ganadoras; la salida cayó en un día así el {s['en_dia_resultados']} %",
          '   Rapidez del stop (sesiones hasta saltar): ' + ' · '.join(f"{r['sesiones']}: {r['pct_stops']} %" for r in a['rapidez'])]
    c = a['camino']
    L += ['', 'CAMINO DE LAS QUE SALTARON EL STOP',
          f"   Mejor cierre antes de caer, mediana {c['stop_mfe_cierre_mediana']:+.1f} % · máximo intradía mediana {c['stop_mfe_max_mediana']:+.1f} % · tocaron +2 % intradía el {c['pct_toco_2']} %, +4 % el {c['pct_toco_4']} %, +8 % el {c['pct_toco_8']} % · cerraron en +2 % o más el {c['pct_cerro_2']} % · nunca pasaron de +1 % el {c['pct_nunca_arriba']} %",
          f"   Las GANADORAS también sufren: caída máxima mediana {c['ganadoras_caida_maxima_mediana']:+.1f} % · el {c['ganadoras_con_caida_mayor_5']} % cayó más de 5 % y el {c['ganadoras_con_caida_mayor_10']} % más de 10 % antes de ganar", '']
    for key, tit in (('por_año', 'POR AÑO'), ('por_mes', 'POR MES DEL AÑO'), ('por_dia_semana', 'POR DÍA DE LA SEMANA DE LA SEÑAL'), ('por_amplitud', 'POR AMPLITUD DEL MERCADO (% de acciones sobre su media de 200)'),
                     ('por_mercado_20', 'POR RENDIMIENTO DEL MERCADO A 20 SESIONES'), ('por_señales_del_dia', 'POR NÚMERO DE SEÑALES DE LA REGLA ESE DÍA'), ('por_hueco_entrada', 'POR HUECO DE LA APERTURA DE ENTRADA'),
                     ('por_resultados_est', 'POR SESIONES DESDE EL ÚLTIMO DÍA TIPO «RESULTADOS» (los resultados vuelven cada ~63)'), ('por_volatilidad', 'POR VOLATILIDAD (ATR/precio)'),
                     ('por_resultados_proximos', 'POR SESIONES ESTIMADAS HASTA LOS PRÓXIMOS RESULTADOS (si vuelven cada 63; la operación dura unas 13)'),
                     ('tras_perdedora_del_valor', 'SEGÚN CÓMO TERMINÓ LA OPERACIÓN ANTERIOR DEL MISMO VALOR'), ('con_resultados_en_la_operacion', 'SEGÚN HAYA UN DÍA TIPO «RESULTADOS» DURANTE LA OPERACIÓN (se sabe después)')):
        rows = a.get(key) or []
        if not rows:
            continue
        L.append(tit)
        for r in rows:
            L.append(f"   {r['grupo']:44s} n {r['n']:6,d} · pierden {r['pierden']:5.1f} % · neto medio {r['neto_medio']:+.2f} %" + (f" · t {r['t']:+.1f}" if r.get('t') is not None else ''))
        L.append('')
    px = a['proxy_resultados']
    L += [f"ESTIMACIÓN DE RESULTADOS: con resultados previstos en las próximas 13 sesiones n {px['con_proxy']['n']:,} · de ellas, tuvieron un día de resultados durante la operación el {_f(px['con_proxy']['con_resultados_en_la_operacion'])} % · pierden {_f(px['con_proxy']['pierden'])} % · neto {_f(px['con_proxy']['neto_medio'], '+.2f')} %"
          f" | sin previsión n {px['sin_proxy']['n']:,} · tuvieron resultados el {_f(px['sin_proxy']['con_resultados_en_la_operacion'])} % · pierden {_f(px['sin_proxy']['pierden'])} % · neto {_f(px['sin_proxy']['neto_medio'], '+.2f')} %", '']
    L.append('RÉGIMEN DE PÁNICO POR AÑO (¿cuántos episodios hay detrás? mirado después de ver la tabla: es descripción, no evidencia)')
    for g in a['regimen_por_año']:
        L.append(f"   {g['txt']}: n {g['n']:,} en {g['dias']} días y {g['episodios']} episodios · pierden {_f(g['pierden'])} % · neto {_f(g['neto'], '+.2f')} % | resto: pierden {_f(g['pierden_resto'])} % · neto {_f(g['neto_resto'], '+.2f')} %")
        L.append('      ' + ' · '.join(f"{r['año']}: n {r['n']}" + (f" {r['pierden']:.0f} % / {r['neto']:+.1f} %" if r['neto'] is not None else '') + (f" (resto {r['neto_resto']:+.1f} %)" if r['neto_resto'] is not None else '') for r in g['por_año']))
    L.append('')
    u = ev['univariado']
    L.append(f"VARIABLES DE ENTRADA QUE MÁS SEPARAN GANADORAS DE PERDEDORAS (AUC de «pierde»: 0,5 = nada; desarrollo = años < {u['split']}, prueba = el resto; misma dirección = el signo se mantiene)")
    for r in u['ranking'][:14]:
        L.append(f"   {r['txt']:60s} AUC dev {r['auc_dev']} · test {r['auc_test']} · {'misma dirección' if r['misma_direccion'] else 'cambia de signo'}")
    L.append('')
    p = ev['patrones_dev']
    L.append(f"PATRONES MÁS FUERTES EN DESARROLLO (años < {p['split']}, uno a uno) Y QUÉ PASÓ DESPUÉS ({p['candidatos']} candidatos con efecto positivo; se mantienen {p['se_mantienen']} de los {len(p['top'])} primeros)")
    for r in p['top']:
        L.append(f"   {r['txt']:62s} dev: n {r['n_dev']:5d} · excl. {r['neto_excl_dev']:+.2f} % frente a {r['neto_resto_dev']:+.2f} % (t {r['t_dev']}) → test: n {r['n_test']:4d} · excl. {r['neto_excl_test']:+.2f} % frente a {r['neto_resto_test']:+.2f} % (t {r['t_test']}) {'SE MANTIENE' if r['se_mantiene'] else 'se pierde'}")
    L.append('')
    f = ev['filtros_hacia_delante']
    L.append(f"FILTROS ELEGIDOS HACIA DELANTE (cada año con lo anterior menos {m['purga_dias']} días; hasta 3 filtros en cadena, t ≥ {T_MIN}, ≤ {int(100 * MAX_EXCL)} % de filas cada uno)")
    for r in f['por_año']:
        L.append(f"   {r['año']}: {r['excluidas']} de {r['n']} excluidas · neto todas {r['neto_todas']:+.2f} % · quedan {r['neto_quedan']:+.2f} % · excluidas {r['neto_excluidas'] if r['neto_excluidas'] is not None else float('nan'):+.2f} % · " + (' + '.join(x['txt'] for x in r['filtros']) or 'ningún filtro'))
    if f.get('resumen'):
        s_ = f['resumen']
        L += [f"   CONJUNTO FUERA DE MUESTRA: {s_['n']:,} operaciones, {s_['excluidas']:,} excluidas ({s_['pct_excluidas']} %) · neto todas {s_['neto_todas']:+.3f} % → quedan {s_['neto_quedan']:+.3f} % (excluidas {s_['neto_excluidas']:+.3f} %) · ganan {s_['ganan_todas']} % → {s_['ganan_quedan']} %",
              f"   evita el {s_['perdedoras_evitadas_pct']} % de las perdedoras y pierde el {s_['ganadoras_perdidas_pct']} % de las ganadoras · diferencia {s_['diferencia']:+.3f} % (t {s_['t']}) · excluir lo mismo al azar deja {s_['azar_neto_quedan_mediana']:+.3f} % (p95 {s_['azar_p95']:+.3f} %): p = {s_['p_azar']}"]
    L.append('')
    t_ = ev['arbol']
    if t_.get('reglas'):
        L += ['ÁRBOL DE DECISIÓN (profundidad 3, ajustado con el desarrollo): resultado neto previsto por hoja y qué pasó después', t_['reglas'].rstrip()]
        for r in t_['hojas']:
            L.append(f"   hoja {r['hoja']:2d}: dev n {r['n_dev']:5d} neto {r['neto_dev']:+.2f} % pierden {r['pierden_dev']} % → test n {r['n_test']:5d} neto {r['neto_test'] if r['neto_test'] is not None else float('nan'):+.2f} % pierden {r['pierden_test']} %")
    if t_.get('hacia_delante'):
        s_ = t_['hacia_delante']
        L.append(f"   Hacia delante (se excluyen las hojas que en el entrenamiento rendían menos de la mitad de la media): {s_['excluidas']:,} de {s_['n']:,} excluidas ({s_['pct_excluidas']} %) · neto {s_['neto_todas']:+.3f} % → {s_['neto_quedan']:+.3f} % (excluidas {s_['neto_excluidas']:+.3f} %) · evita el {s_['perdedoras_evitadas_pct']} % de perdedoras y pierde el {s_['ganadoras_perdidas_pct']} % de ganadoras · t {s_['t']}")
    L.append('')
    sa = ev['salidas']
    L.append('VARIANTES DE SALIDA (mismas filas; fijadas de antemano; diferencia con la escalera por operación)')
    for r in sa['variantes']:
        L.append(f"   {r['txt']:60s} neto {r['neto']:+.3f} % · ganan {r['ganan']} % · gana {r['gan_media']:+.1f} % / pierde {r['perd_media']:+.1f} % · dif {r['dif_con_escalera']:+.3f} % (t {r['t_dif']}) · dev {r['dif_dev']:+.3f} % · test {r['dif_test']:+.3f} %")
    L.append('')
    v = ev['valores']
    L.append(f"VALORES QUE MÁS PIERDEN ({v['valores_con_historial']} valores; correlación entre su tasa de pérdida en desarrollo y en prueba, con ≥ 8 operaciones en cada: {v['corr_tasa_dev_test']}, {v['valores_comparables']} valores)")
    for r in v['top_perdedores']:
        L.append(f"   {r['sym']:6s} n {r['n']:3d} · pierden {r['pierden']} % · neto medio {r['neto_medio']:+.2f} % · {r['pct_del_total_perdido']} % de lo perdido")
    if ev.get('cartera'):
        cp = ev['cartera']
        L += ['', f"EN LA CARTERA SIMULADA (años de prueba desde {cp['desde']}; orden por volatilidad; riesgo 1 %): los filtros hacia delante frente a no filtrar y frente a excluir lo mismo al azar"]
        for nm, d_ in cp['carteras'].items():
            for k_, tx in (('sin_filtro', 'sin filtro'), ('filtros', 'filtros hacia delante'), ('arbol', 'árbol hacia delante')):
                if k_ in d_:
                    x = d_[k_]
                    L.append(f"   {nm:6s} {tx:24s} {x['anual']:+6.1f} %/año · caída {x['caida_max']:+5.1f} % · Sharpe {x['sharpe']} · {x['operaciones']} operaciones · ganan {x['ganan']} %")
            if 'azar' in d_:
                z = d_['azar']['anual']
                L.append(f"   {nm:6s} {'excluyendo lo mismo al azar':24s} {z['mediana']:+6.1f} %/año (p05 {z['p05']:+.1f}, p95 {z['p95']:+.1f}) · p(azar ≥ filtros) = {z['p']}")
    if ev.get('cartera', {}).get('stops_anchos'):
        L += ['', 'STOPS INICIALES MÁS ANCHOS EN LA CARTERA (escalera; mismo tamaño de posición = el de 2,5 ATR; mismo riesgo = posición más pequeña cuanto más ancho el stop)']
        for nm, d_ in ev['cartera']['stops_anchos'].items():
            for ex_, v in d_.items():
                x = v['mismo_tamaño']
                L.append(f"   {nm:6s} {ex_:5s} mismo tamaño: {x['anual']:+6.1f} %/año · caída {x['caida_max']:+5.1f} % · Sharpe {x['sharpe']} · ganan {x['ganan']} % · neto/op {x['neto_medio']:+.2f} %" + (f" | mismo riesgo: {v['mismo_riesgo']['anual']:+6.1f} %/año · caída {v['mismo_riesgo']['caida_max']:+5.1f} % · Sharpe {v['mismo_riesgo']['sharpe']}" if 'mismo_riesgo' in v else ''))
    if ev.get('cartera', {}).get('sizing'):
        L += ['', 'ARRIESGAR SEGÚN EL RÉGIMEN (más en pánico, menos en días normales; riesgo base 1 %; ventana de prueba; el régimen se definió mirando la tabla)']
        for nm, d_ in ev['cartera']['sizing'].items():
            for k_, v in d_.items():
                L.append(f"   {nm:6s} {v['txt']:26s} {v['anual']:+6.1f} %/año · caída {v['caida_max']:+5.1f} % · Sharpe {v['sharpe']} · anual/caída {v['mar']} · invertido {v['exposicion_media']:.0f} % · por año: " + ' '.join(f"{str(a_)[2:]}:{r_:+.0f}" for a_, r_ in v['por_año'].items()))
    if ev.get('cartera', {}).get('regimen'):
        L += ['', 'RÉGIMEN DE PÁNICO EN LA CARTERA (operar solo cuando amplitud < 30 %, mercado < −5 % o 16+ señales, o solo fuera de ello; ventana de prueba; mirado después de ver la tabla)']
        for nm, d_ in ev['cartera']['regimen'].items():
            for k_, v in d_.items():
                L.append(f"   {nm:6s} {k_:11s} {v['anual']:+6.1f} %/año · caída {v['caida_max']:+5.1f} % · {v['operaciones']} operaciones · ganan {v['ganan']} % · invertido {v['exposicion_media']:.0f} % · por año: " + ' '.join(f"{str(a_)[2:]}:{r_:+.0f}" for a_, r_ in v['por_año'].items()))
    return '\n'.join(L)


def run(symbols_: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, reps: int = 300, members: str = 'asof') -> Dict:
    from . import data, universe
    bars, failed = data.fetch_many(symbols_, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=24 * 3, verbose=True)
    if len(bars) < 0.9 * len(symbols_):
        raise SystemExit(f'solo {len(bars)} de {len(symbols_)} valores descargados (Yahoo limita las peticiones): se detiene; lo ya descargado queda en la caché')
    join = None
    if members == 'asof':
        try:
            join = universe.sp500_join_dates()
        except Exception as exc:
            print(f'  no se pudo leer la tabla de miembros del S&P 500 ({type(exc).__name__}: {exc}): se usan los miembros actuales', flush=True)
    ev = analyze(bars, workers=workers, reps=reps, verbose=True, join=join)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(ev, open(os.path.join(out_dir, 'autopsia.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    txt = report(ev)
    open(os.path.join(out_dir, 'autopsia_report.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)
    sys.stdout.flush()
    return ev
