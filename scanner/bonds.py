#!/usr/bin/env python3
"""
bonds.py — renta fija: ¿puede una cartera 60 % TLT + 30 % MBB batir al índice (AGG) en 5 años? (comando `bonds`; NO interviene en producción)

Con pesos FIJOS la rentabilidad de la cartera es la media ponderada de las de sus componentes, así que si TLT rinde mucho menos que AGG y MBB
lo mismo, la cartera tiene que quedar por debajo: solo hay tres salidas (1) el 10 % restante rinde muchísimo, (2) los pesos no eran fijos
(cartera táctica) o la ventana no es la misma, (3) se comparan cosas distintas (precio sin cupones frente a rentabilidad total, otro índice).
Este estudio descarga la rentabilidad total (con cupones/dividendos) y el precio sin dividendos de TLT, MBB, AGG, BND, IEF, SHY, TIP…, y
mide ventanas de 1/3/5/10 años y de 5 años móviles.
"""
from __future__ import annotations
import json
import math
import sys
import time
import urllib.parse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import data as D

TICKERS = ['TLT', 'MBB', 'AGG', 'BND', 'IEF', 'SHY', 'BIL', 'TIP', 'GOVT', 'LQD', 'HYG', 'VGLT', 'SPY']


def fetch_full(sym: str, period1: int) -> Optional[Dict]:
    """Barras diarias con precio sin ajustar (close), rentabilidad total (adjclose) y dividendos."""
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym, safe="")}'
           f'?interval=1d&period1={int(period1)}&period2={int(time.time()) + 86400}&includeAdjustedClose=true&events=div%7Csplit')
    raw = D._get(url)
    if not raw:
        return None
    try:
        res = json.loads(raw)['chart']['result'][0]
        ts = res['timestamp']
        close = np.array([np.nan if x is None else x for x in res['indicators']['quote'][0]['close']], float)
        try:
            adj = np.array([np.nan if x is None else x for x in res['indicators']['adjclose'][0]['adjclose']], float)
        except Exception:
            adj = close.copy()
        divs = res.get('events', {}).get('dividends', {})
        dv = sorted((int(v['date']), float(v['amount'])) for v in divs.values())
        idx = pd.to_datetime(np.array(ts) // 86400 * 86400, unit='s')
        df = pd.DataFrame({'close': close, 'adj': adj}, index=idx)
        df = df[~df.index.duplicated(keep='last')].dropna()
        # la barra de hoy puede estar incompleta
        if len(df) and df.index[-1].date() == pd.Timestamp.utcnow().date():
            df = df.iloc[:-1]
        return {'df': df, 'div': dv}
    except Exception:
        return None


def window_ret(s: pd.Series, years: float) -> Optional[float]:
    end = s.index[-1]
    start = end - pd.DateOffset(years=int(years)) if years == int(years) else end - pd.Timedelta(days=int(365.25 * years))
    if s.index[0] > start + pd.Timedelta(days=5):
        return None
    s2 = s[s.index >= start]
    return float(s2.iloc[-1] / s2.iloc[0] - 1)


def portfolio(rets: pd.DataFrame, w: Dict[str, float], rebalance: str = 'M') -> pd.Series:
    """Rentabilidad diaria de una cartera con pesos objetivo; rebalanceo mensual ('M'), anual ('A') o ninguno ('N')."""
    names = list(w)
    r = rets[names].dropna()
    tw = np.array([w[n] for n in names], float)
    cur = tw.copy()
    out = np.zeros(len(r))
    last = None
    for i, (d, row) in enumerate(r.iterrows()):
        key = d.month if rebalance == 'M' else d.year
        if rebalance in ('M', 'A') and last is not None and key != last:
            cur = tw.copy()
        last = key if rebalance in ('M', 'A') else 0
        x = row.to_numpy()
        pr = float((cur * x).sum())
        out[i] = pr
        cur = cur * (1 + x) / (1 + pr)
    return pd.Series(out, index=r.index)


def stats(r: pd.Series) -> Dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    return {'tot': float(eq.iloc[-1] - 1) * 100, 'cagr': float(eq.iloc[-1] ** (1 / yrs) - 1) * 100,
            'vol': float(r.std() * math.sqrt(252) * 100), 'dd': float((eq / eq.cummax() - 1).min() * 100)}


def run(uni, data, out_dir):
    p1 = int(pd.Timestamp('2007-01-02').timestamp())
    S: Dict[str, Dict] = {}
    for t in TICKERS:
        o = fetch_full(t, p1)
        if o is not None:
            S[t] = o
    print(f'\n######## RENTA FIJA · descargados {sorted(S)} · hasta {max(v["df"].index[-1] for v in S.values()).date()}', flush=True)
    for t, v in S.items():
        df = v['df']
        print(f'  {t:5s} desde {df.index[0].date()} · {len(df)} barras')
    tr = pd.DataFrame({t: v['df']['adj'] for t, v in S.items()})            # rentabilidad total
    px = pd.DataFrame({t: v['df']['close'] for t, v in S.items()})          # precio sin dividendos
    last = tr.dropna(subset=['TLT', 'MBB', 'AGG']).index[-1]

    # rendimiento por distribuciones (últimos 12 meses)
    print('\n  RENDIMIENTO POR DISTRIBUCIONES (suma de dividendos de los últimos 12 meses / último precio):')
    for t in ('TLT', 'MBB', 'AGG', 'BND', 'IEF', 'SHY'):
        if t in S:
            cut = int((last - pd.DateOffset(years=1)).timestamp())
            d12 = sum(a for d, a in S[t]['div'] if d >= cut)
            print(f"    {t:5s} {d12 / S[t]['df']['close'].iloc[-1] * 100:4.2f} %")

    def win_table(title: str, frame: pd.DataFrame):
        print(f'\n  {title}')
        print('    ' + 'ventana'.ljust(10) + ''.join(t.rjust(9) for t in ('TLT', 'MBB', 'AGG', 'BND', 'IEF', 'SHY', 'TIP', 'SPY') if t in frame))
        for yrs in (1, 2, 3, 4, 5, 7, 10):
            row = []
            for t in ('TLT', 'MBB', 'AGG', 'BND', 'IEF', 'SHY', 'TIP', 'SPY'):
                if t not in frame:
                    continue
                v = window_ret(frame[t].dropna(), yrs)
                row.append(f'{v * 100:8.1f}%' if v is not None else '       —')
            print('    ' + f'{yrs} años'.ljust(10) + ''.join(row))
    win_table('RENTABILIDAD TOTAL acumulada (con cupones y dividendos) hasta la última fecha', tr)
    win_table('SOLO PRECIO (sin contar cupones/dividendos): lo que se ve en un gráfico de precio', px)

    rets = tr.pct_change()
    ports = {
        '60 TLT / 30 MBB / 10 liquidez (SHY)': {'TLT': .6, 'MBB': .3, 'SHY': .1},
        '60 TLT / 30 MBB / 10 AGG': {'TLT': .6, 'MBB': .3, 'AGG': .1},
        '60 TLT / 30 MBB / 10 TIP': {'TLT': .6, 'MBB': .3, 'TIP': .1},
        '60 TLT / 30 MBB / 10 HYG': {'TLT': .6, 'MBB': .3, 'HYG': .1},
        '60 TLT / 40 MBB': {'TLT': .6, 'MBB': .4},
        'AGG (el índice)': {'AGG': 1.0},
        'BND': {'BND': 1.0},
        '100 TLT': {'TLT': 1.0},
        '100 MBB': {'MBB': 1.0},
    }
    P = {k: portfolio(rets, w, 'M') for k, w in ports.items() if all(n in rets for n in w)}
    print('\n  CARTERAS DE PESOS FIJOS (rebalanceo mensual, rentabilidad total, sin costes)')
    print('    ' + 'cartera'.ljust(40) + ''.join(f'{y} años'.rjust(11) for y in (1, 3, 5, 10)) + '   vol 5a   caída 5a')
    for k, r in P.items():
        row = []
        for yrs in (1, 3, 5, 10):
            end = r.index[-1]
            sub = r[r.index > end - pd.DateOffset(years=yrs)]
            if r.index[0] > end - pd.DateOffset(years=yrs) + pd.Timedelta(days=5):
                row.append('        —')
            else:
                row.append(f'{((1 + sub).prod() - 1) * 100:9.1f}%')
        sub5 = r[r.index > r.index[-1] - pd.DateOffset(years=5)]
        s5 = stats(sub5)
        print('    ' + k.ljust(40) + ' '.join(row) + f'   {s5["vol"]:5.1f}%   {s5["dd"]:6.1f}%')

    # años naturales
    print('\n  AÑOS NATURALES (rentabilidad total):')
    yrs_ = sorted({d.year for d in tr.index if d.year >= 2018})
    heads = ['AGG (el índice)', '100 TLT', '100 MBB', '60 TLT / 30 MBB / 10 liquidez (SHY)']
    print('    ' + 'año'.ljust(6) + ''.join(h[:22].rjust(24) for h in heads))
    for y in yrs_:
        cells = []
        for h in heads:
            r = P.get(h)
            if r is None:
                cells.append('—'.rjust(24)); continue
            sub = r[r.index.year == y]
            cells.append(f'{((1 + sub).prod() - 1) * 100:+.1f}%'.rjust(24))
        print('    ' + str(y).ljust(6) + ''.join(cells))

    # ventanas móviles de 5 años: ¿con qué frecuencia 60/30/10 bate a AGG?
    A = P['AGG (el índice)']
    B = P['60 TLT / 30 MBB / 10 liquidez (SHY)']
    ends = [d for d in B.index[::21] if d >= B.index[0] + pd.DateOffset(years=5)]
    if B.index[-1] not in ends:
        ends.append(B.index[-1])
    wins, diffs, rows = 0, [], []
    for e in ends:
        s = e - pd.DateOffset(years=5)
        a = float((1 + A[(A.index > s) & (A.index <= e)]).prod() - 1)
        b = float((1 + B[(B.index > s) & (B.index <= e)]).prod() - 1)
        wins += b > a
        diffs.append(b - a)
        rows.append((e, a, b))
    print(f'\n  VENTANAS MÓVILES DE 5 AÑOS (fin de cada mes desde {ends[0].date()}): la cartera 60/30/10 bate a AGG en {wins} de {len(ends)} ({wins / len(ends) * 100:.0f} %) · diferencia mediana {np.median(diffs) * 100:+.1f} pts')
    print('    por año de finalización de la ventana (media de la diferencia cartera − AGG, en puntos de rentabilidad acumulada a 5 años):')
    byy: Dict[int, List[float]] = {}
    for (e, a, b) in rows:
        byy.setdefault(e.year, []).append(b - a)
    print('    ' + ' · '.join(f'{y}: {np.mean(v) * 100:+.1f}' for y, v in sorted(byy.items())))
    e, a, b = rows[-1]
    print(f'    la última ventana (hasta {e.date()}): AGG {a * 100:+.1f}% · cartera 60/30/10 {b * 100:+.1f}%')

    # si la cartera fuera TÁCTICA: ¿cuánto tendría que ser el peso medio de TLT para empatar con AGG en 5 años?
    sub = rets[['TLT', 'MBB', 'AGG']].dropna()
    sub = sub[sub.index > sub.index[-1] - pd.DateOffset(years=5)]
    tl, mb, ag = [(1 + sub[c]).prod() - 1 for c in ('TLT', 'MBB', 'AGG')]
    print(f'\n  ÚLTIMOS 5 AÑOS: TLT {tl * 100:+.1f}% · MBB {mb * 100:+.1f}% · AGG {ag * 100:+.1f}%. Con peso fijo en TLT y MBB la media ponderada 60/30 (+10 % liquidez) '
          f'es ≈ {(0.6 * tl + 0.3 * mb) * 100:+.1f}% frente a {ag * 100:+.1f}% del índice (la diferencia solo puede salir de la liquidez, de otro índice o de pesos no fijos).')
    sys.stdout.flush()



# ═════════════════════════════════════════════════════════════════════════
#  «Reloj» de renta fija: 2/3 TLT + 1/3 MBB con interrupciones tácticas
# ═════════════════════════════════════════════════════════════════════════
ALT = ['SHY', 'TIP', 'DBC', 'GLD']
SW_COST = 0.0005          # 0,05 % por cada cambio de activo (ida y vuelta de la parte que se mueve, aproximación)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=int(n * 0.9)).mean()


def tactical(rets: pd.DataFrame, flag: pd.Series, alt: str, w_tlt: float = 2 / 3, w_mbb: float = 1 / 3, whole: bool = False) -> pd.Series:
    """Rentabilidad diaria: 2/3 TLT + 1/3 MBB; cuando `flag` (decidida con el cierre de AYER) está activa, la parte de TLT (o toda la cartera si whole) pasa a `alt`."""
    f = flag.shift(1).fillna(False).astype(bool)
    r_tlt = np.where(f, rets[alt], rets['TLT'])
    r_mbb = np.where(f & whole, rets[alt], rets['MBB'])
    out = w_tlt * r_tlt + w_mbb * r_mbb
    switches = (f != f.shift(1)).fillna(False)
    cost = switches.to_numpy() * SW_COST * (1.0 if whole else w_tlt)
    return pd.Series(out - cost, index=rets.index)


def run2(uni, data, out_dir):
    p1 = int(pd.Timestamp('2003-01-02').timestamp())
    S = {}
    for t in ['TLT', 'MBB', 'AGG', 'IEF', 'SHY', 'TIP', 'DBC', 'GLD', '^TNX']:
        o = fetch_full(t, p1)
        if o is not None:
            S[t] = o['df']
    print(f'\n######## «RELOJ» DE RENTA FIJA · descargados {sorted(S)}', flush=True)
    tr = pd.DataFrame({t: v['adj'] for t, v in S.items() if t != '^TNX'}).dropna(subset=['TLT', 'MBB', 'AGG', 'SHY'])
    tnx = S['^TNX']['close'].reindex(tr.index).ffill()
    rets = tr.pct_change().fillna(0.0)
    end = tr.index[-1]

    def cum(r, a, b):
        x = r[(r.index > a) & (r.index <= b)]
        return float((1 + x).prod() - 1) * 100

    starts = {'1-ene-2021': pd.Timestamp('2021-01-04'), '5 años exactos (2021-10-08)': end - pd.DateOffset(years=5), '1-ene-2022': pd.Timestamp('2022-01-03')}
    agg = rets['AGG']
    static = 2 / 3 * rets['TLT'] + 1 / 3 * rets['MBB']
    print('\n  A. CARTERA ESTÁTICA 2/3 TLT + 1/3 MBB (pesos fijos, rebalanceo diario) frente a AGG, rentabilidad total hasta ' + str(end.date()))
    for lab, a in starts.items():
        print(f"    desde {lab:28s} estática {cum(static, a, end):+7.1f}% · TLT {cum(rets['TLT'], a, end):+7.1f}% · MBB {cum(rets['MBB'], a, end):+6.1f}% · AGG {cum(agg, a, end):+6.1f}% · SHY {cum(rets['SHY'], a, end):+5.1f}%")
    a5 = starts['5 años exactos (2021-10-08)']

    # B. ¿cuántos meses de interrupción hacen falta? (con conocimiento del futuro: cota superior)
    m = static[static.index > a5].groupby([static[static.index > a5].index.year, static[static.index > a5].index.month]).apply(lambda x: (1 + x).prod() - 1)
    m_agg = cum(agg, a5, end)
    msh = rets['SHY'][rets.index > a5].groupby([rets['SHY'][rets.index > a5].index.year, rets['SHY'][rets.index > a5].index.month]).apply(lambda x: (1 + x).prod() - 1)
    order = m.sort_values().index
    print(f'\n  B. COTA CON CONOCIMIENTO DEL FUTURO (5 años, {len(m)} meses): si los k PEORES meses de la cartera estática se pasaran a liquidez (SHY), ¿se bate a AGG ({m_agg:+.1f}%)?')
    need = None
    for k in (0, 2, 4, 6, 8, 10, 12, 15):
        mm = m.copy()
        mm.loc[order[:k]] = msh.loc[order[:k]]
        tot = float((1 + mm).prod() - 1) * 100
        if need is None and tot > m_agg:
            need = k
        print(f'    k={k:2d} meses fuera → {tot:+6.1f}%' + ('  ← ya bate a AGG' if tot > m_agg else ''))
    wm = m.sort_values().head(10)
    print('    los 10 peores meses de la estática: ' + ' '.join(f'{y}-{mo:02d}:{v * 100:+.1f}%' for (y, mo), v in wm.items()))
    print(f'    (con conocimiento perfecto bastan ≈ {need} meses de 60: una interrupción «corta» puede dar la vuelta al resultado SI se acierta cuándo)')

    # C. reglas de «sobrecalentamiento» sin mirar al futuro
    trend_tlt = tr['TLT'] < sma(tr['TLT'], 200)
    tnx_up = tnx > sma(tnx, 200)
    tnx_shock = (tnx - tnx.shift(63)) > 0.40
    be = tr['TIP'] / tr['IEF']
    infl_up = be > sma(be, 200)
    mom_neg = tr['TLT'] / tr['TLT'].shift(126) - 1 < 0
    rules = {'TLT bajo su media de 200 sesiones': trend_tlt, 'rentabilidad del bono a 10 años sobre su media de 200': tnx_up,
             'subida de >0,40 puntos del 10 años en 3 meses': tnx_shock, 'expectativas de inflación al alza (TIP/IEF sobre su media de 200)': infl_up,
             'TLT con momentum negativo a 6 meses': mom_neg}
    print(f'\n  C. REGLAS SIN MIRAR AL FUTURO (señal con el cierre de ayer; {SW_COST * 100:.2f} % por cambio): 2/3 TLT + 1/3 MBB y, si la regla marca «sobrecalentamiento», la parte de TLT pasa al activo alternativo')
    print(f'     5 años hasta {end.date()}: estática {cum(static, a5, end):+.1f}% · AGG {cum(agg, a5, end):+.1f}%')
    print('     ' + 'regla'.ljust(62) + 'alt'.ljust(5) + ' 5 años  vs AGG  %días en alerta  cambios  caída   | ventanas de 5 años desde 2008: bate a AGG en · dif. mediana · peor')
    ends = [d for d in tr.index[::21] if d >= pd.Timestamp('2008-06-30') + pd.DateOffset(years=5)]
    rows = []
    for rn, fl in rules.items():
        for alt in ALT:
            if alt not in rets:
                continue
            r = tactical(rets, fl, alt)
            win = []
            for e in ends:
                s_ = e - pd.DateOffset(years=5)
                win.append(cum(r, s_, e) - cum(agg, s_, e))
            win = np.array(win)
            x = r[r.index > a5]
            eq = (1 + x).cumprod()
            fl5 = fl.shift(1).fillna(False)[fl.index > a5]
            sw = int((fl5 != fl5.shift(1)).sum())
            rows.append((rn, alt, cum(r, a5, end), cum(agg, a5, end), float(fl5.mean() * 100), sw, float((eq / eq.cummax() - 1).min() * 100), float((win > 0).mean() * 100), float(np.median(win)), float(win.min())))
    for rn, alt, t5, ta, pdays, sw, dd, wp, wmed, wmin in rows:
        print(f'     {rn[:60]:62s}{alt:5s}{t5:+6.1f}% {t5 - ta:+6.1f}  {pdays:10.0f} %   {sw:6d}  {dd:6.1f}% | {wp:4.0f} % · {wmed:+6.1f} pts · {wmin:+6.1f} pts')
    beat = sum(1 for r in rows if r[2] > r[3])
    print(f'     → {beat} de {len(rows)} variantes baten a AGG en los últimos 5 años; en las ventanas móviles de 5 años desde 2008 la mediana de variantes bate a AGG en {np.median([r[7] for r in rows]):.0f} % de las ventanas')
    base_w = np.array([cum(static, e - pd.DateOffset(years=5), e) - cum(agg, e - pd.DateOffset(years=5), e) for e in ends])
    print(f'     referencia: la estática 2/3 TLT + 1/3 MBB bate a AGG en {np.mean(base_w > 0) * 100:.0f} % de esas ventanas · dif. mediana {np.median(base_w):+.1f} pts · peor {base_w.min():+.1f} pts')

    # D. la mejor regla: ¿en qué meses se movió?
    best = max(rows, key=lambda r: r[2])
    print(f'\n  D. LA MEJOR de las {len(rows)} (OJO: elegida mirando estos mismos 5 años): «{best[0]}» → {best[1]}: {best[2]:+.1f}% frente a AGG {best[3]:+.1f}%')
    sys.stdout.flush()
