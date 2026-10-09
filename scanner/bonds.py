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
        adj = np.array([np.nan if x is None else x for x in res['indicators']['adjclose'][0]['adjclose']], float)
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
