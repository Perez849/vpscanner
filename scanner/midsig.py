"""
midsig.py — Señales de MEDIO plazo (se calculan a diario con los datos del scan; no generan alertas ni operaciones en el seguimiento).

  · Tendencia: índice (SPY, QQQ, IWM, EFA) sobre su media de 200 sesiones → dentro; bajo ella → liquidez.
  · Rotación: cada mes, los 3 ETF con mejor momentum medio (3, 6 y 12 meses) Y positivo; el resto, en liquidez.

Validado en `research.py midterm` (2005–2026, incluye 2008; netos de costes; señal con el cierre y ejecución al cierre del día siguiente).
Para qué sirve: NO bate al S&P 500 en rentabilidad; recorta las grandes caídas (−55 % → −25 %) y se combina bien con el sistema de rebotes.
"""
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd

TREND_SYMS = ['SPY', 'QQQ', 'IWM', 'EFA']
ROT_ETFS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC',
            'TLT', 'IEF', 'HYG', 'GLD', 'DBC', 'VNQ']
LOOKBACKS = (63, 126, 252)
TOP_K = 3

# Resultados de la validación (research.py midterm, 2005-01 → 2026-10, netos de costes). Se actualizan a mano al repetir el estudio.
STATS = {
    'period': '2005–2026 (21 años, incluye 2008)', 'source': 'research.py midterm',
    'rows': [
        {'name': 'S&P 500 comprar y mantener', 'cagr': 10.9, 'vol': 18.9, 'sharpe': 0.64, 'dd': -55.2, 'worstMonth': -16.5},
        {'name': 'Tendencia SPY (sobre su media de 200)', 'cagr': 8.5, 'vol': 11.6, 'sharpe': 0.76, 'dd': -25.1, 'worstMonth': -8.2},
        {'name': 'Tendencia QQQ', 'cagr': 12.2, 'vol': 15.6, 'sharpe': 0.82, 'dd': -26.5, 'worstMonth': -12.2},
        {'name': 'Rotación de ETF por momentum (3 mejores)', 'cagr': 10.3, 'vol': 16.9, 'sharpe': 0.66, 'dd': -25.2, 'worstMonth': -10.1},
    ],
    'crises': {'2008': {'sp500': -36.8, 'trend': 4.0, 'rotation': 7.2}, '2018': {'sp500': -4.6, 'trend': -3.9, 'rotation': -2.0},
               '2022': {'sp500': -18.2, 'trend': -20.1, 'rotation': 10.2}},
    'combo2019': {'dip': {'cagr': 32.9, 'vol': 17.2, 'sharpe': 1.74, 'dd': -27.6, 'worstMonth': -20.8},
                  'mix': {'cagr': 23.0, 'vol': 11.0, 'sharpe': 1.93, 'dd': -18.7, 'worstMonth': -14.3},
                  'note': '50 % rebotes (Equilibrado) + 50 % tendencia SPY, 2019–2026; correlación diaria ≈ +0,1'},
}


def _days(b: Dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(b['t']) // 86400).astype(np.int64)


def trend_status(b: Dict[str, np.ndarray]) -> Dict[str, Any] | None:
    c = np.asarray(b['c'], dtype=float)
    if len(c) < 205:
        return None
    sma = pd.Series(c).rolling(200).mean().to_numpy()
    above = c > sma
    ok = np.isfinite(sma)
    last = len(c) - 1
    side = bool(above[last])
    k = last
    while k > 0 and ok[k - 1] and bool(above[k - 1]) == side:
        k -= 1
    flips = int(np.sum(above[last - 19:last + 1][1:] != above[last - 19:last + 1][:-1])) if last >= 20 else 0
    return {'close': round(float(c[last]), 2), 'sma200': round(float(sma[last]), 2), 'distPct': round(float((c[last] / sma[last] - 1) * 100), 2),
            'above': side, 'daysSince': int(last - k + 1), 'flips20': flips}


def rotation_at(bars: Dict[str, Dict[str, np.ndarray]], cut_day: int, pool: List[str] | None = None, top_k: int = TOP_K) -> Dict[str, Any]:
    """Ranking por momentum medio (3, 6 y 12 meses) con el último dato ≤ cut_day; entran los top_k con momentum medio > 0."""
    rows = []
    for s in (pool or ROT_ETFS):
        b = bars.get(s)
        if b is None:
            continue
        d = _days(b)
        i = int(np.searchsorted(d, cut_day, side='right')) - 1
        c = np.asarray(b['c'], dtype=float)
        if i < max(LOOKBACKS) or not np.isfinite(c[i]):
            continue
        moms = []
        for lb in LOOKBACKS:
            if not (np.isfinite(c[i - lb]) and c[i - lb] > 0):
                moms = None; break
            moms.append(float(c[i] / c[i - lb] - 1) * 100)
        if moms:
            rows.append({'sym': s, 'm63': round(moms[0], 1), 'm126': round(moms[1], 1), 'm252': round(moms[2], 1), 'avg': round(float(np.mean(moms)), 1)})
    rows.sort(key=lambda r: -r['avg'])
    for k, r in enumerate(rows):
        r['rank'] = k + 1
    held = [r for r in rows if r['avg'] > 0][:top_k]
    for r in held:
        r['w'] = round(1.0 / top_k, 4)
    return {'cutDay': int(cut_day), 'ranking': rows, 'holdings': held, 'cashW': round(1.0 - len(held) / top_k, 4)}


def month_end_before(last_day: int) -> int:
    """Último día de datos que cierra un mes (el último día natural del mes anterior, o el propio último día si ya lo cierra)."""
    d = pd.Timestamp(last_day * 86400, unit='s')
    nxt = d + pd.offsets.BDay(1)
    if nxt.month != d.month:
        return int(last_day)
    return int(((d.replace(day=1) - pd.Timedelta(days=1)).timestamp()) // 86400)


def compute(bars: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any] | None:
    spy = bars.get('SPY')
    if spy is None:
        return None
    last_day = int(_days(spy)[-1])
    trend = []
    for s in TREND_SYMS:
        if s in bars:
            t = trend_status(bars[s])
            if t:
                trend.append({'sym': s, **t})
    cut = month_end_before(last_day)
    cur = rotation_at(bars, cut)
    prev = rotation_at(bars, last_day)
    d = pd.Timestamp(last_day * 86400, unit='s')
    nxt_me = (d + pd.offsets.MonthEnd(0)) if (d + pd.offsets.BDay(1)).month == d.month else d
    return {'asOf': str(d.date()), 'trend': trend,
            'rotation': {'cut': str(pd.Timestamp(cut * 86400, unit='s').date()), 'holdings': cur['holdings'], 'cashW': cur['cashW'],
                         'previewHoldings': prev['holdings'], 'previewCashW': prev['cashW'], 'ranking': prev['ranking'][:12],
                         'nextRebalance': str(nxt_me.date())},
            'stats': STATS}
