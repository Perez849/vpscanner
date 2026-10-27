"""Universos de valores: la lista de seguimiento del laboratorio (watchlist.txt, editable) y el universo del estudio (S&P 500 + Nasdaq-100, con lista de reserva)."""
from __future__ import annotations
import io
import os
import re
import urllib.request
from typing import Dict, List

WATCHLIST = ('AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX ORCL CRM ADBE QCOM INTC MU PLTR COIN SMCI ARM UBER DIS BA CAT GE JPM BAC GS V MA LLY UNH XOM CVX COST WMT HD KO PEP PFE MRK ABBV '
             'NKE SBUX SPY QQQ IWM GLD TLT XLE XLF SMH').split()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKER = re.compile(r'^[A-Z0-9^][A-Z0-9.=^-]{0,11}$')


def parse_watchlist(text: str) -> List[str]:
    """Valores de un watchlist.txt: separados por líneas, espacios o comas; # comenta; sin repetidos y en mayúsculas; los que no parecen un símbolo de Yahoo se descartan."""
    out: List[str] = []
    for line in text.splitlines():
        for tok in re.split(r'[\s,;]+', line.split('#')[0].strip()):
            t = tok.upper()
            if t and TICKER.match(t) and t not in out:
                out.append(t)
    return out


def watchlist(path: str = '') -> List[str]:
    """Lista de seguimiento: watchlist.txt de la raíz del repositorio (o `path`); si no existe o está vacía, la lista interna."""
    p = path or os.path.join(ROOT, 'watchlist.txt')
    try:
        got = parse_watchlist(open(p, encoding='utf-8').read())
    except OSError:
        got = []
    return got or list(WATCHLIST)


# lista de reserva si Wikipedia no responde: grandes compañías de EE.UU.
FALLBACK = ('AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO BRK-B LLY JPM V UNH XOM MA COST HD PG JNJ ORCL ABBV BAC KO CRM CVX MRK NFLX AMD PEP TMO WMT ADBE LIN ACN MCD CSCO ABT WFC TXN DIS GE IBM '
            'QCOM AMGN CAT INTU VZ PM ISRG NOW DHR UBER AXP NEE PFE RTX HON SPGI GS LOW BKNG T UNP AMAT TJX SYK ELV BLK PGR C MS LMT DE VRTX MDT ETN CB BSX SCHW ADI PLD MMC REGN ADP KLAC PANW BX '
            'CMG MU ANET SBUX GILD MDLZ CVS SO MO FI TMUS BMY ICE DUK ZTS CL SHW WM EOG EQIX MCK APH CDNS SNPS ITW PYPL TT HCA NOC USB PNC CSX FCX NKE MAR ORLY GD MMM EMR ECL COF PH CTAS '
            'ROP AJG APD MSI FDX OXY AON CARR PSX MPC TGT NSC DXCM AIG SLB HLT KMB PCAR TRV AZO KDP GM F COIN PLTR SMCI ARM MRVL CRWD SNOW DDOG NET ABNB LRCX WDAY TEAM ADSK MELI ASML').split()


def _read_html(url: str):
    import pandas as pd
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return pd.read_html(io.StringIO(r.read().decode('utf-8')))


def _clean(xs) -> List[str]:
    return sorted({str(x).strip().upper().replace('.', '-') for x in xs if isinstance(x, str) and 0 < len(str(x).strip()) <= 6})


def sp500() -> List[str]:
    t = _read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return _clean(t['Symbol'])


def _flat_cols(t) -> List[str]:
    return [' '.join(str(x) for x in (c if isinstance(c, tuple) else (c,)) if 'unnamed' not in str(x).lower()).strip().lower() for c in t.columns]


def _changes_cols(cols: List[str]):
    """Columnas (fecha, ticker añadido, ticker retirado) de una tabla de cambios de Wikipedia, o None si no lo es. Se ignoran mayúsculas; vale «ticker» o «symbol»."""
    dcol = next((c for c in cols if 'date' in c), None)
    acol = next((c for c in cols if 'added' in c and ('ticker' in c or 'symbol' in c)), None)
    rcol = next((c for c in cols if 'removed' in c and ('ticker' in c or 'symbol' in c)), None)
    return (dcol, acol, rcol) if dcol and acol and rcol else None


def parse_sp500_changes(tables) -> Dict[str, str]:
    """Tablas de la página del S&P 500 de Wikipedia (o una sola) → fecha (AAAA-MM-DD) de la ÚLTIMA incorporación de cada valor que no ha salido después, leída de la tabla de cambios
    («Selected changes to the list of S&P 500 components»). Los valores que no aparecen como incorporados están en el índice desde antes del primer cambio de la tabla."""
    import pandas as pd
    tables = tables if isinstance(tables, (list, tuple)) else [tables]
    found = None
    for t in tables:
        cols = _flat_cols(t)
        m = _changes_cols(cols)
        if m:
            found = (t.copy(), cols, m)
            break
    if found is None:
        raise ValueError('ninguna tabla tiene columnas de fecha, añadido y retirado: ' + ' | '.join(str(_flat_cols(t))[:160] for t in tables[:6]))
    t, cols, (dcol, acol, rcol) = found
    t.columns = cols
    ev = []
    for _, r in t.iterrows():
        d = pd.to_datetime(r[dcol], errors='coerce')
        if pd.isna(d):
            continue
        for col, kind in ((acol, 'add'), (rcol, 'rem')):
            v = r[col]
            if isinstance(v, str) and v.strip():
                ev.append((d, kind, v.strip().upper().replace('.', '-')))
    ev.sort(key=lambda x: (x[0], 0 if x[1] == 'rem' else 1))           # el mismo día: primero salidas, luego entradas
    state: Dict[str, str] = {}
    for d, kind, tk in ev:
        if kind == 'add':
            state[tk] = d.strftime('%Y-%m-%d')
        else:
            state.pop(tk, None)
    if not state:
        raise ValueError('la tabla de cambios no dio ninguna incorporación')
    return state


def sp500_join_dates() -> Dict[str, str]:
    """Fecha de la última incorporación al S&P 500 de cada miembro actual que entró en algún momento de los cambios de la tabla."""
    return parse_sp500_changes(_read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'))


def nasdaq100() -> List[str]:
    for t in _read_html('https://en.wikipedia.org/wiki/Nasdaq-100'):
        for col in ('Ticker', 'Symbol'):
            if col in t.columns and len(t) >= 90:
                return _clean(t[col])
    return []


def sp400() -> List[str]:
    t = _read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]
    return _clean(t['Symbol'])


def sp600() -> List[str]:
    t = _read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]
    return _clean(t['Symbol'])


def smallmid() -> List[str]:
    """Acciones medianas y pequeñas de EE.UU.: miembros actuales del S&P 400 y del S&P 600 (menos líquidas y menos vigiladas que las grandes). Vacía si Wikipedia no responde."""
    got: List[str] = []
    for fn in (sp400, sp600):
        try:
            got += fn()
        except Exception as exc:
            print(f'  universo: {fn.__name__} falló ({type(exc).__name__}: {exc})', flush=True)
    return sorted(set(got))


# cesta de otros mercados: criptomonedas, divisas, materias primas, bonos, índices bursátiles y futuros de índices
MIXED = ('BTC-USD ETH-USD BNB-USD XRP-USD ADA-USD SOL-USD DOGE-USD LTC-USD LINK-USD XLM-USD '
         'EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCAD=X USDCHF=X NZDUSD=X EURJPY=X GBPJPY=X EURGBP=X '
         'GC=F SI=F HG=F CL=F BZ=F NG=F ZC=F ZW=F ZS=F KC=F SB=F CT=F '
         'ES=F NQ=F YM=F RTY=F ZN=F ZB=F '
         '^GSPC ^NDX ^DJI ^RUT ^FTSE ^GDAXI ^FCHI ^N225 ^HSI ^STOXX50E ^IBEX').split()


def mixed() -> List[str]:
    return list(MIXED)


def universe_by_name(name: str) -> List[str]:
    return {'large': study_universe, 'smallmid': smallmid, 'mixed': mixed}[name]()


def study_universe() -> List[str]:
    """S&P 500 ∪ Nasdaq-100 ∪ lista de seguimiento. Si Wikipedia falla, la lista de reserva. Devuelve (valores)."""
    got: List[str] = []
    for fn in (sp500, nasdaq100):
        try:
            got += fn()
        except Exception as exc:                          # sin red, cambio de formato…
            print(f'  universo: {fn.__name__} falló ({type(exc).__name__}: {exc})', flush=True)
    base = got if len(got) >= 300 else FALLBACK + got
    return sorted(set(base) | set(WATCHLIST) - {'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'XLE', 'XLF', 'SMH'})
