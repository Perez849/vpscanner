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


def parse_sp500_join_dates(tables) -> Dict[str, str]:
    """Tabla de miembros de la página del S&P 500 de Wikipedia (o una sola) → fecha de incorporación al índice (AAAA-MM-DD) de cada miembro actual, de su columna «Date added».
    Las celdas sin fecha legible se ignoran (el valor cuenta como miembro desde siempre)."""
    import re
    tables = tables if isinstance(tables, (list, tuple)) else [tables]
    seen = []
    for t in tables:
        cols = [' '.join(dict.fromkeys(str(x).strip().lower() for x in (c if isinstance(c, tuple) else (c,)) if 'unnamed' not in str(x).lower())) for c in t.columns]     # niveles repetidos de la cabecera: una sola vez
        seen.append(cols)
        sc = next((c for c in cols if c in ('symbol', 'ticker')), None)
        dc = next((c for c in cols if 'date' in c and 'added' in c), None)
        if not (sc and dc):
            continue
        out: Dict[str, str] = {}
        for sym, raw in zip(t.iloc[:, cols.index(sc)], t.iloc[:, cols.index(dc)]):
            m = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(raw))
            if isinstance(sym, str) and sym.strip() and m:
                out[sym.strip().upper().replace('.', '-')] = m.group(0)
        if out:
            return out
    raise ValueError('ninguna tabla tiene símbolo y fecha de incorporación: ' + ' | '.join(str(c)[:160] for c in seen[:6]))


def sp500_join_dates() -> Dict[str, str]:
    """Fecha de incorporación al S&P 500 de cada miembro actual (columna «Date added» de la tabla de miembros de Wikipedia)."""
    return parse_sp500_join_dates(_read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'))


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
