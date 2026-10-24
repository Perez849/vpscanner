"""Universos de valores: la lista de seguimiento del laboratorio y el universo del estudio (S&P 500 + Nasdaq-100, con lista de reserva)."""
from __future__ import annotations
import io
import urllib.request
from typing import List

WATCHLIST = ('AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX ORCL CRM ADBE QCOM INTC MU PLTR COIN SMCI ARM UBER DIS BA CAT GE JPM BAC GS V MA LLY UNH XOM CVX COST WMT HD KO PEP PFE MRK ABBV '
             'NKE SBUX SPY QQQ IWM GLD TLT XLE XLF SMH').split()

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


def nasdaq100() -> List[str]:
    for t in _read_html('https://en.wikipedia.org/wiki/Nasdaq-100'):
        for col in ('Ticker', 'Symbol'):
            if col in t.columns and len(t) >= 90:
                return _clean(t[col])
    return []


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
