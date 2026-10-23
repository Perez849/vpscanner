#!/usr/bin/env python3
"""
alt.py — DIMENSIONES MÁS ALLÁ DEL PRECIO Y EL VOLUMEN (comandos `altprobe` y `alt`; NO intervienen en producción).

Hasta ahora el sistema solo veía precios y volúmenes diarios (AUC ≈ 0,51–0,52). Aquí se buscan datos nuevos, gratuitos y con fecha de publicación (para que todo sea
causal): operaciones de directivos (Form 4 de la SEC), ventas en corto diarias (FINRA Reg SHO), calendario de resultados (8-K ítem 2.02), régimen macro y de crédito
(FRED), estructura temporal y asimetría de la volatilidad (CBOE), fundamentales (XBRL de la SEC)…

`altprobe` solo comprueba qué fuentes responden desde el entorno de ejecución y con qué formato (el entorno de desarrollo no tiene acceso a ellas).
"""
from __future__ import annotations
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import Dict, Optional, Tuple

UA = 'vpscanner-research/1.0 (perezmerinodaniel1@gmail.com)'
BROWSER = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'


def fetch(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60, maxbytes: Optional[int] = None) -> Tuple[int, bytes, str]:
    """(estado HTTP, cuerpo, error). Nunca lanza: devuelve estado 0 y el motivo si falla la conexión."""
    h = {'User-Agent': UA, 'Accept-Encoding': 'identity'}
    h.update(headers or {})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
            return r.status, r.read(maxbytes) if maxbytes else r.read(), ''
    except urllib.error.HTTPError as e:
        try:
            body = e.read(300)
        except Exception:
            body = b''
        return e.code, body, f'HTTP {e.code} {e.reason}'
    except Exception as e:                                                   # DNS, TLS, tiempo agotado…
        return 0, b'', f'{type(e).__name__}: {e}'


def _show(label: str, url: str, **kw):
    t = time.time()
    st, body, err = fetch(url, **kw)
    snippet = body[:200].decode('utf-8', errors='replace').replace('\n', ' ⏎ ') if body and st == 200 else (err or '')
    print(f'  [{st:3d}] {label:42s} {len(body):>10,d} B · {time.time() - t:4.1f}s · {snippet}', flush=True)
    return st, body


def probe():
    print('\n######## SONDEO DE FUENTES DE DATOS ALTERNATIVOS', flush=True)
    sec = {'User-Agent': UA}
    print('\n  SEC EDGAR')
    st, b = _show('company_tickers.json', 'https://www.sec.gov/files/company_tickers.json', headers=sec)
    if st == 200:
        d = json.loads(b)
        print(f'        {len(d)} empresas · ejemplo {list(d.values())[:2]}')
    st, b = _show('submissions AAPL (CIK 320193)', 'https://data.sec.gov/submissions/CIK0000320193.json', headers=sec)
    if st == 200:
        j = json.loads(b)
        rec = j['filings']['recent']
        n8 = [(d, it) for f, d, it in zip(rec['form'], rec['filingDate'], rec['items']) if f == '8-K' and '2.02' in it]
        print(f'        recientes {len(rec["form"])} · archivos adicionales {len(j["filings"].get("files", []))} · 8-K con ítem 2.02: {len(n8)} · fechas {n8[:3]} … {n8[-2:]}')
    st, b = _show('frames us-gaap NetIncomeLoss CY2023Q4I?', 'https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2023Q3.json', headers=sec)
    if st == 200:
        j = json.loads(b)
        print(f'        {len(j["data"])} empresas · ejemplo {j["data"][:1]}')
    for q in ('2024q1', '2018q2'):
        st, b = _show(f'insider transactions {q}', f'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip', headers=sec, timeout=180)
        if st == 200:
            z = zipfile.ZipFile(io.BytesIO(b))
            print('        ficheros: ' + ', '.join(f'{i.filename} ({i.file_size // 1024} KB)' for i in z.infolist()))
            for nm in ('SUBMISSION.tsv', 'NONDERIV_TRANS.tsv', 'REPORTINGOWNER.tsv'):
                if nm in z.namelist():
                    head = z.read(nm)[:700].decode('utf-8', errors='replace').split('\n')[:2]
                    print(f'        {nm}: {head[0][:300]}\n          {head[1][:300] if len(head) > 1 else ""}')

    print('\n  FINRA')
    for d in ('20240102', '20180604', '20120601'):
        st, b = _show(f'Reg SHO volumen corto diario {d}', f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt')
        if st == 200:
            lines = b.decode('utf-8', errors='replace').split('\n')
            print(f'        {len(lines)} líneas · {lines[0]} · {lines[1]} · {lines[2]}')
    for u in ('https://cdn.finra.org/equity/otcmarket/biweekly/shrt20240131.csv', 'https://cdn.finra.org/equity/otcmarket/biweekly/shrt20240115.csv',
              'https://cdn.finra.org/equity/shortinterest/shrt20240131.csv'):
        _show('interés corto ' + u.rsplit('/', 1)[-1], u)

    print('\n  FRED (CSV sin clave)')
    for sid in ('BAMLH0A0HYM2', 'BAMLC0A0CM', 'T10Y2Y', 'T10Y3M', 'DFII10', 'DTWEXBGS', 'DCOILWTICO', 'NFCI', 'ICSA', 'VIXCLS', 'T5YIE', 'USEPUINDXD'):
        st, b = _show(sid, f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2015-01-01')
        if st == 200:
            ls = b.decode().strip().split('\n')
            print(f'        {len(ls) - 1} filas · {ls[1]} … {ls[-1]}')

    print('\n  CBOE')
    for nm in ('VIX', 'VIX9D', 'VIX3M', 'VIX6M', 'VVIX', 'SKEW', 'OVX', 'GVZ', 'VXN', 'RVX'):
        st, b = _show(f'{nm}_History.csv', f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{nm}_History.csv')
        if st == 200:
            ls = b.decode().strip().split('\n')
            print(f'        {len(ls) - 1} filas · {ls[0]} · {ls[1]} … {ls[-1]}')
    for u in ('https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv', 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypcarchive.csv',
              'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv'):
        st, b = _show(u.rsplit('/', 1)[-1], u)
        if st == 200:
            ls = b.decode(errors='replace').strip().split('\n')
            print(f'        {len(ls)} líneas · {ls[0][:100]} · {ls[1][:100]} … {ls[-1][:100]}')

    print('\n  OTRAS')
    _show('Nasdaq calendario de resultados 2024-01-30', 'https://api.nasdaq.com/api/calendar/earnings?date=2024-01-30', headers={'User-Agent': BROWSER, 'Accept': 'application/json'})
    _show('Yahoo quoteSummary (sin clave)', 'https://query1.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=calendarEvents,earnings', headers={'User-Agent': BROWSER})
    _show('Yahoo series de fundamentales', 'https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/AAPL?type=quarterlyNetIncome&period1=1577836800&period2=1735689600', headers={'User-Agent': BROWSER})
    _show('CNN miedo y codicia', 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2020-01-01', headers={'User-Agent': BROWSER, 'Origin': 'https://www.cnn.com', 'Referer': 'https://www.cnn.com/'})
    _show('AAII sentimiento', 'https://www.aaii.com/files/surveys/sentiment.xls', headers={'User-Agent': BROWSER})
    _show('Wikipedia visitas (Apple Inc.)', 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Apple_Inc./daily/20240101/20240131')
    _show('GDELT tono (Apple)', 'https://api.gdeltproject.org/api/v2/doc/doc?query=%22Apple%20Inc%22&mode=timelinetone&format=json&timespan=1m', timeout=40)
    _show('Treasury tipos diarios 2024', 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2024/all?type=daily_treasury_yield_curve&field_tdr_date_value=2024&page&_format=csv')
    sys.stdout.flush()


if __name__ == '__main__':
    probe()
