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


def probe2():
    print('\n######## SONDEO 2: PROFUNDIDAD HISTÓRICA', flush=True)
    print('\n  FINRA volumen corto diario: ¿hasta qué fecha hacia atrás?')
    for d in ('20260102', '20250102', '20240801', '20230103', '20220103', '20210104', '20200102', '20190102', '20181231', '20181102', '20180102', '20170103', '20160104'):
        st, b, _ = fetch(f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt', timeout=30)
        print(f'  [{st:3d}] CNMSshvol{d}  {len(b):>9,d} B' + ('' if st == 200 else f'  {b[:90]!r}'), flush=True)
    st, b, _ = fetch('https://cdn.finra.org/equity/regsho/daily/CNMSshvol20230103.txt', timeout=30)
    for pre in ('FNQC', 'FNSQ', 'FNYX', 'FORF'):
        st2, b2, _ = fetch(f'https://cdn.finra.org/equity/regsho/daily/{pre}shvol20230103.txt', timeout=30)
        print(f'  [{st2:3d}] {pre}shvol20230103  {len(b2):>9,d} B', flush=True)
    print('\n  FINRA interés corto (quincenal): ¿fechas de liquidación?')
    for d in ('20240112', '20240131', '20240215', '20231229', '20231215', '20230731', '20230714', '20220531', '20220513', '20210430', '20200131', '20190131', '20180629', '20171229', '20170131'):
        st, b, _ = fetch(f'https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d}.csv', timeout=30)
        print(f'  [{st:3d}] shrt{d}  {len(b):>10,d} B' + ('' if st == 200 else f'  {b[:80]!r}'), flush=True)
    print('\n  Nasdaq calendario de resultados: ¿hasta cuándo hacia atrás?')
    for d in ('2016-11-01', '2017-05-02', '2018-05-01', '2019-02-05', '2020-05-05', '2022-05-03', '2025-08-05', '2026-10-22'):
        st, b, _ = fetch(f'https://api.nasdaq.com/api/calendar/earnings?date={d}', headers={'User-Agent': BROWSER, 'Accept': 'application/json'}, timeout=30)
        n = -1
        try:
            n = len(json.loads(b)['data']['rows'] or [])
        except Exception:
            pass
        print(f'  [{st:3d}] {d}  {len(b):>8,d} B · filas {n}', flush=True)
    st, b, _ = fetch('https://api.nasdaq.com/api/calendar/earnings?date=2023-05-02', headers={'User-Agent': BROWSER, 'Accept': 'application/json'}, timeout=30)
    try:
        print('        ejemplo: ' + json.dumps(json.loads(b)['data']['rows'][:2], ensure_ascii=False)[:600])
    except Exception:
        pass
    print('\n  SEC: cobertura de 8-K 2.02 y archivos adicionales')
    sec = {'User-Agent': UA}
    for cik, nm in ((320193, 'AAPL'), (789019, 'MSFT'), (1318605, 'TSLA'), (1090727, 'UPS'), (37996, 'F')):
        st, b, _ = fetch(f'https://data.sec.gov/submissions/CIK{cik:010d}.json', headers=sec, timeout=30)
        if st != 200:
            print(f'  [{st}] {nm}')
            continue
        j = json.loads(b)
        rec = j['filings']['recent']
        d22 = sorted(d for f, d, it in zip(rec['form'], rec['filingDate'], rec['items']) if f == '8-K' and '2.02' in it)
        print(f'  [{st}] {nm}: recientes {len(rec["form"])} desde {min(rec["filingDate"])} · 8-K 2.02: {len(d22)} desde {d22[0] if d22 else "-"} · archivos extra {[f["name"] for f in j["filings"].get("files", [])]}', flush=True)
    print('\n  SEC: trimestres de operaciones de directivos disponibles')
    for q in ('2016q4', '2017q1', '2019q3', '2022q2', '2025q4', '2026q1', '2026q2', '2026q3'):
        st, b, _ = fetch(f'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip', headers=sec, timeout=120, maxbytes=2000)
        print(f'  [{st:3d}] {q}', flush=True)
    print('\n  FRED: otras series')
    for sid in ('BAA10Y', 'AAA10Y', 'DGS10', 'DGS2', 'T10YIE', 'DEXUSEU', 'WALCL', 'RRPONTSYD', 'STLFSI4', 'DCOILBRENTEU', 'GOLDAMGBD228NLBM', 'UMCSENT', 'USREC', 'USEPUINDXD', 'EMVOVERALLEMV'):
        st, b, _ = fetch(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2015-01-01', timeout=30)
        if st == 200:
            ls = b.decode().strip().split('\n')
            print(f'  [{st}] {sid:18s} {len(ls) - 1} filas · {ls[1]} … {ls[-1]}', flush=True)
        else:
            print(f'  [{st}] {sid}', flush=True)
    print('\n  AAII: formato de la hoja')
    st, b, _ = fetch('https://www.aaii.com/files/surveys/sentiment.xls', headers={'User-Agent': BROWSER}, timeout=60)
    try:
        import pandas as pd
        x = pd.read_excel(io.BytesIO(b), sheet_name=None, header=None)
        for k, v in x.items():
            print(f'        hoja {k}: {v.shape} · primeras filas {v.head(6).values.tolist()[:6]}')
    except Exception as e:
        print(f'        no se pudo leer: {type(e).__name__}: {e}')
    sys.stdout.flush()


if __name__ == '__main__':
    probe()
    probe2()
