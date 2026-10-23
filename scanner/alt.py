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



# ═════════════════════════════════════════════════════════════════════════
#  Cargadores (cada uno devuelve estructuras de numpy; todo con fecha de publicación)
# ═════════════════════════════════════════════════════════════════════════
import csv
import datetime as dt
import gzip
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache', 'alt')
SEC_H = {'User-Agent': UA}
NASDAQ_H = {'User-Agent': BROWSER, 'Accept': 'application/json'}


def _epoch(d: dt.date) -> int:
    return (d - dt.date(1970, 1, 1)).days


def _cget(name: str):
    pth = os.path.join(CACHE, name + '.pkl.gz')
    if os.path.exists(pth):
        try:
            with gzip.open(pth, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _cput(name: str, obj) -> None:
    os.makedirs(CACHE, exist_ok=True)
    with gzip.open(os.path.join(CACHE, name + '.pkl.gz'), 'wb') as f:
        pickle.dump(obj, f, protocol=4)


class _Limiter:
    """Como mucho `rate` peticiones por segundo entre todos los hilos."""
    def __init__(self, rate: float):
        self.dt, self.lock, self.next = 1.0 / rate, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.dt
        if t > now:
            time.sleep(t - now)


def _retry(url: str, headers=None, tries: int = 4, timeout: int = 60, limiter: Optional[_Limiter] = None):
    st, body = 0, b''
    for k in range(tries):
        if limiter:
            limiter.wait()
        st, body, _ = fetch(url, headers=headers, timeout=timeout)
        if st == 200:
            return st, body
        if st in (403, 404):                      # no existe / denegado: no insistir
            return st, body
        time.sleep(min(2 ** k * 2, 30))           # 429, 5xx, conexión
    return st, body


def _daily(days: np.ndarray, vals: np.ndarray):
    ok = np.isfinite(vals)
    d, v = days[ok], vals[ok]
    o = np.argsort(d, kind='stable')
    return d[o], v[o]


def fred(sid: str):
    """(días, valores) de una serie diaria o semanal de FRED, sin clave."""
    c = _cget('fred_' + sid)
    if c is not None and time.time() - c['t'] < 24 * 3600:
        return c['d'], c['v']
    st, b = _retry(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2012-01-01', timeout=60)
    if st != 200:
        return (c['d'], c['v']) if c is not None else (np.array([], np.int64), np.array([]))
    d, v = [], []
    for ln in b.decode().strip().split('\n')[1:]:
        a = ln.split(',')
        if len(a) < 2:
            continue
        try:
            x = float(a[1])
        except ValueError:
            x = float('nan')
        d.append(_epoch(dt.date.fromisoformat(a[0]))); v.append(x)
    d, v = _daily(np.array(d, np.int64), np.array(v))
    _cput('fred_' + sid, {'t': time.time(), 'd': d, 'v': v})
    return d, v


def cboe(nm: str):
    """(días, cierre) de un índice de volatilidad del CBOE."""
    c = _cget('cboe_' + nm)
    if c is not None and time.time() - c['t'] < 24 * 3600:
        return c['d'], c['v']
    st, b = _retry(f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{nm}_History.csv', timeout=60)
    if st != 200:
        return (c['d'], c['v']) if c is not None else (np.array([], np.int64), np.array([]))
    rd = csv.reader(io.StringIO(b.decode()))
    hdr = [h.strip().upper() for h in next(rd)]
    col = hdr.index('CLOSE') if 'CLOSE' in hdr else 1
    d, v = [], []
    for a in rd:
        try:
            m, dd, yy = a[0].split('/')
            d.append(_epoch(dt.date(int(yy), int(m), int(dd)))); v.append(float(a[col]))
        except Exception:
            continue
    d, v = _daily(np.array(d, np.int64), np.array(v))
    _cput('cboe_' + nm, {'t': time.time(), 'd': d, 'v': v})
    return d, v


def aaii():
    """Encuesta semanal de AAII: (día de publicación, % alcistas, % bajistas) en fracción."""
    c = _cget('aaii')
    if c is not None and time.time() - c['t'] < 24 * 3600:
        return c['d'], c['bull'], c['bear']
    st, b = _retry('https://www.aaii.com/files/surveys/sentiment.xls', headers={'User-Agent': BROWSER}, timeout=90)
    empty = (np.array([], np.int64), np.array([]), np.array([]))
    if st != 200:
        return (c['d'], c['bull'], c['bear']) if c is not None else empty
    try:
        import pandas as pd
        x = pd.read_excel(io.BytesIO(b), sheet_name='SENTIMENT', header=None)
        d, bu, be = [], [], []
        for r in x.itertuples(index=False):
            a = list(r)
            if hasattr(a[0], 'year') and not isinstance(a[0], str):
                try:
                    bull, bear = float(a[1]), float(a[3])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(bull) and np.isfinite(bear):
                    d.append(_epoch(a[0].date() if hasattr(a[0], 'date') else a[0])); bu.append(bull); be.append(bear)
        o = np.argsort(d)
        res = (np.array(d, np.int64)[o], np.array(bu)[o], np.array(be)[o])
        _cput('aaii', {'t': time.time(), 'd': res[0], 'bull': res[1], 'bear': res[2]})
        return res
    except Exception as e:
        print(f'  AAII: no se pudo leer ({type(e).__name__}: {e})', flush=True)
        return empty


def _num(x) -> float:
    if x is None:
        return float('nan')
    t = str(x).strip().replace('$', '').replace(',', '').replace('%', '')
    if t in ('', 'N/A', 'NA', '--', 'null'):
        return float('nan')
    neg = t.startswith('(') and t.endswith(')')
    t = t.strip('()')
    try:
        v = float(t)
    except ValueError:
        return float('nan')
    return -v if neg else v


def nasdaq_earnings(start: dt.date, end: dt.date, workers: int = 6):
    """Calendario de resultados del Nasdaq: {fecha: [(símbolo, momento, eps, previsión, sorpresa %, capitalización, nº de analistas)]}. Caché incremental."""
    cache = _cget('nasdaq_earn') or {}
    todo = []
    d = start
    today = dt.date.today()
    while d <= end:
        if d.weekday() < 5 and (d not in cache or (today - d).days <= 3):          # las fechas recientes se rehacen (cambian las previsiones)
            todo.append(d)
        d += dt.timedelta(days=1)
    lim = _Limiter(4.0)
    done = [0]

    def one(day):
        st, b = _retry(f'https://api.nasdaq.com/api/calendar/earnings?date={day.isoformat()}', headers=NASDAQ_H, timeout=45, limiter=lim)
        done[0] += 1
        if done[0] % 300 == 0:
            print(f'    nasdaq: {done[0]}/{len(todo)} días', flush=True)
        if st != 200:
            return day, None
        try:
            rows = json.loads(b)['data']['rows'] or []
        except Exception:
            return day, None
        out = []
        for r in rows:
            out.append((str(r.get('symbol', '')).strip().upper().replace('/', '-').replace('.', '-'), str(r.get('time', '')), _num(r.get('eps')), _num(r.get('epsForecast')),
                        _num(r.get('surprise')), _num(r.get('marketCap')), _num(r.get('noOfEsts'))))
        return day, out

    if todo:
        print(f'  Nasdaq: {len(todo)} días por descargar', flush=True)
        fails = 0
        with ThreadPoolExecutor(workers) as ex:
            for day, out in ex.map(one, todo):
                if out is None:
                    fails += 1
                else:
                    cache[day] = out
        print(f'  Nasdaq: {len(cache)} días en caché · fallidos {fails}', flush=True)
        _cput('nasdaq_earn', cache)
    return cache


def finra_short_volume(start: dt.date, end: dt.date, symset: set, workers: int = 8):
    """Volumen corto diario (Reg SHO, consolidado): (días ordenados, símbolos, corto[días, símbolos], total[días, símbolos])."""
    cache = _cget('finra_sv') or {}
    todo, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in cache:
            todo.append(d)
        d += dt.timedelta(days=1)
    lim = _Limiter(12.0)

    def one(day):
        st, b = _retry(f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{day.strftime("%Y%m%d")}.txt', timeout=45, limiter=lim)
        if st != 200:
            return day, (st, None)
        rows = {}
        for ln in b.decode('utf-8', errors='ignore').split('\n')[1:]:
            a = ln.split('|')
            sym_ = a[1].replace('.', '-') if len(a) > 1 else ''
            if len(a) >= 5 and sym_ in symset:
                try:
                    rows[sym_] = (int(a[2]), int(a[4]))
                except ValueError:
                    pass
        return day, (200, rows)

    if todo:
        print(f'  FINRA volumen corto: {len(todo)} días por descargar', flush=True)
        miss = 0
        with ThreadPoolExecutor(workers) as ex:
            for day, (st, rows) in ex.map(one, todo):
                if st == 200:
                    cache[day] = rows
                elif st in (403, 404):
                    cache[day] = None                                   # festivo o fuera de la ventana publicada
                else:
                    miss += 1
        print(f'  FINRA volumen corto: {sum(1 for v in cache.values() if v)} días con datos · sin respuesta fiable {miss}', flush=True)
        _cput('finra_sv', cache)
    days = sorted(k for k, v in cache.items() if v)
    syms = sorted({s for k in days for s in cache[k]})
    si = {s: i for i, s in enumerate(syms)}
    SH = np.full((len(days), len(syms)), np.nan, np.float32)
    TV = np.full((len(days), len(syms)), np.nan, np.float32)
    for r, k in enumerate(days):
        for s, (a, b_) in cache[k].items():
            SH[r, si[s]], TV[r, si[s]] = a, b_
    return np.array([_epoch(k) for k in days], np.int64), syms, SH, TV


def finra_short_interest(start: dt.date, end: dt.date, symset: set, workers: int = 6):
    """Interés corto quincenal: {símbolo: (día de publicación, posición corta, volumen medio diario, días para cubrir, cambio vs anterior)} (publicación = liquidación + 11 días)."""
    cache = _cget('finra_si') or {}
    targets = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        mid = dt.date(y, m, 15)
        last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
        targets += [mid, last]
        y, m = y + (m == 12), m % 12 + 1
    targets = [t for t in targets if t < end - dt.timedelta(days=12)]
    todo = [t for t in targets if t not in cache]
    lim = _Limiter(8.0)

    def one(t):
        for back in range(0, 6):
            d = t - dt.timedelta(days=back)
            if d.weekday() >= 5:
                continue
            st, b = _retry(f'https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d.strftime("%Y%m%d")}.csv', timeout=60, limiter=lim, tries=2)
            if st == 200:
                rd = csv.reader(io.StringIO(b.decode('utf-8', errors='ignore')), delimiter='|', quoting=csv.QUOTE_NONE)
                hdr = next(rd)
                ix = {h: i for i, h in enumerate(hdr)}
                out = {}
                for a in rd:
                    try:
                        s = a[ix['symbolCode']].strip().upper().replace('.', '-')
                    except Exception:
                        continue
                    if s in symset:
                        out[s] = (_num(a[ix['currentShortPositionQuantity']]), _num(a[ix['averageDailyVolumeQuantity']]), _num(a[ix['daysToCoverQuantity']]), _num(a[ix['previousShortPositionQuantity']]))
                return t, (d, out)
        return t, None

    if todo:
        print(f'  FINRA interés corto: {len(todo)} quincenas por descargar', flush=True)
        with ThreadPoolExecutor(workers) as ex:
            for t, res in ex.map(one, todo):
                cache[t] = res
        _cput('finra_si', cache)
    per: Dict[str, list] = {}
    for t, res in sorted(cache.items()):
        if not res:
            continue
        d, out = res
        pub = _epoch(d) + 11
        for s, (cur, adv, dtc, prev) in out.items():
            per.setdefault(s, []).append((pub, cur, adv, dtc, prev))
    return {s: np.array(v, dtype=np.float64) for s, v in per.items()}


QUARTERS = [f'{y}q{q}' for y in range(2016, 2027) for q in range(1, 5) if (y, q) >= (2016, 4) and (y, q) <= (2026, 1)]
EXEC_RE = re.compile(r'(?i)chief|ceo|cfo|coo|president|chair|founder')


def _insider_quarter(q: str):
    """Agregado por (símbolo, día de presentación) de las compras (P) y ventas (S) en mercado abierto de un trimestre de Form 4."""
    c = _cget('ins_' + q)
    if c is not None:
        return c
    st, b = _retry(f'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip', headers=SEC_H, timeout=240, tries=3)
    if st != 200:
        return None
    import pandas as pd
    z = zipfile.ZipFile(io.BytesIO(b))
    rd = lambda nm, cols: pd.read_csv(z.open(nm), sep='\t', usecols=cols, dtype=str, low_memory=False, on_bad_lines='skip')
    sub = rd('SUBMISSION.tsv', ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE', 'ISSUERTRADINGSYMBOL'])
    sub = sub[sub['DOCUMENT_TYPE'] == '4'].dropna(subset=['ISSUERTRADINGSYMBOL', 'FILING_DATE'])
    tr = rd('NONDERIV_TRANS.tsv', ['ACCESSION_NUMBER', 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD'])
    tr = tr[tr['TRANS_CODE'].isin(['P', 'S'])]
    tr['usd'] = pd.to_numeric(tr['TRANS_SHARES'], errors='coerce') * pd.to_numeric(tr['TRANS_PRICEPERSHARE'], errors='coerce')
    tr = tr.dropna(subset=['usd'])
    own = rd('REPORTINGOWNER.tsv', ['ACCESSION_NUMBER', 'RPTOWNER_RELATIONSHIP', 'RPTOWNER_TITLE']).fillna('')
    own['exe'] = (own['RPTOWNER_RELATIONSHIP'].str.contains('Officer', case=False) & own['RPTOWNER_TITLE'].str.contains(EXEC_RE)).astype(int)
    own['dir'] = own['RPTOWNER_RELATIONSHIP'].str.contains('Director', case=False).astype(int)
    own = own.groupby('ACCESSION_NUMBER')[['exe', 'dir']].max().reset_index()
    m = tr.merge(sub, on='ACCESSION_NUMBER').merge(own, on='ACCESSION_NUMBER', how='left').fillna({'exe': 0, 'dir': 0})
    m['sym'] = m['ISSUERTRADINGSYMBOL'].str.upper().str.strip().str.replace('.', '-', regex=False)
    m['fd'] = pd.to_datetime(m['FILING_DATE'], format='%d-%b-%Y', errors='coerce')
    m = m.dropna(subset=['fd'])
    m['day'] = (m['fd'].values.astype('datetime64[D]').astype(np.int64))
    buy, sell = m['TRANS_CODE'] == 'P', m['TRANS_CODE'] == 'S'
    m['buy'] = np.where(buy, m['usd'], 0.0)
    m['sell'] = np.where(sell, m['usd'], 0.0)
    m['exebuy'] = np.where(buy, m['usd'] * m['exe'], 0.0)
    m['dirbuy'] = np.where(buy, m['usd'] * m['dir'], 0.0)
    m['nbuy'] = buy.astype(int)
    g = m.groupby(['sym', 'day']).agg(buy=('buy', 'sum'), sell=('sell', 'sum'), exebuy=('exebuy', 'sum'), dirbuy=('dirbuy', 'sum'), nbuy=('nbuy', 'sum'),
                                      nacc=('ACCESSION_NUMBER', 'nunique')).reset_index()
    out = {k: g[k].to_numpy() for k in g.columns}
    _cput('ins_' + q, out)
    return out


def insiders(workers: int = 3):
    """{símbolo: matriz [día de presentación, compras USD, ventas USD, compras de directivos USD, compras de consejeros USD, nº de compras]} de todos los trimestres."""
    parts, miss = [], []
    cached = {q: _cget('ins_' + q) for q in QUARTERS}
    todo = [q for q in QUARTERS if cached[q] is None]
    if todo:
        print(f'  SEC Form 4: {len(todo)} trimestres por descargar', flush=True)
        lim = _Limiter(5.0)
        with ThreadPoolExecutor(workers) as ex:
            for q, res in zip(todo, ex.map(lambda q: (lim.wait(), _insider_quarter(q))[1], todo)):
                cached[q] = res
    for q in QUARTERS:
        if cached[q] is None:
            miss.append(q)
        else:
            parts.append(cached[q])
    if miss:
        print(f'  SEC Form 4: trimestres sin datos {miss}', flush=True)
    per: Dict[str, list] = {}
    for p in parts:
        for i in range(len(p['sym'])):
            per.setdefault(p['sym'][i], []).append((p['day'][i], p['buy'][i], p['sell'][i], p['exebuy'][i], p['dirbuy'][i], p['nbuy'][i]))
    out = {}
    for s, v in per.items():
        a = np.array(v, dtype=np.float64)
        out[s] = a[np.argsort(a[:, 0], kind='stable')]
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Funciones puras de unión (probadas con datos sintéticos)
# ═════════════════════════════════════════════════════════════════════════
def asof_vals(dates: np.ndarray, vals: np.ndarray, q: np.ndarray, lag: int = 1, max_age: Optional[int] = None) -> np.ndarray:
    """Último valor con fecha ≤ q - lag (lag de publicación). NaN si no hay o si es más viejo que max_age días."""
    out = np.full(len(q), np.nan)
    if len(dates) == 0:
        return out
    j = np.searchsorted(dates, q - lag, side='right') - 1
    ok = j >= 0
    out[ok] = vals[j[ok]]
    if max_age is not None:
        old = ok & ((q - lag - dates[np.clip(j, 0, None)]) > max_age)
        out[old] = np.nan
    return out


def window_sum(days: np.ndarray, vals: np.ndarray, q: np.ndarray, width: int) -> np.ndarray:
    """Suma de vals con día en (q - width, q - 1] (estrictamente antes del evento)."""
    if len(days) == 0:
        return np.zeros(len(q))
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    lo = np.searchsorted(days, q - width, side='right')
    hi = np.searchsorted(days, q - 1, side='right')
    return cs[hi] - cs[lo]


def _rolling_series(d: np.ndarray, v: np.ndarray, k: int, fn: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    s = pd.Series(v)
    if fn == 'chg':
        r = (s / s.shift(k) - 1.0) * 100.0
    elif fn == 'diff':
        r = s - s.shift(k)
    elif fn == 'z':
        r = (s - s.rolling(k, min_periods=k // 2).mean()) / s.rolling(k, min_periods=k // 2).std()
    elif fn == 'rel':
        r = s.rolling(5, min_periods=3).mean() / s.rolling(k, min_periods=k // 2).mean()
    else:
        raise ValueError(fn)
    return d, r.to_numpy()


def market_block(day: np.ndarray, src: Optional[Dict] = None) -> Tuple[np.ndarray, List[str]]:
    """Variables de régimen que dependen solo del día: crédito, tipos, dólar, petróleo, condiciones financieras, estructura temporal de la volatilidad, sentimiento."""
    src = src or {}
    g = lambda kind, key: (src.get((kind, key)) if (kind, key) in src else (fred(key) if kind == 'fred' else cboe(key)))
    cols, names = [], []

    def add(nm, arr):
        names.append(nm); cols.append(arr)

    def lvl(kind, key, nm, lag=1, tf=None, k=0):
        d, v = g(kind, key)
        if tf:
            d, v = _rolling_series(d, v, k, tf)
        add(nm, asof_vals(d, v, day, lag, max_age=15))

    lvl('fred', 'BAA10Y', 'baa10y'); lvl('fred', 'BAA10Y', 'baa10y_d20', tf='diff', k=20)
    lvl('fred', 'AAA10Y', 'aaa10y')
    lvl('fred', 'T10Y2Y', 't10y2y'); lvl('fred', 'T10Y3M', 't10y3m')
    lvl('fred', 'DFII10', 'real10y'); lvl('fred', 'DFII10', 'real10y_d20', tf='diff', k=20)
    lvl('fred', 'T5YIE', 'be5y'); lvl('fred', 'T5YIE', 'be5y_d20', tf='diff', k=20)
    lvl('fred', 'DTWEXBGS', 'usd_chg20', tf='chg', k=20)
    lvl('fred', 'DCOILWTICO', 'oil_chg20', tf='chg', k=20)
    lvl('fred', 'DGS10', 'dgs10_d20', tf='diff', k=20)
    lvl('fred', 'USEPUINDXD', 'epu_rel60', tf='rel', k=60)
    lvl('fred', 'NFCI', 'nfci', lag=6); lvl('fred', 'NFCI', 'nfci_d4', lag=6, tf='diff', k=4)
    lvl('fred', 'STLFSI4', 'stlfsi', lag=6)
    lvl('fred', 'ICSA', 'icsa_chg13', lag=5, tf='chg', k=13)
    lvl('fred', 'WALCL', 'fedbs_chg13', lag=2, tf='chg', k=13)
    d_, v_ = g('fred', 'RRPONTSYD'); add('rrp_log', np.log1p(asof_vals(d_, np.clip(v_, 0, None), day, 1, 10)))
    # estructura temporal y asimetría de la volatilidad
    v1, v9, v3, v6 = g('cboe', 'VIX'), g('cboe', 'VIX9D'), g('cboe', 'VIX3M'), g('cboe', 'VIX6M')

    def ratio(a, b, nm):
        da, va = a; db, vb = b
        add(nm, asof_vals(da, va, day, 1, 10) / asof_vals(db, vb, day, 1, 10))

    ratio(v9, v1, 'vix9d_vix'); ratio(v1, v3, 'vix_vix3m'); ratio(v3, v6, 'vix3m_vix6m')
    for nm, key in (('vvix', 'VVIX'), ('skew', 'SKEW'), ('ovx', 'OVX'), ('gvz', 'GVZ')):
        lvl('cboe', key, nm)
    lvl('cboe', 'VVIX', 'vvix_z60', tf='z', k=60)
    lvl('cboe', 'SKEW', 'skew_z60', tf='z', k=60)
    ratio(g('cboe', 'VXN'), v1, 'vxn_vix'); ratio(g('cboe', 'RVX'), v1, 'rvx_vix')
    ad, ab, ae = src.get(('aaii', 'all')) if ('aaii', 'all') in src else aaii()
    if len(ad):
        spread = ab - ae
        add('aaii_spread', asof_vals(ad, spread, day, 1, 14))
        import pandas as pd
        add('aaii_spread_d4', asof_vals(ad, spread - pd.Series(spread).shift(4).to_numpy(), day, 1, 14))
    else:
        add('aaii_spread', np.full(len(day), np.nan)); add('aaii_spread_d4', np.full(len(day), np.nan))
    return np.stack(cols, 1).astype(np.float32), names


def _by_symbol(sym: np.ndarray):
    """(orden, cortes): índices de eventos agrupados por símbolo."""
    order = np.argsort(sym, kind='stable')
    cuts = np.flatnonzero(np.diff(sym[order])) + 1
    return np.split(order, cuts)


def earn_block(sym_names: List[str], sym: np.ndarray, day: np.ndarray, cal: Dict) -> Tuple[np.ndarray, List[str]]:
    """Calendario y sorpresa de resultados. Un informe es conocido desde el mismo día si es antes de la apertura y desde el siguiente si es tras el cierre o no consta."""
    per: Dict[str, list] = {}
    for dte, rows in cal.items():
        if not rows:
            continue
        e = _epoch(dte)
        for (s, tm, eps, fc, sur, mcap, nest) in rows:
            avail = e if 'pre' in tm else e + 1
            if not np.isfinite(sur) and np.isfinite(eps) and np.isfinite(fc) and abs(fc) > 1e-9:
                sur = (eps - fc) / abs(fc) * 100.0
            per.setdefault(s, []).append((avail, e, sur, mcap, nest, eps, fc))
    names = ['earn_since', 'earn_next', 'earn_react3', 'earn_sur', 'earn_sur_pos', 'earn_sur_neg', 'earn_nest', 'earn_eps_neg']
    M = np.full((len(day), len(names)), np.nan, dtype=np.float32)
    for grp in _by_symbol(sym):
        if not len(grp):
            continue
        a = per.get(sym_names[sym[grp[0]]])
        if not a:
            continue
        arr = np.array(sorted(a), dtype=np.float64)
        av = arr[:, 0]
        q = day[grp]
        j = np.searchsorted(av, q, side='right') - 1
        ok = j >= 0
        if not ok.any():
            continue
        jj = np.clip(j, 0, None)
        since = q - arr[jj, 1]                                            # días desde la fecha del informe
        valid = ok & (since <= 200)
        r = grp[valid]
        M[r, 0] = since[valid]
        M[r, 1] = 91.0 - since[valid]                                     # días hasta el siguiente informe previsto (cadencia trimestral)
        M[r, 2] = (since[valid] <= 3).astype(np.float32)
        sur = arr[jj, 2][valid]
        M[r, 3] = np.clip(sur, -100, 100)
        M[r, 4] = np.where(np.isfinite(sur), (sur > 0).astype(np.float32), np.nan)
        M[r, 5] = np.where(np.isfinite(sur), (sur < -5).astype(np.float32), np.nan)
        M[r, 6] = arr[jj, 4][valid]
        M[r, 7] = np.where(np.isfinite(arr[jj, 5][valid]), (arr[jj, 5][valid] < 0).astype(np.float32), np.nan)      # (la capitalización del Nasdaq es la ACTUAL, no la de entonces: se descarta)
    return M, names


def insider_block(sym_names: List[str], sym: np.ndarray, day: np.ndarray, ins: Dict, last_day: int) -> Tuple[np.ndarray, List[str]]:
    names = ['ins_buy90', 'ins_buy30', 'ins_exe90', 'ins_dir90', 'ins_nbuy90', 'ins_sell90', 'ins_sellratio90', 'ins_any_buy90']
    M = np.full((len(day), len(names)), np.nan, dtype=np.float32)
    for grp in _by_symbol(sym):
        if not len(grp):
            continue
        a = ins.get(sym_names[sym[grp[0]]])
        q = day[grp]
        if a is None:
            z = np.zeros(len(q))
            buy90 = buy30 = exe = dire = nb = sell = z
        else:
            d = a[:, 0].astype(np.int64)
            buy90, buy30 = window_sum(d, a[:, 1], q, 90), window_sum(d, a[:, 1], q, 30)
            exe, dire = window_sum(d, a[:, 3], q, 90), window_sum(d, a[:, 4], q, 90)
            nb, sell = window_sum(d, a[:, 5], q, 90), window_sum(d, a[:, 2], q, 90)
        tot = buy90 + sell
        M[grp, 0], M[grp, 1] = np.log1p(buy90), np.log1p(buy30)
        M[grp, 2], M[grp, 3] = np.log1p(exe), np.log1p(dire)
        M[grp, 4] = nb
        M[grp, 5] = np.log1p(sell)
        M[grp, 6] = np.where(tot > 0, (buy90 - sell) / np.where(tot > 0, tot, 1), 0.0)
        M[grp, 7] = (buy90 > 0).astype(np.float32)
    M[day > last_day] = np.nan                                           # la ventana de 90 días llega a trimestres que aún no están publicados
    return M, names


def short_block(sym_names: List[str], sym: np.ndarray, day: np.ndarray, sv, si: Dict) -> Tuple[np.ndarray, List[str]]:
    names = ['sv1', 'sv5', 'sv5_rel60', 'sv_chg5', 'si_dtc', 'si_pct_adv', 'si_chg', 'si_logshares']
    M = np.full((len(day), len(names)), np.nan, dtype=np.float32)
    sd, ssyms, SH, TV = sv
    if len(sd):
        col = {s: i for i, s in enumerate(ssyms)}
        zs, zt = np.nan_to_num(SH).astype(np.float64).cumsum(0), np.nan_to_num(TV).astype(np.float64).cumsum(0)
        cnt = np.isfinite(TV).astype(np.int32).cumsum(0)
        j = np.searchsorted(sd, day - 1, side='right') - 1                # el archivo del día t se publica tras el cierre
        def win(c, k, jj, cj):
            lo = np.clip(jj - k, -1, None)
            top = c[jj, cj]
            base = np.where(lo >= 0, c[np.clip(lo, 0, None), cj], 0.0)
            return top - base
        for grp in _by_symbol(sym):
            if not len(grp):
                continue
            cj = col.get(sym_names[sym[grp[0]]])
            if cj is None:
                continue
            jj = j[grp]
            ok = (jj >= 5) & (day[grp] - sd[np.clip(jj, 0, None)] <= 6)
            if not ok.any():
                continue
            g_ = grp[ok]; jx = jj[ok]; cc = np.full(len(jx), cj)
            tv1, sh1 = TV[jx, cj], SH[jx, cj]
            sh5, tv5 = win(zs, 5, jx, cc), win(zt, 5, jx, cc)
            sh60, tv60 = win(zs, 60, jx, cc), win(zt, 60, jx, cc)
            n60 = win(cnt, 60, jx, cc)
            with np.errstate(invalid='ignore', divide='ignore'):
                r1, r5, r60 = sh1 / tv1, sh5 / tv5, sh60 / tv60
                r5b = np.where(jx >= 10, win(zs, 5, jx - 5, cc) / win(zt, 5, jx - 5, cc), np.nan)
            M[g_, 0], M[g_, 1] = r1, r5
            M[g_, 2] = np.where(n60 >= 40, r5 / r60, np.nan)
            M[g_, 3] = r5 - r5b
    for grp in _by_symbol(sym):
        if not len(grp):
            continue
        a = si.get(sym_names[sym[grp[0]]])
        if a is None or not len(a):
            continue
        pub = a[:, 0]
        q = day[grp]
        j = np.searchsorted(pub, q, side='right') - 1
        ok = (j >= 0) & ((q - pub[np.clip(j, 0, None)]) <= 45)
        jj = np.clip(j, 0, None)
        with np.errstate(invalid='ignore', divide='ignore'):
            M[grp[ok], 4] = a[jj, 3][ok]
            M[grp[ok], 5] = (a[jj, 1] / a[jj, 2])[ok]
            M[grp[ok], 6] = (a[jj, 1] / a[jj, 4] - 1.0)[ok]
            M[grp[ok], 7] = np.log1p(np.clip(a[jj, 1], 0, None))[ok]
    return M, names


def size_block(data: Dict, EV: Dict, rows: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Tamaño y liquidez con los propios precios (valen también para ETF y cripto): dólares negociados, precio y volumen relativo a su mediana anual."""
    import feats
    import pandas as pd
    sym, bi = EV['sym'][rows], EV['bi'][rows]
    names = ['logdvol20', 'logprice', 'dvol_rel250']
    M = np.full((len(rows), len(names)), np.nan, dtype=np.float32)
    for grp in _by_symbol(sym):
        if not len(grp):
            continue
        F = feats.build(data[EV['sym_names'][sym[grp[0]]]])
        i = bi[grp].astype(np.int64)
        dv = F['dvol20']
        med = pd.Series(dv).rolling(250, min_periods=120).median().to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            M[grp, 0] = np.log10(np.where(dv[i] > 0, dv[i], np.nan))
            M[grp, 1] = np.log10(np.where(F['c'][i] > 0, F['c'][i], np.nan))
            M[grp, 2] = dv[i] / med[i]
    return M, names


FAMILY = {}


def build_features(data: Dict, EV: Dict, rows: np.ndarray, src: Optional[Dict] = None):
    """Matriz [candidatos × variables alternativas], nombres y familia de cada una. `src` permite inyectar fuentes ya cargadas (pruebas)."""
    sym, day = EV['sym'][rows], EV['day'][rows].astype(np.int64)
    sn = EV['sym_names']
    symset = set(sn)
    t0 = time.time()
    if src is None:
        src = {}
    first_day, last_day = int(day.min()), int(day.max())
    d0 = dt.date(1970, 1, 1) + dt.timedelta(days=first_day - 200)
    d1 = dt.date(1970, 1, 1) + dt.timedelta(days=last_day)
    mk, mnames = market_block(day, src)
    print(f'  [alt] mercado/volatilidad: {len(mnames)} variables · {time.time() - t0:.0f}s', flush=True)
    cal = src['nasdaq'] if 'nasdaq' in src else nasdaq_earnings(d0, d1)
    ek, enames = earn_block(sn, sym, day, cal)
    print(f'  [alt] resultados: {len(cal)} días de calendario · cobertura {np.isfinite(ek[:, 0]).mean() * 100:.0f}% · {time.time() - t0:.0f}s', flush=True)
    ins = src['insiders'] if 'insiders' in src else insiders()
    last_ins = _epoch(dt.date(2026, 3, 31)) - 5 if 'insiders' not in src else last_day + 1000
    ik, inames = insider_block(sn, sym, day, ins, last_ins)
    print(f'  [alt] directivos: {len(ins)} símbolos con operaciones · con alguna compra en 90 días {np.nanmean(ik[:, 7]) * 100:.1f}% · {time.time() - t0:.0f}s', flush=True)
    sv = src['short_volume'] if 'short_volume' in src else finra_short_volume(max(d0, dt.date(2018, 11, 1)), d1, symset)
    si = src['short_interest'] if 'short_interest' in src else finra_short_interest(max(d0, dt.date(2017, 12, 1)), d1, symset)
    sk, snames = short_block(sn, sym, day, sv, si)
    print(f'  [alt] corto: volumen {len(sv[0])} días, interés {len(si)} símbolos · cobertura vol {np.isfinite(sk[:, 0]).mean() * 100:.0f}% / interés {np.isfinite(sk[:, 4]).mean() * 100:.0f}% · {time.time() - t0:.0f}s', flush=True)
    zk, znames = size_block(data, EV, rows) if data is not None else (np.full((len(day), 3), np.nan, np.float32), ['logdvol20', 'logprice', 'dvol_rel250'])
    M = np.concatenate([mk, ek, ik, sk, zk], 1)
    names = mnames + enames + inames + snames + znames
    fam = ['macro'] * len(mnames) + ['resultados'] * len(enames) + ['directivos'] * len(inames) + ['corto'] * len(snames) + ['tamaño'] * len(znames)
    return M, names, fam


def _grp(label: str, inn: np.ndarray, out: np.ndarray, pnl: np.ndarray, day: np.ndarray, sym: np.ndarray, year: np.ndarray) -> None:
    """Rentabilidad del grupo frente al resto del fondo (mismo periodo y mismas condiciones de elegibilidad), con t por clústeres de día y consistencia por años."""
    import research as RS
    if inn.sum() < 100 or out.sum() < 100:
        print(f'     {label:62s} n={int(inn.sum()):6d} (muy pocas)')
        return
    a, b = RS.stats(pnl[inn], day[inn], sym[inn]), RS.stats(pnl[out], day[out], sym[out])
    dif = a['mean'] - b['mean']
    se = (a['se'] ** 2 + b['se'] ** 2) ** 0.5
    pos = tot = 0
    for Y in sorted(set(year[inn])):
        mi, mo = inn & (year == Y), out & (year == Y)
        if mi.sum() >= 30 and mo.sum() >= 30:
            tot += 1
            pos += pnl[mi].mean() > pnl[mo].mean()
    print(f'     {label:62s} n={a["n"]:6d}  acierto {a["wr"]:5.1f}% vs {b["wr"]:5.1f}% · media {a["mean"]:+6.2f}% vs {b["mean"]:+6.2f}% · dif {dif:+5.2f} (t={dif / (se + 1e-12):+4.1f}) · años mejor {pos}/{tot}', flush=True)


def market_shift_test(M: np.ndarray, cols: List[str], fam: List[str], elig: np.ndarray, day: np.ndarray, pnl: np.ndarray, year: np.ndarray, reps: int = 60, top: int = 12):
    """Las variables de mercado son lentas (meses): permutar los resultados día a día no rompe su autocorrelación y deja el umbral demasiado bajo. Aquí el nulo se obtiene
    desplazando la serie diaria de la variable respecto a los resultados (desfase aleatorio de varios meses), que conserva la autocorrelación de ambas."""
    import osc
    idx = [j for j, f in enumerate(fam) if f == 'macro']
    d_e, y_e, yr_e = day[elig], pnl[elig], year[elig]
    u, inv = np.unique(d_e, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(u))
    fd = {}
    for j in idx:
        x = M[elig, j].astype(np.float64)
        ok = np.isfinite(x)
        n_ok = np.bincount(inv, weights=ok, minlength=len(u))
        v = np.bincount(inv, weights=np.where(ok, x, 0.0), minlength=len(u)) / np.maximum(n_ok, 1)
        v[n_ok == 0] = np.nan
        fd[j] = v

    def tstat(v):
        xe = v[inv]
        mk = np.isfinite(xe)
        if mk.sum() < 2000 or np.nanstd(xe[mk]) == 0:
            return np.nan
        return osc._slope(osc._rank_within_year(xe, yr_e, mk), y_e, inv, mk.astype(np.float64))[1]

    real = {cols[j]: tstat(fd[j]) for j in idx}
    rng = np.random.default_rng(23)
    mx = []
    for r in range(reps):
        sh = int(rng.integers(40, max(len(u) - 40, 41)))
        mx.append(np.nanmax([abs(tstat(np.roll(fd[j], sh))) for j in idx]))
    thr95, thr99 = float(np.quantile(mx, 0.95)), float(np.quantile(mx, 0.99))
    print('\n  A2. VARIABLES DE MERCADO CON EL NULO DE DESPLAZAMIENTO TEMPORAL (conserva la autocorrelación de la variable y de los resultados)')
    print(f'     el mejor |t| de {len(idx)} variables de mercado supera {thr95:.1f} el 5 % de las veces y {thr99:.1f} el 1 % ({reps} desfases aleatorios de ≥ 40 días con eventos)')
    top_ = sorted(real, key=lambda k: -abs(real[k]) if np.isfinite(real[k]) else 0)[:top]
    print('     ' + ' · '.join(f"{k}{'★' if abs(real[k]) >= thr95 else ''} t={real[k]:+.1f}" for k in top_))
    surv = [k for k in real if np.isfinite(real[k]) and abs(real[k]) >= thr95]
    print(f'     superan el umbral con el nulo temporal: {len(surv)} ({", ".join(surv) if surv else "ninguna"})', flush=True)
    return surv


def size_table(M: np.ndarray, cols: List[str], elig: np.ndarray, day: np.ndarray, sym: np.ndarray, gid: np.ndarray, pnl: np.ndarray, year: np.ndarray) -> None:
    import pandas as pd
    import research as RS
    print('\n  D. EFECTO TAMAÑO: rebote por quintil de capitalización (dentro de cada año y grupo)')
    for nm, lab in (('logdvol20', 'dólares negociados al día (precio × volumen)'),):
        x = M[:, cols.index(nm)].astype(np.float64)
        for gname in ('todas', 'us_large', 'us_mid'):
            m = elig & np.isfinite(x)
            if gname != 'todas':
                m &= gid == RS.GROUPS.index(gname)
            if m.sum() < 1000:
                continue
            df = pd.DataFrame({'x': x[m], 'g': gid[m], 'y': year[m]})
            q = (df.groupby(['g', 'y'])['x'].rank(pct=True, method='first').to_numpy() * 5).clip(0, 4.999).astype(int)
            rowsq = []
            for k in range(5):
                mk = q == k
                st = RS.stats(pnl[m][mk], day[m][mk], sym[m][mk])
                rowsq.append(f"Q{k + 1}: {st['wr']:.1f}% / {st['mean']:+.2f}%")
            print(f'     {lab[:42]:42s} · {gname:8s} n={int(m.sum()):6d} · ' + ' · '.join(rowsq), flush=True)


def run(uni: Dict, data: Dict, args, out_dir: str):
    import osc
    import pit
    import research as RS
    import setups as SU
    import simulate as SM
    import stops as ST
    uni2, data2, row_ok, info = pit.prepare(uni, data, args)
    EV = RS.build_events(data2, uni2, SU.SETUPS, SM.default_variants())
    rows, FL, names = RS.union_events(EV, +1)
    k = row_ok(EV, rows)
    rows, FL = rows[k], FL[k]
    last_day = int(EV['day'].max())
    day, sym, gid = EV['day'][rows], EV['sym'][rows], EV['grp'][rows]
    year = RS.year_of(day)
    print(f'\n######## DIMENSIONES MÁS ALLÁ DE PRECIO Y VOLUMEN · {len(rows):,} candidatos · datos hasta {np.datetime64("1970-01-01") + np.timedelta64(last_day, "D")}', flush=True)
    exp: Dict = {}
    osc._quiet(RS.build_plan, EV, rows, FL, names, ST.plan_of(4.0), last_day, spy=data2['SPY'], export=exp)
    allowed = exp['allowed']
    M, cols, fam = build_features(data2, EV, rows)
    cov = {f: float(np.isfinite(M[:, [j for j, ff in enumerate(fam) if ff == f]]).mean() * 100) for f in dict.fromkeys(fam)}
    print('  cobertura media por familia: ' + ' · '.join(f'{f} {v:.0f}%' for f, v in cov.items()) + f' · {len(cols)} variables', flush=True)
    vi = EV['variants'].index(ST.vname(4.0))
    pnl = EV['pnl'][rows, vi]
    atrp = EV['X'][rows][:, RS.FEATURES.index('atrp')]
    elig = np.isfinite(pnl) & (day <= last_day) & np.isfinite(atrp) & (atrp >= RS.MIN_ATR_PCT) & (atrp < RS.MAX_ATR_PCT) & np.isin(gid, allowed)
    print(f'  fondo del cribado: {int(elig.sum()):,} candidatos elegibles · media {pnl[elig].mean():+.2f}% · acierto {np.mean(pnl[elig] > 0) * 100:.1f}%', flush=True)
    print('\n  A.', end='')
    sc = osc.screen(M, cols, elig, day, pnl, year, noun='variables alternativas', fam=fam, top=30)
    market_shift_test(M, cols, fam, elig, day, pnl, year)
    size_table(M, cols, elig, day, sym, gid, pnl, year)

    # ── subgrupos con significado económico (frente al resto del fondo)
    c = {nm: M[:, j] for j, nm in enumerate(cols)}
    fin = lambda x: np.isfinite(x)
    print('\n  B. SUBGRUPOS (frente al resto del fondo elegible con dato)')
    z = lambda x: np.nan_to_num(x, nan=-9e9)
    g = lambda lab, inn, known: _grp(lab, elig & inn & known, elig & ~inn & known, pnl, day, sym, year)
    k = fin(c['earn_since'])
    g('resultados hace ≤3 días y sorpresa positiva (caída tras buenos resultados)', (c['earn_react3'] == 1) & (z(c['earn_sur']) > 0), k)
    g('resultados hace ≤3 días y sorpresa negativa (< −5 %)', (c['earn_react3'] == 1) & (c['earn_sur_neg'] == 1), k)
    g('resultados hace ≤3 días (cualquiera)', c['earn_react3'] == 1, k)
    g('resultados previstos en los próximos 14 días', (c['earn_next'] >= 0) & (c['earn_next'] <= 14), k)
    g('sin resultados en los últimos 10 días ni en los próximos 14', (c['earn_since'] > 10) & (c['earn_next'] > 14), k)
    k = fin(c['ins_buy90'])
    g('hay alguna compra de directivos en 90 días', c['ins_any_buy90'] == 1, k)
    g('compra de directivos ejecutivos (CEO/CFO/presidente) en 90 días', z(c['ins_exe90']) > 0, k)
    g('al menos 2 compras de directivos en 90 días (racimo)', z(c['ins_nbuy90']) >= 2, k)
    g('ventas de directivos en 90 días y ninguna compra', (z(c['ins_sell90']) > 0) & (c['ins_any_buy90'] == 0), k)
    k = fin(c['sv5_rel60'])
    g('volumen corto de 5 días ≥ 1,2 × su media de 60 días', z(c['sv5_rel60']) >= 1.2, k)
    g('volumen corto de 5 días ≤ 0,9 × su media de 60 días', c['sv5_rel60'] <= 0.9, k)
    g('proporción de venta en corto del día anterior ≥ 60 %', z(c['sv1']) >= 0.6, fin(c['sv1']))
    k = fin(c['si_dtc'])
    g('días para cubrir el interés corto ≥ 5', z(c['si_dtc']) >= 5, k)
    g('interés corto subiendo ≥ 10 % en la quincena', z(c['si_chg']) >= 0.10, fin(c['si_chg']))
    k = fin(c['vix_vix3m'])
    g('VIX/VIX3M > 1 (curva de volatilidad invertida, estrés agudo)', z(c['vix_vix3m']) > 1.0, k)
    g('VVIX en el 20 % más alto del año', c['vvix_z60'] > 1.0, fin(c['vvix_z60']))
    g('diferencial de crédito Baa−10a subiendo en 20 días (> +0,15)', z(c['baa10y_d20']) > 0.15, fin(c['baa10y_d20']))
    g('condiciones financieras (NFCI) más estrictas que su media (> 0)', z(c['nfci']) > 0, fin(c['nfci']))
    g('AAII: alcistas − bajistas < −20 puntos (pesimismo extremo)', c['aaii_spread'] < -0.20, fin(c['aaii_spread']))
    g('AAII: alcistas − bajistas > +20 puntos (optimismo extremo)', z(c['aaii_spread']) > 0.20, fin(c['aaii_spread']))

    # ── modelo ampliado
    famc = {f: [nm for nm, ff in zip(cols, fam) if ff == f] for f in dict.fromkeys(fam)}
    pre_sel = osc.pre_selection(sc, cols)
    arms = [('modelo actual (28 variables)', [])]
    for f, lst in famc.items():
        arms.append((f'+ {f} ({len(lst)})', lst))
    arms.append((f'+ todas las alternativas ({len(cols)})', list(cols)))
    arms.append((f'+ 8 mejores con datos ≤2018: {",".join(pre_sel[:2])}…', pre_sel))
    arms.append(('+ las que superan el umbral del mejor de N', sc['sig']))
    osc.model_arms(EV, rows, FL, names, M, cols, arms, last_day, data2['SPY'], 'C. SISTEMA ENTERO REENTRENADO CON LAS VARIABLES ALTERNATIVAS (modelo, política y cartera recalculados; AUC fuera de muestra)')
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
    probe2()
