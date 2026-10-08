"""
data.py — Descarga de precios diarios (Yahoo Finance) con reintentos, paralelismo y caché.

Una "serie" es un dict de arrays numpy float64 (t en segundos UTC):
    {'t','o','h','l','c','v'}  — ordenadas ascendentes, sin NaN, SIN la barra del día en curso
Los OHLC se ajustan por dividendos con adjclose (retorno total) para que una barra
ex-dividendo no parezca un hueco bajista; la última barra siempre tiene factor 1.0,
así que su cierre coincide con el precio real.
"""
from __future__ import annotations
import gzip
import json
import os
import pickle
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Iterable

import numpy as np

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/124.0 Safari/537.36')
MIN_BARS = 120


def _get(url: str, timeout: int = 25) -> Optional[bytes]:
    """GET con reintentos; backoff largo ante 429/5xx."""
    for attempt in range(5):
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2.0 * (attempt + 1) + random.random() * 2)     # 429, 5xx...
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_yahoo(ysym: str, rng: str = '2y', now: Optional[float] = None) -> Optional[Dict[str, np.ndarray]]:
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ysym, safe="")}'
           f'?interval=1d&range={rng}&includeAdjustedClose=true&events=div%7Csplit')
    raw = _get(url)
    if not raw:
        return None
    try:
        res = json.loads(raw)['chart']['result'][0]
        ts = res.get('timestamp')
        if not ts:
            return None
        q = res['indicators']['quote'][0]
        adj = None
        try:
            adj = res['indicators']['adjclose'][0]['adjclose']
        except Exception:
            pass
        t = np.array(ts, dtype=np.float64)
        o = np.array([np.nan if x is None else x for x in q['open']], dtype=np.float64)
        h = np.array([np.nan if x is None else x for x in q['high']], dtype=np.float64)
        l = np.array([np.nan if x is None else x for x in q['low']], dtype=np.float64)
        c = np.array([np.nan if x is None else x for x in q['close']], dtype=np.float64)
        v = np.array([0.0 if x is None else x for x in q.get('volume', [0] * len(ts))], dtype=np.float64)
        if adj is not None and len(adj) == len(c):
            a = np.array([np.nan if x is None else x for x in adj], dtype=np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                f = a / c
            f = np.where(np.isfinite(f) & (f > 0), f, 1.0)
            o, h, l, c = o * f, h * f, l * f, c * f
        ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (c > 0) & (h >= l)
        t, o, h, l, c, v = t[ok], o[ok], h[ok], l[ok], c[ok], v[ok]
        if len(t) == 0:
            return None
        # ¿La última barra es la sesión de hoy aún sin cerrar? → fuera.
        reg = (res.get('meta') or {}).get('currentTradingPeriod', {}).get('regular', {})
        start, end = reg.get('start'), reg.get('end')
        now = time.time() if now is None else now
        if start and end and t[-1] >= start and now < end:
            t, o, h, l, c, v = t[:-1], o[:-1], h[:-1], l[:-1], c[:-1], v[:-1]
        if len(t) < MIN_BARS:
            return None
        # duplicados de timestamp (Yahoo a veces repite la última barra)
        keep = np.concatenate([[True], np.diff(t) > 0])
        return {'t': t[keep], 'o': o[keep], 'h': h[keep], 'l': l[keep], 'c': c[keep], 'v': v[keep]}
    except Exception:
        return None


def load_cache(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with gzip.open(path, 'wb', compresslevel=3) as f:
        pickle.dump(cache, f, protocol=4)


def fetch_many(symbols: Iterable[str], rng: str = '2y', workers: int = 8,
               cache_path: Optional[str] = None, max_age_h: float = 18.0,
               verbose: bool = True):
    """Descarga todas las series (reutilizando caché fresca). Devuelve ({sym: serie}, [fallidos])."""
    symbols = list(dict.fromkeys(symbols))
    cache = load_cache(cache_path) if cache_path else {}
    now = time.time()
    out: Dict[str, Dict[str, np.ndarray]] = {}
    todo = []
    for s in symbols:
        e = cache.get(s)
        if e and e.get('rng') == rng and (now - e['fetched']) < max_age_h * 3600:
            out[s] = e['bars']
        else:
            todo.append(s)
    if verbose:
        print(f'  datos: {len(out)} en caché, {len(todo)} por descargar ({rng})', flush=True)
    failed = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_yahoo, s, rng): s for s in todo}
        for fu in as_completed(futs):
            s = futs[fu]
            done += 1
            try:
                bars = fu.result()
            except Exception:
                bars = None
            if bars is None:
                failed.append(s)
            else:
                out[s] = bars
                cache[s] = {'rng': rng, 'fetched': now, 'bars': bars}
            if verbose and done % 200 == 0:
                print(f'  ... {done}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)
    if cache_path and todo:
        save_cache(cache_path, cache)
    if verbose:
        print(f'  datos: OK {len(out)} · fallidos {len(failed)}', flush=True)
        if failed:
            print('  fallidos:', ' '.join(failed[:60]) + (' …' if len(failed) > 60 else ''), flush=True)
    return out, failed


def fetch_intraday(ysym: str, interval: str = '60m', rng: str = '730d'):
    """Barras intradía (sesión regular). Devuelve dict de arrays + 'off' (desfase horario en segundos) o None."""
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ysym, safe="")}'
           f'?interval={interval}&range={rng}&includePrePost=false')
    raw = _get(url)
    if not raw:
        return None
    try:
        res = json.loads(raw)['chart']['result'][0]
        ts = res.get('timestamp')
        if not ts:
            return None
        q = res['indicators']['quote'][0]
        t = np.array(ts, dtype=np.float64)
        arr = {k: np.array([np.nan if x is None else x for x in q[k]], dtype=np.float64) for k in ('open', 'high', 'low', 'close')}
        v = np.array([0.0 if x is None else x for x in q.get('volume', [0] * len(ts))], dtype=np.float64)
        ok = np.isfinite(arr['open']) & np.isfinite(arr['high']) & np.isfinite(arr['low']) & np.isfinite(arr['close'])
        off = float((res.get('meta') or {}).get('gmtoffset', 0))
        return {'t': t[ok], 'o': arr['open'][ok], 'h': arr['high'][ok], 'l': arr['low'][ok], 'c': arr['close'][ok], 'v': v[ok], 'off': off}
    except Exception:
        return None


def fetch_intraday_many(symbols, workers: int = 8, **kw):
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_intraday, s, **kw): s for s in symbols}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception:
                r = None
            if r is not None:
                out[futs[fu]] = r
    return out
