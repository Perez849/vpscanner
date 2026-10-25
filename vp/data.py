"""Descarga de precios diarios de Yahoo Finance (API de gráficos). Precios con split aplicado pero SIN ajustar por dividendos, para que los niveles coincidan con los del bróker."""
from __future__ import annotations
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

HOSTS = ('query1.finance.yahoo.com', 'query2.finance.yahoo.com')          # se alternan en los reintentos
URL = 'https://{host}/v8/finance/chart/{sym}?{span}&interval=1d&includeAdjustedClose=true&events=div%7Csplit'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
Bars = Dict[str, np.ndarray]
ERRORS: Dict[str, str] = {}          # último motivo de fallo por valor (para el diagnóstico)
WAIT = 1.0                           # factor de las esperas (las pruebas lo ponen a 0)
_lock = threading.Lock()
_pause_until = 0.0                   # cuando Yahoo responde 429, todos los hilos esperan a la vez
_strikes = 0                         # 429 seguidos sin ninguna respuesta buena: con MAX_STRIKES se abandona (Yahoo está limitando esta IP)
MAX_STRIKES = 12


def _throttle(seconds: float) -> None:
    global _pause_until, _strikes
    with _lock:
        _strikes += 1
        _pause_until = max(_pause_until, time.time() + seconds * WAIT)


def _ok() -> None:
    global _strikes
    _strikes = 0


def throttled() -> bool:
    return _strikes >= MAX_STRIKES


def _wait_pause() -> None:
    while True:
        d = _pause_until - time.time()
        if d <= 0:
            return
        time.sleep(min(d, 2.0))


def parse(payload: Dict) -> Optional[Bars]:
    """Respuesta JSON de Yahoo → barras {t, o, h, l, c, v, name}. Descarta las filas incompletas. None si no hay datos."""
    try:
        res = payload['chart']['result'][0]
        ts = res['timestamp']
        q = res['indicators']['quote'][0]
        cols = [q['open'], q['high'], q['low'], q['close'], q['volume']]
    except (KeyError, IndexError, TypeError):
        return None
    arr = np.array([[np.nan if x is None else x for x in col] for col in cols], dtype=np.float64)
    t = np.array(ts, dtype=np.int64)
    ok = np.isfinite(arr).all(axis=0) & (arr[3] > 0) & (arr[1] >= arr[2])
    if ok.sum() < 5:
        return None
    t, arr = t[ok], arr[:, ok]
    order = np.argsort(t)
    t, arr = t[order], arr[:, order]
    keep = np.concatenate([[True], np.diff(t) > 0])                    # sin marcas repetidas
    meta = res.get('meta', {})
    b: Bars = {'t': t[keep], 'o': arr[0][keep], 'h': arr[1][keep], 'l': arr[2][keep], 'c': arr[3][keep], 'v': arr[4][keep]}
    b['name'] = meta.get('longName') or meta.get('shortName') or ''        # type: ignore[assignment]
    return b


def drop_open_bar(b: Bars, now: Optional[datetime] = None) -> Bars:
    """Quita la última barra si es la de hoy y la bolsa de Nueva York aún no ha cerrado (una barra incompleta falsearía las señales)."""
    from zoneinfo import ZoneInfo
    ny = ZoneInfo('America/New_York')
    now = (now or datetime.now(timezone.utc)).astimezone(ny)
    last = datetime.fromtimestamp(int(b['t'][-1]), timezone.utc).astimezone(ny)
    if last.date() == now.date() and (now.hour, now.minute) < (16, 20) and len(b['c']) > 1:
        return {k: (v[:-1] if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    return b


def span(rng: str, now: Optional[float] = None) -> str:
    """'3y' → fechas explícitas (con `range=10y` Yahoo puede devolver barras que no son diarias); otro valor ('6mo', '2y'…) se pasa tal cual."""
    if rng.endswith('y') and rng[:-1].isdigit():
        now = time.time() if now is None else now
        return f'period1={int(now - (int(rng[:-1]) * 365.25 + 6) * 86400)}&period2={int(now) + 86400}'
    return f'range={rng}'


def fetch_one(sym: str, rng: str = '3y', tries: int = 5, timeout: int = 25) -> Optional[Bars]:
    for k in range(tries):
        if throttled():
            ERRORS[sym] = 'HTTP 429 (se abandona: Yahoo limita las peticiones desde esta IP)'
            return None
        _wait_pause()
        url = URL.format(host=HOSTS[k % len(HOSTS)], sym=urllib.parse.quote(sym.replace('.', '-'), safe=''), span=span(rng))        # ^GSPC, EURUSD=X, GC=F… con la codificación de URL
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                b = parse(json.loads(r.read().decode('utf-8')))
            _ok()
            return drop_open_bar(b) if b is not None else None
        except urllib.error.HTTPError as e:
            ERRORS[sym] = f'HTTP {e.code}'
            if e.code == 429:
                _throttle(15 * (k + 1) + random.random() * 5)          # límite de Yahoo: pausa común y se alterna de servidor
            elif e.code < 500:                                          # 404, 400…: reintentar no sirve de nada
                return None
            else:
                time.sleep(2.0 * (k + 1) * WAIT)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as e:
            ERRORS[sym] = type(e).__name__
            time.sleep(1.5 * (k + 1) * WAIT)
    return None


def _cache_path(cache_dir: str, sym: str, rng: str) -> str:
    return os.path.join(cache_dir, f'{sym}_{rng}.npz')


def _load_cache(path: str, max_age_h: float) -> Optional[Bars]:
    try:
        if time.time() - os.path.getmtime(path) > max_age_h * 3600:
            return None
        z = np.load(path, allow_pickle=False)
        b: Bars = {k: z[k] for k in ('t', 'o', 'h', 'l', 'c', 'v')}
        b['name'] = str(z['name']) if 'name' in z else ''              # type: ignore[assignment]
        return b
    except (OSError, ValueError, KeyError):
        return None


def fetch_many(symbols: List[str], rng: str = '3y', workers: int = 6, cache_dir: Optional[str] = None, max_age_h: float = 6.0, verbose: bool = False) -> Tuple[Dict[str, Bars], List[str]]:
    """Descarga en paralelo (con caché en disco opcional). Devuelve (barras por valor, valores fallidos)."""
    out: Dict[str, Bars] = {}
    todo: List[str] = []
    for s in symbols:
        b = _load_cache(_cache_path(cache_dir, s, rng), max_age_h) if cache_dir else None
        if b is not None:
            out[s] = b
        else:
            todo.append(s)
    global _strikes
    _strikes = 0
    if verbose:
        print(f'  datos: {len(out)} en caché, {len(todo)} por descargar ({rng})', flush=True)

    def one(s: str) -> Tuple[str, Optional[Bars]]:
        time.sleep(random.random() * 0.3 * WAIT)
        return s, fetch_one(s, rng)

    failed: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s, b in ex.map(one, todo):
            if b is None:
                failed.append(s)
                continue
            out[s] = b
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(_cache_path(cache_dir, s, rng), **{k: v for k, v in b.items() if k != 'name'}, name=np.array(b.get('name', '')))
    if verbose:
        print(f'  datos: OK {len(out)} · fallidos {len(failed)}' + (f' ({" ".join(failed[:20])})' if failed else ''), flush=True)
        if failed:
            print('  motivos: ' + ' · '.join(f'{s}: {ERRORS.get(s, "sin datos")}' for s in failed[:5]), flush=True)
    return out, failed
