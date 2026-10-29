"""
Cribador diario de la regla de impulso con caída (parte 2 del laboratorio, separada del chartismo), con la escalera del operador y disciplina de gestión.

Aplica al ÚLTIMO cierre del universo (S&P 500 + Nasdaq-100, las 300 más negociadas ese día) las reglas de entrada de `search.ENTRADAS` (las mismas entradas que las candidatas confirmadas, con nombre claro; la salida es siempre la escalera), con exactamente las mismas condiciones y los mismos
percentiles del día que usa la búsqueda y su confirmación (`search.conditions`): una acción cumple una regla si hoy está en el 20 % mejor del universo por impulso a 12 meses (sin el último mes)
Y ha caído (bajo su media de 200 sesiones, RSI(2) ≤ 10 o 2 o más señales bajistas a la vez).

Para cada acción que cumple alguna regla da el plan de la escalera: stop inicial a 2,5 ATR (y la alternativa al 6 % del precio), nivel donde se asegura +4 % (stop plano) y +10 % (stop que
persigue a 2 ATR), tamaño de la posición para arriesgar el 1 % de la cuenta, la probabilidad de ganar que da el modelo de fiabilidad (`fiab.py`, entrenado solo con operaciones ya cerradas)
y el grupo de valores correlacionados (para no comprar cinco veces lo mismo). Guarda cada día qué acciones cumplían cada regla (`impulso_hist.json`) y simula hacia delante, con precios
reales, lo que pasó con cada una (`impulso_track.json`): el track record que no se puede trucar.
"""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from . import fiab, ml, plan as P, search, ta
from .ml import CTX

HIST_MAX = 600             # días que se conservan en el registro
STOP_ATR = 2.5             # stop inicial de la escalera (en ATR)
STOP_PCT = 6.0             # stop inicial alternativo (% del precio)
CORR_WIN = 60              # sesiones para la correlación entre valores
CORR_MIN = 0.6             # dos valores con correlación de rendimientos ≥ 0,6 van en el mismo grupo
RISK_PCT = 1.0             # riesgo por operación (% de la cuenta) del plan de hoy
MAX_POS = 5                # posiciones máximas del plan de hoy


def _flags(D: ml.Data, rules) -> Dict[str, np.ndarray]:
    cond = search.conditions(D)
    out = {}
    for r in rules:
        m = np.ones(len(D.sym), bool)
        for k in r['regla']:
            m &= cond[k]
        out[r['nombre']] = m
    return out


def corr_matrix(bars: Dict[str, Dict[str, np.ndarray]], syms: List[str], win: int = CORR_WIN, upto: Optional[int] = None) -> np.ndarray:
    """Correlación de los rendimientos diarios de las últimas `win` sesiones entre `syms` (NaN donde falta historia). Con `upto` (día en días desde 1970) solo se usan barras hasta ese día."""
    R = []
    for s in syms:
        b = bars.get(s)
        c = np.asarray(b['c'], dtype=float) if b is not None else np.zeros(0)
        if upto is not None and b is not None:
            days = ((np.asarray(b['t']) + 5 * 3600) // 86400).astype(np.int64)
            c = c[days <= upto]
        c = c[-(win + 1):]
        r = np.diff(np.log(c)) if len(c) > 1 else np.zeros(0)
        R.append(r if len(r) == win else np.full(win, np.nan))
    R = np.array(R) if R else np.zeros((0, win))
    C = np.full((len(syms), len(syms)), np.nan)
    ok = np.isfinite(R).all(axis=1) if len(R) else np.zeros(0, bool)
    if ok.sum() >= 2:
        idx = np.flatnonzero(ok)
        C[np.ix_(idx, idx)] = np.corrcoef(R[ok])
    return C


def clusters(bars: Dict[str, Dict[str, np.ndarray]], syms: List[str], win: int = CORR_WIN, thr: float = CORR_MIN) -> Dict[str, int]:
    """Grupos de valores correlacionados (unión de pares con correlación de rendimientos diarios ≥ thr en las últimas `win` sesiones)."""
    if not syms:
        return {}
    C = corr_matrix(bars, syms, win)
    ok = np.isfinite(np.diagonal(C))
    parent = list(range(len(syms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if ok.sum() >= 2:
        idx = np.flatnonzero(ok)
        for a in idx:
            for b in idx:
                if b > a and C[a, b] >= thr:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
    roots = {}
    out = {}
    for i, s in enumerate(syms):
        r = find(i)
        roots.setdefault(r, len(roots) + 1)
        out[s] = roots[r]
    return out


def screen(bars: Dict[str, Dict[str, np.ndarray]], k: int = ml.K, workers: int = 1, rules=search.ENTRADAS, verbose: bool = False, model: bool = True, abiertas: Tuple[str, ...] = ()) -> Dict:
    """Acciones que cumplen cada regla en el último cierre, con su plan, su fiabilidad y su grupo. `bars`: barras diarias por valor (≥ 400 barras). `abiertas`: valores que la cartera del
    plan tiene abiertos (del seguimiento): ocupan hueco y ningún valor correlacionado con ellos entra."""
    t0 = time.time()
    D, Pn, _ = search.build(bars, workers, k, verbose, search.MIN_DAY, search.MIN_COMBO, None, search.COST, library=False, keep_last=True)
    flags = _flags(D, rules)
    last = int(D.dcode.max())
    today, before = D.dcode == last, D.dcode == last - 1
    ctx = lambda name: D.ctx[:, CTX.index(name)]
    pct = lambda name: D.X_ctx[:, CTX.index(name)]
    pn = {n: j for j, n in enumerate(D.pnames)}
    nbear = D.pat[:, pn['n_velas_dn']].astype(int) + D.pat[:, pn['n_fig_dn']].astype(int) + D.pat[:, pn['n_rup_dn']].astype(int)
    prob = np.full(len(D.sym), np.nan)
    fiab_meta = {}
    if model:
        try:
            X = fiab.features(D)
            y, ok, _ = fiab.target(Pn)
            dates = D.date[D.dcode]
            tr = ok & (dates < D.date[last] - np.timedelta64(fiab.PURGE_DAYS, 'D'))       # solo operaciones cerradas sin sesgo: las de los últimos 95 días solo están cerradas si salieron pronto
            if tr.sum() >= 20000:
                m_ = fiab._fit(X[tr], y[tr], fiab.PARAMS)
                prob[today] = m_.predict_proba(X[today])[:, 1]
                fiab_meta = {'filas_entrenamiento': int(tr.sum()), 'hasta': str(dates[tr].max()), 'base_hoy': round(float(100 * np.nanmean(prob[today])), 1)}
        except Exception as exc:                                             # sin scikit-learn o sin historia suficiente: el cribador sigue sin la columna
            fiab_meta = {'error': f'{type(exc).__name__}: {exc}'}
    hit_today = {r['nombre']: set(D.sym[flags[r['nombre']] & today].tolist()) for r in rules}
    hit_before = {r['nombre']: set(D.sym[flags[r['nombre']] & before].tolist()) for r in rules}
    picks = []
    for row in np.flatnonzero(today):
        j = int(D.sym[row])
        names = [r['nombre'] for r in rules if j in hit_today[r['nombre']]]
        if not names:
            continue
        s = D.syms[j]
        close = float(bars[s]['c'][int(D.bar[row])])
        atrp = float(ctx('atrp')[row])
        atr = atrp * close / 100.0
        risk = STOP_ATR * atr / close * 100
        picks.append({'sym': s, 'close': round(close, 4), 'atr': round(atr, 4), 'atrp': round(atrp, 2), 'vol': round(float(pct('atrp')[row]), 3), 'dist200': round(float(ctx('dist200')[row]) * 100, 2),
                      'r120': round(float(pct('r120')[row]), 3), 'mom121': round(float(pct('mom12_1')[row]), 3), 'rsi2': round(float(ctx('rsi2')[row]), 1), 'pos52': round(float(ctx('pos52')[row]), 3),
                      'volrel': round(float(D.volraw[row]), 2), 'bajistas': int(nbear[row]), 'reglas': names, 'nuevas': [n for n in names if j not in hit_before[n]],
                      'fiab': round(float(100 * prob[row]), 1) if np.isfinite(prob[row]) else None,
                      'plan': {'stop': round(close - STOP_ATR * atr, 4), 'stop_pct': round(-risk, 2), 'stop6': round(close * (1 - STOP_PCT / 100), 4), 'asegura4': round(close * 1.04, 4), 'asegura10': round(close * 1.10, 4),
                               'asegura6': round(close * 1.06, 4), 'persigue': round(P.TRAIL2_ATR * atr, 4), 'tam1': round(RISK_PCT * 100.0 / risk, 1), 'tam1_s6': round(RISK_PCT * 100.0 / STOP_PCT, 1),
                               'rr4': round(4.0 / risk, 2), 'rr10': round(10.0 / risk, 2)}})
    abiertas = tuple(a for a in abiertas if a in bars)
    syms_all = [p['sym'] for p in picks] + [a for a in abiertas if a not in {p['sym'] for p in picks}]
    grp = clusters(bars, syms_all)
    for p in picks:
        p['grupo'] = grp.get(p['sym'])
    picks.sort(key=lambda p: (-(p['fiab'] if p['fiab'] is not None else -1), -len(p['reglas']), -p['mom121'], p['sym']))
    bre = float(D.breadth_d[last])
    plan_hoy = daily_plan(picks, abiertas, corr_matrix(bars, syms_all) if syms_all else None, syms_all)
    plan_hoy['grupos_abiertas'] = {a: grp.get(a) for a in abiertas}
    return {'meta': {'asof': str(D.date[last]), 'generated': int(time.time()), 'valores': len(D.syms), 'k': k, 'en_universo': int(today.sum()), 'sobre_200': round(bre * 100, 1), 'coste': ml.COST,
                     'escalera': [list(x) for x in P.LADDER], 'stop_atr': STOP_ATR, 'stop_pct': STOP_PCT, 'riesgo_pct': RISK_PCT, 'max_pos': MAX_POS, 'corr_min': CORR_MIN, 'fiab': fiab_meta,
                     'reglas': [{'nombre': r['nombre'], 'etiqueta': r.get('etiqueta', r['nombre']), 'corta': r.get('corta', r['nombre']), 'regla': r['regla'], 'sentido': r['sentido'], 'salida': r['salida'], 'texto': search._rule(r['regla'], r['sentido'])} for r in rules], 'segundos': round(time.time() - t0)},
            'resumen': {r['nombre']: {'hoy': len(hit_today[r['nombre']]), 'nuevas': len(hit_today[r['nombre']] - hit_before[r['nombre']])} for r in rules}, 'picks': picks, 'plan_hoy': plan_hoy}


def daily_plan(picks: List[Dict], abiertas: Tuple[str, ...] = (), corr: Optional[np.ndarray] = None, syms: Optional[List[str]] = None, max_pos: int = MAX_POS, corr_min: float = CORR_MIN) -> Dict:
    """La cartera del plan, con estado: `abiertas` son las posiciones que el plan ya tiene (del seguimiento) y ocupan hueco; solo se rellenan los huecos libres (`max_pos` − abiertas) con las
    acciones de mayor fiabilidad del día que tengan volatilidad alta, no estén ya abiertas y no tengan correlación ≥ `corr_min` (rendimientos a 60 sesiones; `corr` sobre `syms`) con ninguna
    abierta ni con ninguna elegida antes. Con los huecos llenos no entra nada hasta que cierre alguna. Es exactamente la cartera que simula vp/cartera.py."""
    abiertas = [a for a in abiertas]
    pos = {s_: j for j, s_ in enumerate(syms or [])}
    huecos = max(0, max_pos - len(abiertas))

    def correlado(a: str, bs: List[str]) -> bool:
        if corr is None or a not in pos:
            return False
        return any(b in pos and np.isfinite(corr[pos[a], pos[b]]) and corr[pos[a], pos[b]] >= corr_min for b in bs if b != a)
    nuevas: List[str] = []
    for p in picks:
        if len(nuevas) >= huecos:
            break
        if p['sym'] in abiertas or p['sym'] in nuevas or (p.get('vol') is not None and p['vol'] < 0.6):
            continue
        if correlado(p['sym'], abiertas + nuevas):
            continue
        nuevas.append(p['sym'])
    return {'posiciones': nuevas, 'nuevas': nuevas, 'abiertas': abiertas, 'huecos': huecos, 'max_pos': max_pos, 'riesgo_total_pct': round(RISK_PCT * (len(abiertas) + len(nuevas)), 1),
            'criterio': f'cartera de {max_pos} posiciones: las abiertas ocupan hueco y solo se rellenan los huecos libres, por orden de fiabilidad, solo volatilidad alta, sin valores ya abiertos ni correlacionados (≥ {corr_min:g}) con ellos, {RISK_PCT:g} % de riesgo cada una'}


def update_history(path: str, ev: Dict) -> List[Dict]:
    """Añade al registro (un elemento por día, sin repetir fecha) las acciones que cumplían cada regla ese día, con lo necesario para simularlas después."""
    try:
        hist = json.load(open(path, encoding='utf-8'))
    except (OSError, ValueError):
        hist = []
    pl = ev.get('plan_hoy', {})
    today = {'date': ev['meta']['asof'], 'reglas': {r['nombre']: sorted(p['sym'] for p in ev['picks'] if r['nombre'] in p['reglas']) for r in ev['meta']['reglas']},
             'picks': {p['sym']: {'close': p['close'], 'atr': p['atr'], 'vol': p['vol'], 'fiab': p['fiab'], 'grupo': p['grupo'], 'reglas': p['reglas']} for p in ev['picks']},
             'plan_hoy': pl.get('nuevas', pl.get('posiciones', [])), 'plan_abiertas': pl.get('abiertas', [])}
    hist = [h for h in hist if h.get('date') != today['date']] + [today]
    hist.sort(key=lambda h: h['date'])
    return hist[-HIST_MAX:]


def track(hist: List[Dict], bars: Dict[str, Dict[str, np.ndarray]], cost: float = ml.COST) -> Dict:
    """Seguimiento hacia delante: cada acción que salió en el cribador se simula con la escalera (entrada en la apertura siguiente, stop 2,5 ATR del día de la señal) con los precios reales
    posteriores. Una operación por valor y día de señal; si el valor ya tenía una operación abierta no se abre otra (como haría un operador): esa señal repetida se anota en la operación
    abierta (`repite`: fechas en que volvió a salir; `plan_dias`: las que además estaban en el plan de hoy). Aparte, `plan_ops` es la CARTERA DEL PLAN: solo las entradas que el plan de hoy
    mandó comprar cada día (`plan_hoy` del registro), simuladas igual; es el track record de la cartera de 5 huecos."""
    rows = []
    open_until: Dict[str, int] = {}
    last_row: Dict[str, Dict] = {}
    Fs: Dict[str, Dict[str, np.ndarray]] = {}

    def sim(h: Dict, s: str, plan: bool) -> Optional[Dict]:
        b = bars.get(s)
        if b is None:
            return None
        days = ((np.asarray(b['t']) + 5 * 3600) // 86400).astype(np.int64)
        w = np.flatnonzero(days == np.datetime64(h['date'], 'D').astype('datetime64[D]').astype(np.int64))
        if not len(w):
            return None
        i = int(w[0])
        if s not in Fs:
            Fs[s] = ta.prepare(b)
        F = Fs[s]
        a = float(F['atr'][i])
        if not np.isfinite(a) or a <= 0:
            return None
        pk = (h.get('picks') or {}).get(s, {})
        base = {'fecha': h['date'], 'sym': s, 'reglas': pk.get('reglas') or [r_ for r_, v in h['reglas'].items() if s in v], 'fiab': pk.get('fiab'), 'vol': pk.get('vol'), 'plan_hoy': plan or s in h.get('plan_hoy', [])}
        if i + 1 >= len(F['c']):
            return {**base, 'res': 'pendiente', 'i': i, 'xi': None}
        entry = float(F['o'][i + 1])
        stop = float(F['c'][i]) - STOP_ATR * a
        o = P.outcome_ladder(F, i, 1, stop)
        cur = float(F['c'][-1])
        r = {**base, 'entrada': round(entry, 4), 'stop': round(stop, 4), 'res': o['res'], 'bars': o.get('bars'), 'ret': o.get('ret'), 'net': o.get('net'), 'level': o.get('level', 0), 'xp': o.get('xp'),
             'ahora': round((cur / entry - 1) * 100, 2) if o['res'] == 'open' else None, 'salida': str(np.datetime64(int(days[o['xi']]), 'D')) if o.get('xi') is not None and o['res'] != 'open' else None,
             'stop_actual': o.get('stop_now') if o['res'] == 'open' else None, 'i': i, 'xi': o.get('xi') if o['res'] != 'open' else None}
        return r
    # la cartera del plan: una operación por entrada del plan, una abierta por valor como mucho
    plan_ops: List[Dict] = []
    plan_until: Dict[str, int] = {}
    for h in hist:
        for s in h.get('plan_hoy', []):
            r = sim(h, s, True)
            if r is None or r['res'] == 'na':
                continue
            if s in plan_until and r['i'] <= plan_until[s]:
                continue
            plan_ops.append({k_: v for k_, v in r.items() if k_ not in ('i', 'xi')})
            plan_until[s] = r['xi'] if r.get('xi') is not None else 10 ** 9
    for h in hist:
        syms = sorted(set().union(*[set(v) for v in h.get('reglas', {}).values()]))
        for s in syms:
            r = sim(h, s, False)
            if r is None:
                continue
            if s in open_until and r['i'] <= open_until[s]:
                rw = last_row.get(s)
                if rw is not None:
                    rw.setdefault('repite', []).append(h['date'])
                    if s in h.get('plan_hoy', []):
                        rw.setdefault('plan_dias', []).append(h['date'])
                continue
            i, xi = r.pop('i'), r.pop('xi')
            rows.append(r)
            last_row[s] = r
            if r['res'] != 'na' and r['res'] != 'pendiente':                  # una operación que no llegó a abrirse (la apertura ya estaba más allá del stop) no bloquea futuras señales del valor
                open_until[s] = int(xi) if xi is not None else 10 ** 9

    def tally(rs: List[Dict]) -> Dict:
        cl = [x for x in rs if x['res'] in ('target', 'stop', 'timeout') and x.get('net') is not None]
        op = [x for x in rs if x['res'] == 'open']
        nets = [x['net'] for x in cl]
        wins = [x for x in nets if x > 0]
        loss = [x for x in nets if x <= 0]
        return {'n': len(rs), 'cerradas': len(cl), 'abiertas': len(op), 'pendientes': sum(1 for x in rs if x['res'] == 'pendiente'), 'ganan': len(wins), 'pct_ganan': round(100 * len(wins) / len(cl), 1) if cl else None, 'asegura4': sum(1 for x in cl if x['level'] >= 1),
                'asegura10': sum(1 for x in cl if x['level'] >= 2), 'stop': sum(1 for x in cl if x['res'] == 'stop'), 'media_neta': round(float(np.mean(nets)), 3) if nets else None, 'suma_neta': round(float(np.sum(nets)), 2) if nets else None,
                'gan_media': round(float(np.mean(wins)), 2) if wins else None, 'perd_media': round(float(np.mean(loss)), 2) if loss else None, 'abiertas_media': round(float(np.mean([x['ahora'] for x in op])), 2) if op else None,
                'asegurado_abiertas': sum(1 for x in op if x.get('level', 0) >= 1)}
    rules = sorted({r_ for x in rows for r_ in (x.get('reglas') or [])})
    return {'generated': int(time.time()), 'desde': hist[0]['date'] if hist else None, 'hasta': hist[-1]['date'] if hist else None, 'dias': len(hist), 'todas': tally(rows),
            'plan_hoy': tally(plan_ops), 'vol_alta': tally([x for x in rows if (x.get('vol') or 0) >= 0.6]),
            'por_regla': {r_: tally([x for x in rows if r_ in (x.get('reglas') or [])]) for r_ in rules}, 'operaciones': rows, 'plan_ops': plan_ops, 'max_pos': MAX_POS}


def plan_abiertas(tr: Dict) -> List[str]:
    """Valores que la cartera del plan tiene abiertos o pendientes de entrar (del seguimiento): ocupan hueco en el plan de hoy."""
    return sorted({o['sym'] for o in tr.get('plan_ops', []) if o['res'] in ('open', 'pendiente')})


def replan_history(hist: List[Dict], bars: Dict[str, Dict[str, np.ndarray]], max_pos: int = MAX_POS) -> List[Dict]:
    """Vuelve a calcular el plan de cada día del registro con la regla de la cartera (huecos, abiertas, correlación), en orden y solo con lo que se sabía ese día (las candidatas guardadas con su
    fiabilidad y volatilidad, las barras hasta esa fecha). Sirve para pasar un registro hecho con el plan sin estado al plan con estado; con el mismo código es idempotente."""
    out: List[Dict] = []
    for h in hist:
        prev = out + []
        tr = track(prev, {s_: b for s_, b in bars.items()}) if prev else {'plan_ops': []}
        day = int(np.datetime64(h['date'], 'D').astype('datetime64[D]').astype(np.int64))
        # «abiertas» ese día: entradas del plan cuyo valor seguía abierto al cierre de ese día (la salida, si la hay, es posterior)
        ab = []
        for o in tr.get('plan_ops', []):
            if o['res'] == 'pendiente' or (o['res'] == 'open') or (o.get('salida') and o['salida'] > h['date']):
                ab.append(o['sym'])
        ab = sorted(set(ab))
        picks = [{'sym': s_, **v} for s_, v in (h.get('picks') or {}).items()]
        picks.sort(key=lambda p: (-(p['fiab'] if p.get('fiab') is not None else -1), p['sym']))
        syms_all = [p['sym'] for p in picks] + [a for a in ab if a not in {p['sym'] for p in picks}]
        corr = corr_matrix(bars, syms_all, upto=day) if syms_all else None
        pl = daily_plan(picks, tuple(ab), corr, syms_all, max_pos)
        out.append({**h, 'plan_hoy': pl['nuevas'], 'plan_abiertas': pl['abiertas']})
    return out


def run(symbols: List[str], out_dir: str, rng: str = '10y', cache_dir: Optional[str] = None, workers: int = 2, replan: bool = False) -> Dict:
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, cache_dir=cache_dir, max_age_h=12, verbose=True)
    if len(bars) < 0.9 * len(symbols):
        raise SystemExit(f'solo {len(bars)} de {len(symbols)} valores descargados (Yahoo limita las peticiones): se detiene')
    os.makedirs(out_dir, exist_ok=True)
    hpath = os.path.join(out_dir, 'impulso_hist.json')
    try:
        prev = json.load(open(hpath, encoding='utf-8'))
    except (OSError, ValueError):
        prev = []
    if replan and prev:
        prev = replan_history(prev, bars)
        json.dump(prev, open(hpath, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
        print(f'  registro replanificado con la cartera de {MAX_POS} huecos: ' + ' · '.join(f"{h['date']}: abiertas {h.get('plan_abiertas')} → entra {h['plan_hoy']}" for h in prev), flush=True)
    abiertas = plan_abiertas(track(prev, bars)) if prev else []
    ev = screen(bars, workers=workers, verbose=True, abiertas=tuple(abiertas))
    json.dump(ev, open(os.path.join(out_dir, 'impulso.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    hist = update_history(hpath, ev)
    json.dump(hist, open(os.path.join(out_dir, 'impulso_hist.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    tr = track(hist, bars)
    json.dump(tr, open(os.path.join(out_dir, 'impulso_track.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    m = ev['meta']
    print(f"CRIBADOR DE LA REGLA DE IMPULSO · {m['asof']} · {m['en_universo']} acciones en el universo (las {m['k']} más negociadas) · {m['sobre_200']} % sobre su media de 200 · fiabilidad: {m['fiab']}", flush=True)
    for r in m['reglas']:
        s = ev['resumen'][r['nombre']]
        print(f"  {r['etiqueta']} ({r['nombre']}: {r['texto']} · {r['salida']}): {s['hoy']} hoy ({s['nuevas']} nuevas)", flush=True)
    for p in ev['picks'][:20]:
        print(f"    {p['sym']:6s} {p['close']:9.2f}  fiab {p['fiab'] if p['fiab'] is not None else '  -'}  vol {p['vol'] * 100:3.0f}  impulso {p['mom121'] * 100:3.0f}  vs 200 {p['dist200']:+6.1f} %  grupo {p['grupo']}  stop {p['plan']['stop_pct']:+.1f} %  {','.join(p['reglas'])}", flush=True)
    print(f"  plan de hoy: abiertas {ev['plan_hoy']['abiertas']} · huecos {ev['plan_hoy']['huecos']} · entran {ev['plan_hoy']['nuevas']} · cartera del plan: {tr['plan_hoy']} · todas las señales: {tr['todas']}", flush=True)
    sys.stdout.flush()
    return ev
