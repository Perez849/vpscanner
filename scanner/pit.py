#!/usr/bin/env python3
"""
pit.py — SESGO DE SUPERVIVENCIA: ¿cuánto de la ventaja es real? (comandos `pit` y `pit_mid` de research.py; NO interviene en producción)

El universo de las pruebas es la lista ACTUAL de los índices S&P 500/400/600. Eso cuela dos sesgos:
  1. Hindsight de entrada: una acción que hoy está en el índice entró DESPUÉS de subir mucho (Tesla, Palantir…); en el pasado se la
     cuenta como si siempre hubiera estado en el universo.
  2. Las que salieron del índice (caídas, quiebras, compras) no están.
Aquí se reconstruye la pertenencia en cada fecha con la tabla «Selected changes to the list of S&P … components» de Wikipedia, se
descargan las retiradas que aún cotizan (las que quebraron o se fusionaron ya no tienen datos: sesgo residual, se cuantifica la cobertura)
y se repiten las pruebas: sistema de rebotes (`pit`) y búsqueda masiva a medio plazo (`pit_mid`) solo con acciones MIEMBRO EN ESA FECHA.
"""
from __future__ import annotations
import math
import re
import sys
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import universe as UV

PAGES = {'sp500': 'List_of_S%26P_500_companies', 'sp400': 'List_of_S%26P_400_companies', 'sp600': 'List_of_S%26P_600_companies'}
GROUP_OF = {'sp500': 'us_large', 'sp400': 'us_mid', 'sp600': 'us_small'}
NEG, POS = -10 ** 9, 10 ** 9
TICK = re.compile(r'[A-Z0-9\-]{1,6}')
FROM_YEAR = '2015-06-01'                      # retiradas anteriores no pueden tocar los 10 años de datos


def _norm(s) -> str | None:
    s = re.sub(r'\[[^\]]*\]', '', str(s)).strip().replace('.', '-')
    return s if TICK.fullmatch(s) else None


def _flat(c) -> str:
    parts = c if isinstance(c, tuple) else (c,)
    return ' '.join(dict.fromkeys(str(x).strip().lower() for x in parts))


_DATE_RES = (re.compile(r'([A-Z][a-z]{2,8}\.? \d{1,2},? \d{4})'), re.compile(r'(\d{1,2} [A-Z][a-z]{2,8}\.? \d{4})'), re.compile(r'(\d{4}-\d{2}-\d{2})'))


def _day(x) -> int | None:
    txt = re.sub(r'\[[^\]]*\]', '', str(x))
    for rx in _DATE_RES:
        m = rx.search(txt)
        if m:
            t = pd.to_datetime(m.group(1), errors='coerce')
            if not pd.isna(t):
                return int(t.value // 86400 // 10 ** 9)
    return None


def parse_page(html: str, min_rows: int = 25) -> Tuple[set, List[Tuple[int, str | None, str | None]]]:
    """(miembros actuales, cambios [(día efectivo, ticker añadido, ticker retirado)]) de una página de índice de Wikipedia."""
    cur: set = set()
    chg: List[Tuple[int, str | None, str | None]] = []
    for t in pd.read_html(StringIO(html)):
        cols = [_flat(c) for c in t.columns]
        def pick(word):
            cand = [i for i, c in enumerate(cols) if word in c and 'date' not in c]
            return next((i for i in cand if 'ticker' in cols[i] or 'symbol' in cols[i]), cand[0] if cand else None)
        ia, ir = pick('added'), pick('removed')
        idt = next((i for i, c in enumerate(cols) if 'date' in c and 'added' not in c), None)
        if ia is not None and ir is not None and idt is not None:
            for row in t.itertuples(index=False):
                d = _day(row[idt])
                if d is not None:
                    chg.append((d, _norm(row[ia]), _norm(row[ir])))
            continue
        for cn in ('symbol', 'ticker'):
            if cn in cols and not cur:
                syms = [_norm(s) for s in t.iloc[:, cols.index(cn)].tolist()]
                ok = [s for s in syms if s]
                if len(ok) >= min_rows:
                    cur = set(ok)
    if cur and not chg:
        for t in pd.read_html(StringIO(html)):
            print(f'    [diagnóstico] tabla {t.shape}: columnas {[_flat(c)[:30] for c in t.columns][:7]} · primera fila {[str(x)[:22] for x in t.iloc[0].tolist()][:6] if len(t) else []}', flush=True)
    return cur, chg


def intervals(cur: set, chg: List[Tuple[int, str | None, str | None]]) -> Dict[str, List[Tuple[int, int]]]:
    """Intervalos [alta, baja) de pertenencia por ticker. Sin cambios registrados: miembro siempre si lo es hoy."""
    ev: Dict[str, List[Tuple[int, int]]] = {}
    for d, a, r in chg:
        if a:
            ev.setdefault(a, []).append((d, +1))
        if r:
            ev.setdefault(r, []).append((d, -1))
    out: Dict[str, List[Tuple[int, int]]] = {}
    for t in set(ev) | cur:
        e = sorted(ev.get(t, []))
        if not e:
            if t in cur:
                out[t] = [(NEG, POS)]
            continue
        state = e[0][1] == -1                                   # primer cambio = baja ⇒ ya era miembro antes
        start = NEG if state else None
        iv: List[Tuple[int, int]] = []
        for d, s in e:
            if s == +1 and not state:
                state, start = True, d
            elif s == -1 and state:
                iv.append((start, d)); state = False
        if state:
            iv.append((start, POS))
        elif t in cur:                                           # incoherencia de la tabla: manda la lista actual
            iv.append((e[-1][0], POS))
        out[t] = iv
    return out


def _api(query: str):
    import json
    txt = UV._get(f'https://en.wikipedia.org/w/api.php?{query}&format=json')
    try:
        return json.loads(txt) if txt else None
    except Exception:
        return None


def changes_elsewhere(page: str, label: str) -> List[Tuple[int, str | None, str | None]]:
    """Si la página del índice no trae la tabla de cambios (p. ej. porque se movió a otro artículo), la busca por secciones transcluidas y por búsqueda."""
    from urllib.parse import quote
    cands: List[str] = []
    j = _api(f'action=parse&page={page}&prop=sections')
    if j and 'parse' in j:
        secs = j['parse'].get('sections', [])
        print(f'    [diagnóstico] secciones con «change»: {[(s.get("line"), s.get("fromtitle")) for s in secs if "hange" in s.get("line", "")]}', flush=True)
        cands += [s['fromtitle'].replace(' ', '_') for s in secs if 'hange' in s.get('line', '') and s.get('fromtitle') and s['fromtitle'].replace(' ', '_') != page]
    j = _api('action=query&list=search&srlimit=8&srsearch=' + quote(f'{label} component changes'))
    if j and 'query' in j:
        found = [r['title'].replace(' ', '_') for r in j['query'].get('search', [])]
        print(f'    [diagnóstico] búsqueda: {found}', flush=True)
        cands += found
    for c in dict.fromkeys(cands):
        html = UV._get(f'https://en.wikipedia.org/wiki/{quote(c)}')
        if not html:
            continue
        try:
            _, chg = parse_page(html, min_rows=10 ** 9)
        except Exception:
            continue
        print(f'    [diagnóstico] {c}: {len(chg)} cambios', flush=True)
        if len(chg) >= 50 and any(r for _, _, r in chg):
            return chg
    return []


def fetch_membership() -> Dict[str, Dict]:
    """Descarga las tres páginas. Devuelve {índice: {'cur': set, 'chg': [...], 'iv': {ticker: [(a,b)]}}} (vacío si no hay red)."""
    res: Dict[str, Dict] = {}
    for k, page in PAGES.items():
        html = UV._get(f'https://en.wikipedia.org/wiki/{page}')
        if not html:
            print(f'  {k}: página no disponible', flush=True)
            continue
        cur, chg = parse_page(html)
        if cur and not chg:
            chg = changes_elsewhere(PAGES[k], {'sp500': 'S&P 500', 'sp400': 'S&P 400', 'sp600': 'S&P 600'}[k])
        iv = intervals(cur, chg)
        rem = sum(1 for _, _, r in chg if r)
        print(f'  {k}: {len(cur)} miembros actuales · {len(chg)} cambios en la tabla ({rem} retiradas) · desde {pd.to_datetime(min([c[0] for c in chg], default=0) * 86400, unit="s").date()}', flush=True)
        yrs = pd.to_datetime(np.array([c[0] for c in chg]) * 86400, unit='s').year if chg else []
        print('    cambios por año (2005–2026): ' + ' '.join(f'{y}:{int((np.asarray(yrs) == y).sum())}' for y in range(2005, 2027, 1)), flush=True)
        res[k] = {'cur': cur, 'chg': chg, 'iv': iv}
    return res


def merged_intervals(mem: Dict[str, Dict], which=None) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    for k in (which or list(mem)):
        for t, ivs in mem.get(k, {}).get('iv', {}).items():
            out.setdefault(t, []).extend(ivs)
    return out


def event_mask(names: List[str], sym: np.ndarray, day: np.ndarray, iv: Dict[str, List[Tuple[int, int]]], require_info: bool = False) -> np.ndarray:
    """True si el símbolo era miembro en ese día. Sin información (ETF, EU, cripto…): True salvo que require_info."""
    ok = np.ones(len(sym), bool)
    for k in np.unique(sym):
        ivs = iv.get(names[k])
        m = sym == k
        if ivs is None:
            if require_info:
                ok[m] = False
            continue
        d = day[m]
        inside = np.zeros(len(d), bool)
        for a, b in ivs:
            inside |= (d >= a) & (d < b)
        ok[m] = inside
    return ok


def member_matrix(names: List[str], days: np.ndarray, iv: Dict[str, List[Tuple[int, int]]], default: bool) -> np.ndarray:
    """(len(days) × len(names)) pertenencia por fecha; sin información → default."""
    M = np.full((len(days), len(names)), default, bool)
    for j, n in enumerate(names):
        ivs = iv.get(n)
        if ivs is None:
            continue
        col = np.zeros(len(days), bool)
        for a, b in ivs:
            col |= (days >= a) & (days < b)
        M[:, j] = col
    return M


def extra_universe(uni: Dict, mem: Dict[str, Dict], since_str: str = FROM_YEAR) -> Dict[str, str]:
    """Retiradas del índice desde FROM_YEAR que no están en el universo actual → {ticker: grupo}."""
    since = int(pd.Timestamp(since_str).value // 86400 // 10 ** 9)
    extra: Dict[str, str] = {}
    for k in ('sp500', 'sp400', 'sp600'):
        for d, a, r in mem.get(k, {}).get('chg', []):
            for t in (r, a):
                if t and d >= since and t not in uni and t not in extra and not (a == t and t in mem[k]['cur']):
                    extra[t] = GROUP_OF[k]
    return extra


def load_extras(uni: Dict, data: Dict, mem: Dict[str, Dict], args, since: str = FROM_YEAR, period1: int | None = None) -> Tuple[Dict, Dict, Dict]:
    """Descarga las retiradas que aún tienen datos y las añade (copias) a uni/data. Informa de la cobertura."""
    import data as D
    extra = extra_universe(uni, mem, since)
    print(f'\n  retiradas/altas ya fuera del universo actual desde {since}: {len(extra)} tickers', flush=True)
    bars, failed = D.fetch_many(list(extra), rng=args.range, workers=args.workers, cache_path=None, verbose=False, period1=period1)
    got = {s: b for s, b in bars.items() if len(b['c']) >= 260}
    iv = merged_intervals(mem)
    ends = 0
    for s, b in got.items():
        last = int(b['t'][-1] // 86400)
        ivs = iv.get(s, [])
        rem = max([e for _, e in ivs if e < POS], default=None)
        if rem is not None and last < rem - 5:
            ends += 1
    print(f'  con datos de Yahoo: {len(got)} de {len(extra)} ({len(got) / max(len(extra), 1) * 100:.0f} %) · de ellas, {ends} dejaron de cotizar antes de salir del índice · '
          f'SIN datos (quiebras, fusiones, cambios de ticker): {len(extra) - len(got)} → sesgo residual: faltan sobre todo las peores y algunas compradas con prima', flush=True)
    uni2 = dict(uni)
    data2 = dict(data)
    for s, b in got.items():
        uni2[s] = {'yahoo': s, 'group': extra[s], 'pit_extra': True}
        data2[s] = b
    return uni2, data2, {'extra': len(extra), 'got': len(got)}


# ═════════════════════════════════════════════════════════════════════════
#  A. Sistema de rebotes sobre pertenencia histórica
# ═════════════════════════════════════════════════════════════════════════
def run(uni: Dict, data: Dict, args, out_dir: str):
    import research as RS
    import setups as SU
    import simulate as SM
    mem = fetch_membership()
    if not mem:
        raise SystemExit('sin red a Wikipedia: no se puede reconstruir la pertenencia')
    uni2, data2, cov = load_extras(uni, data, mem, args)
    iv = merged_intervals(mem)
    EV = RS.build_events(data2, uni2, SU.SETUPS, SM.default_variants())
    rows, FL, names = RS.union_events(EV, +1)
    last_day = int(EV['day'].max())
    sn = EV['sym_names']
    symr, dayr = EV['sym'][rows], EV['day'][rows]
    is_extra = np.array([bool(uni2[s].get('pit_extra')) for s in sn])[symr]
    in_cur_index = np.array([s in iv for s in sn])[symr]
    pit_ok = event_mask(sn, symr, dayr, iv)
    grp = EV['grp'][rows]
    us = np.isin(grp, [RS.GROUPS.index(g) for g in ('us_large', 'us_mid', 'us_small')])
    print(f'\n########  PERTENENCIA HISTÓRICA · {len(rows):,} candidatos (unión de patrones) · datos hasta {np.datetime64("1970-01-01") + np.timedelta64(last_day, "D")}', flush=True)
    print(f'  candidatos de acciones de EE.UU.: {int(us.sum()):,} · de retiradas del índice: {int((us & is_extra).sum()):,} · '
          f'eventos de acciones actuales ANTES de entrar en el índice (hindsight): {int((us & ~is_extra & in_cur_index & ~pit_ok).sum()):,} '
          f'({(us & ~is_extra & in_cur_index & ~pit_ok).sum() / max((us & ~is_extra).sum(), 1) * 100:.1f} % de los de EE.UU.)', flush=True)
    variants = {
        'A · como está desplegado (lista actual)': ~is_extra,
        'B · solo cuando ya era miembro (lista actual con fecha de alta)': ~is_extra & pit_ok,
        'C · B + retiradas que aún cotizan (pertenencia histórica)': pit_ok,
    }
    summary = {}
    for lab, mask in variants.items():
        print(f'\n================ {lab} · {int(mask.sum()):,} candidatos ================', flush=True)
        plans = []
        for pl in RS.PLANS:
            try:
                plans.append(RS.build_plan(EV, rows[mask], FL[mask], names, pl, last_day, spy=data2.get('SPY')))
            except (KeyError, ValueError, IndexError) as e:           # ningún grupo con evidencia suficiente
                print(f'     (sin resultado: {type(e).__name__} {e})', flush=True)
                plans.append(None)
        summary[lab] = plans
    print('\n################ RESUMEN (política top-N/día con grupos auto-seleccionados, walk-forward 2019+) ################')
    for pi, pl in enumerate(RS.PLANS):
        print(f"\n  Plan «{pl['label']}»")
        for lab, plans in summary.items():
            p = plans[pi]
            if p is None:
                print(f"    {lab[:62]:62s} sin evidencia suficiente")
                continue
            o, rc, vs = p['stats']['oos'], p['stats'].get('recent'), p['stats'].get('vsSpy', {})
            pf = p['stats'].get('portfolio', {}).get('10x10', {})
            print(f"    {lab[:62]:62s} n={o['n']:6.0f} · acierto {o['wr']:4.1f}% · media {o['mean']:+.2f}% · PF {o['pf']:.2f} · "
                  f"desde 2024 {rc['wr'] if rc else float('nan'):4.1f}%/{rc['mean'] if rc else float('nan'):+.2f}% · alfa vs S&P {vs.get('alpha', float('nan')):+.2f}% (t={vs.get('alphaT', float('nan')):.1f}) · "
                  f"cartera 10×10%: {pf.get('cagr', float('nan')):+.1f}% anual / caída {pf.get('dd', float('nan')):.0f}% / Sharpe {pf.get('sharpe', float('nan')):.2f} · grupos {p['groups']}")
    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════
#  B. Búsqueda masiva a medio plazo sobre pertenencia histórica
# ═════════════════════════════════════════════════════════════════════════
def run_mid(uni: Dict, data: Dict, args, out_dir: str):
    import stockmid as SMD
    mem = fetch_membership()
    if not mem:
        raise SystemExit('sin red a Wikipedia: no se puede reconstruir la pertenencia')
    uni2, data2, _ = load_extras(uni, data, mem, args)
    iv500 = merged_intervals(mem, ['sp500'])
    ivall = merged_intervals(mem)
    stk = [s for s in uni2 if uni2[s]['group'] in ('us_large', 'us_mid', 'us_small') and not uni2[s].get('aux') and s in data2 and len(data2[s]['c']) >= 330]

    def member(names: List[str], days: np.ndarray):
        return member_matrix(names, days, iv500, False), member_matrix(names, days, ivall, True)

    SMD.core(uni2, data2, stk, True, 'MEDIO PLAZO CON ACCIONES · PERTENENCIA HISTÓRICA (S&P 500 en cada fecha, con retiradas que aún cotizan)', member=member)
    SMD._etf_persistence(data2)


# ═════════════════════════════════════════════════════════════════════════
#  C. Sistema de rebotes 2005–2026 (incluye 2008) con pertenencia histórica
# ═════════════════════════════════════════════════════════════════════════
def unknown_before(names: List[str], sym: np.ndarray, day: np.ndarray, mem: Dict[str, Dict]) -> np.ndarray:
    """True para eventos de valores cuyo ÚNICO dato de pertenencia viene de una tabla que empieza después de esa fecha (S&P 400/600)."""
    first = {k: min([c[0] for c in v['chg']], default=NEG) for k, v in mem.items()}
    cut: Dict[str, int] = {}
    for t in set().union(*[set(v['iv']) for v in mem.values()]):
        idx_in = [k for k, v in mem.items() if t in v['iv']]
        if idx_in and 'sp500' not in idx_in:
            cut[t] = min(first[k] for k in idx_in)
    out = np.zeros(len(sym), bool)
    for k in np.unique(sym):
        c = cut.get(names[k])
        if c is not None:
            m = sym == k
            out[m] = day[m] < c
    return out


def run_deep(uni: Dict, data: Dict, args, out_dir: str):
    import data as D
    import research as RS
    import setups as SU
    import simulate as SM
    mem = fetch_membership()
    if not mem:
        raise SystemExit('sin red a Wikipedia: no se puede reconstruir la pertenencia')
    p1 = int(pd.Timestamp('2003-01-02').timestamp())
    bars, failed = D.fetch_many(list(uni), workers=args.workers, cache_path=None, verbose=False, period1=p1)
    spy = bars['SPY']
    gaps = np.diff(spy['t'] // 86400)
    if np.median(gaps) > 3 or gaps.max() > 12:
        raise SystemExit(f'barras no diarias (mediana {np.median(gaps)}, máx. {gaps.max()})')
    print(f'  descargadas {len(bars)} series desde 2003 (fallidas {len(failed)}) · SPY desde {pd.to_datetime(spy["t"][0], unit="s").date()}', flush=True)
    uni1 = {s: uni[s] for s in bars}
    uni2, data2, _ = load_extras(uni1, bars, mem, args, since='2004-01-01', period1=p1)
    iv = merged_intervals(mem)
    RS.FIRST_WF_YEAR = 2008
    RS.PORT_START = '2008-01-01'
    EV = RS.build_events(data2, uni2, SU.SETUPS, SM.default_variants())
    rows, FL, names = RS.union_events(EV, +1)
    last_day = int(EV['day'].max())
    sn = EV['sym_names']
    symr, dayr = EV['sym'][rows], EV['day'][rows]
    is_extra = np.array([bool(uni2[s].get('pit_extra')) for s in sn])[symr]
    pit_ok = event_mask(sn, symr, dayr, iv)
    unk = unknown_before(sn, symr, dayr, mem)
    grp = EV['grp'][rows]
    us = np.isin(grp, [RS.GROUPS.index(g) for g in ('us_large', 'us_mid', 'us_small')])
    yrs = RS.year_of(dayr)
    print(f'\n########  SISTEMA DE REBOTES 2005–2026 · {len(rows):,} candidatos · datos hasta {np.datetime64("1970-01-01") + np.timedelta64(last_day, "D")}', flush=True)
    print('  candidatos de acciones de EE.UU. por año (todos / de retiradas / sin información de pertenencia): ' + ' '.join(
        f'{Y}:{int((us & (yrs == Y)).sum())}/{int((us & is_extra & (yrs == Y)).sum())}/{int((us & unk & (yrs == Y)).sum())}' for Y in range(2005, 2027, 3)), flush=True)
    variants = {
        'A · lista actual, tal como está desplegado': ~is_extra,
        'C · pertenencia histórica (con retiradas) y sin eventos de S&P 400/600 anteriores a su tabla': pit_ok & ~unk,
    }
    summary = {}
    curves_by_lab: Dict[str, Dict] = {}
    for lab, mask in variants.items():
        print(f'\n================ {lab} · {int(mask.sum()):,} candidatos ================', flush=True)
        plans = []
        for pl in RS.PLANS:
            try:
                plans.append(RS.build_plan(EV, rows[mask], FL[mask], names, pl, last_day, spy=data2.get('SPY'), curves=curves_by_lab.setdefault(lab, {})))
            except (KeyError, ValueError, IndexError) as e:
                print(f'     (sin resultado: {type(e).__name__} {e})', flush=True)
                plans.append(None)
        summary[lab] = plans
    print('\n################ RESUMEN 2008–2026 ################')
    for pi, pl in enumerate(RS.PLANS):
        print(f"\n  Plan «{pl['label']}»")
        for lab, plans in summary.items():
            p = plans[pi]
            if p is None:
                print(f'    {lab[:70]}: sin evidencia suficiente')
                continue
            o, vs = p['stats']['oos'], p['stats'].get('vsSpy', {})
            pf = p['stats'].get('portfolio', {}).get('10x10', {})
            by = p['stats']['byYear']
            print(f"    {lab[:70]}\n       n={o['n']:.0f} · acierto {o['wr']:.1f}% · media {o['mean']:+.2f}% · PF {o['pf']:.2f} · alfa vs S&P {vs.get('alpha', float('nan')):+.2f}% (t={vs.get('alphaT', float('nan')):.1f}) · "
                  f"cartera 10×10%: {pf.get('cagr', float('nan')):+.1f}% anual / caída {pf.get('dd', float('nan')):.0f}% / Sharpe {pf.get('sharpe', float('nan')):.2f} (S&P: {p['stats'].get('spyHold', {}).get('cagr', float('nan')):+.1f}% / {p['stats'].get('spyHold', {}).get('dd', float('nan')):.0f}%) · grupos {p['groups']}\n"
                  f"       por año: " + ' '.join(f"{y}:{v['wr']:.0f}%/{v['mean']:+.2f}" for y, v in by.items()))
    sys.stdout.flush()
    lab_c = [k for k in curves_by_lab if k.startswith('C')][0]
    cv = curves_by_lab[lab_c].get(('rsi_S4_H10', '10x10'))
    if cv is not None:
        combos(data2, cv)


def combos(data: Dict, curve: np.ndarray):
    """¿Cómo repartir el dinero? S&P 500 (comprar y mantener), tendencia (sobre su SMA200) y rebotes con pertenencia histórica, 2008–2026."""
    spy = data['SPY']
    cal = (spy['t'] // 86400).astype(np.int64)
    n = len(cal)
    c = spy['c']
    r_spy = np.r_[0.0, c[1:] / c[:-1] - 1]
    sma = pd.Series(c).rolling(200).mean().to_numpy()
    cash_sym = 'BIL' if 'BIL' in data else 'SHY'
    cb = data[cash_sym]
    cs = pd.Series(cb['c'], index=(cb['t'] // 86400).astype(np.int64))
    cs = cs[~cs.index.duplicated(keep='last')].reindex(cal).ffill()
    r_cash = np.nan_to_num(cs.pct_change().to_numpy())
    up = np.r_[False, c[:-1] > sma[:-1]]                                  # la regla se decide con el cierre de ayer
    r_tr = np.where(up, r_spy, r_cash)
    full = np.ones(n)
    full[n - len(curve):] = curve
    r_dip = np.r_[0.0, full[1:] / full[:-1] - 1]
    st = n - len(curve) + 1
    ix = slice(st, n)
    yrs = pd.to_datetime(cal[ix] * 86400, unit='s').year.to_numpy()

    def row(name, r):
        r = np.nan_to_num(r[ix])
        eq = np.cumprod(1 + r)
        y = len(r) / 252.0
        by = {Y: float(np.prod(1 + r[yrs == Y]) - 1) * 100 for Y in sorted(set(yrs))}
        print(f"    {name:44s} {(eq[-1] ** (1 / y) - 1) * 100:+6.1f}% anual · vol {r.std() * math.sqrt(252) * 100:4.1f}% · Sharpe {r.mean() / (r.std() + 1e-12) * math.sqrt(252):4.2f} · caída {((eq / np.maximum.accumulate(eq)) - 1).min() * 100:5.1f}% · 2008 {by.get(2008, float('nan')):+5.0f}% · 2018 {by.get(2018, float('nan')):+4.0f}% · 2022 {by.get(2022, float('nan')):+4.0f}%")
    print(f"\n################ CÓMO REPARTIR EL DINERO · {pd.to_datetime(cal[st] * 86400, unit='s').date()} → {pd.to_datetime(cal[-1] * 86400, unit='s').date()} (rebotes con pertenencia histórica, curva a precio realizado; el resto en liquidez {cash_sym}) ################")
    row('S&P 500 comprar y mantener', r_spy)
    row('Tendencia: S&P 500 sobre SMA200, si no liquidez', r_tr)
    row('Rebotes (Equilibrado, 10 posiciones × 10 %)', r_dip)
    row('50 % tendencia + 50 % rebotes', 0.5 * r_tr + 0.5 * r_dip)
    row('70 % tendencia + 30 % rebotes', 0.7 * r_tr + 0.3 * r_dip)
    row('50 % S&P 500 + 50 % rebotes', 0.5 * r_spy + 0.5 * r_dip)
    row('S&P 500 + rebotes encima (sin apalancar la base)', r_spy + 0.5 * r_dip)
    row('70 % tendencia + 30 % rebotes, apalancado x1,5', 1.5 * (0.7 * r_tr + 0.3 * r_dip))
    sys.stdout.flush()
