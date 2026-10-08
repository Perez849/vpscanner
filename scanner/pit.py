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
    s = str(s).strip().replace('.', '-')
    return s if TICK.fullmatch(s) else None


def _flat(c) -> str:
    parts = c if isinstance(c, tuple) else (c,)
    return ' '.join(dict.fromkeys(str(x).strip().lower() for x in parts))


def _day(x) -> int | None:
    t = pd.to_datetime(str(x), errors='coerce')
    return None if pd.isna(t) else int(t.value // 86400 // 10 ** 9)


def parse_page(html: str, min_rows: int = 25) -> Tuple[set, List[Tuple[int, str | None, str | None]]]:
    """(miembros actuales, cambios [(día efectivo, ticker añadido, ticker retirado)]) de una página de índice de Wikipedia."""
    cur: set = set()
    chg: List[Tuple[int, str | None, str | None]] = []
    for t in pd.read_html(StringIO(html)):
        cols = [_flat(c) for c in t.columns]
        ia = next((i for i, c in enumerate(cols) if 'added' in c and ('ticker' in c or 'symbol' in c)), None)
        ir = next((i for i, c in enumerate(cols) if 'removed' in c and ('ticker' in c or 'symbol' in c)), None)
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


def fetch_membership() -> Dict[str, Dict]:
    """Descarga las tres páginas. Devuelve {índice: {'cur': set, 'chg': [...], 'iv': {ticker: [(a,b)]}}} (vacío si no hay red)."""
    res: Dict[str, Dict] = {}
    for k, page in PAGES.items():
        html = UV._get(f'https://en.wikipedia.org/wiki/{page}')
        if not html:
            print(f'  {k}: página no disponible', flush=True)
            continue
        cur, chg = parse_page(html)
        iv = intervals(cur, chg)
        rem = sum(1 for _, _, r in chg if r)
        print(f'  {k}: {len(cur)} miembros actuales · {len(chg)} cambios en la tabla ({rem} retiradas) · desde {pd.to_datetime(min([c[0] for c in chg], default=0) * 86400, unit="s").date()}', flush=True)
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


def extra_universe(uni: Dict, mem: Dict[str, Dict], first_day: int) -> Dict[str, str]:
    """Retiradas del índice desde FROM_YEAR que no están en el universo actual → {ticker: grupo}."""
    since = int(pd.Timestamp(FROM_YEAR).value // 86400 // 10 ** 9)
    extra: Dict[str, str] = {}
    for k in ('sp500', 'sp400', 'sp600'):
        for d, a, r in mem.get(k, {}).get('chg', []):
            for t in (r, a):
                if t and d >= since and t not in uni and t not in extra and not (a == t and t in mem[k]['cur']):
                    extra[t] = GROUP_OF[k]
    return extra


def load_extras(uni: Dict, data: Dict, mem: Dict[str, Dict], args) -> Tuple[Dict, Dict, Dict]:
    """Descarga las retiradas que aún tienen datos y las añade (copias) a uni/data. Informa de la cobertura."""
    import data as D
    extra = extra_universe(uni, mem, 0)
    print(f'\n  retiradas/altas ya fuera del universo actual desde {FROM_YEAR}: {len(extra)} tickers', flush=True)
    bars, failed = D.fetch_many(list(extra), rng=args.range, workers=args.workers, cache_path=None, verbose=False)
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
