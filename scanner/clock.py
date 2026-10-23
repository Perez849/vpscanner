#!/usr/bin/env python3
"""
clock.py — ¿MEJORA EL RESULTADO SI SOLO SE COGEN LAS ALERTAS CUYO SECTOR RECOMIENDA EL «RELOJ DE INVERSIÓN»? (comando `clock`; NO interviene en producción)

El reloj (repositorio Perez849/reloj-inversion) clasifica el ciclo de EE.UU. en cuatro fases y recomienda 2–5 sectores de renta variable (y a veces oro).
`clock_picks.py` reproduce, mes a mes y solo con datos previos, qué sectores habría recomendado (guardado en model/clock_picks.json).

Aquí se cruza ese calendario de sectores con los candidatos del sistema de rebotes:
  A. A nivel de candidato (todas las señales con sector conocido): rentabilidad de las que caen en un sector recomendado frente a las que no,
     con el contraste de permutaciones: si cada mes se eligieran al azar tantos sectores como recomienda el reloj, ¿cuántas veces saldría igual de bien?
  B. Sobre las alertas del plan desplegado (Equilibrado, top-3 por día): quedarse solo con las de sector recomendado.
  C. Sistema entero reentrenado con el fondo de candidatos restringido a sectores recomendados (modelo y política incluidos).
Todo se compara contra el MISMO conjunto de valores con sector conocido sin filtrar, no contra el universo completo, para que el sesgo de supervivencia del
sector (solo hay sector de los valores que cotizan hoy) afecte por igual a las dos ramas.
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Dict

import numpy as np

import pit
import research as RS
import setups as SU
import simulate as SM
import stops as ST

HERE = os.path.dirname(os.path.abspath(__file__))
PICKS_PATH = os.path.join(HERE, 'model', 'clock_picks.json')

EQ = ['Consumo básico', 'Industria', 'Energía', 'Materiales / Químicas', 'Tecnología', 'Comunicaciones', 'Utilities',
      'Consumo discrecional', 'Salud', 'Financiero', 'Semiconductores']
GOLD = ['Oro (lingote)', 'Metales preciosos (mineras)']
OTHER = ['Inmobiliario']                    # el reloj nunca lo recomienda: no está en su bloque de renta variable
NAMES = EQ + GOLD + OTHER
BIT = {n: 1 << i for i, n in enumerate(NAMES)}
EQ_MASK = sum(BIT[n] for n in EQ)
GOLD_MASK = sum(BIT[n] for n in GOLD)

GICS_MAP = {'Consumer Staples': 'Consumo básico', 'Industrials': 'Industria', 'Energy': 'Energía', 'Materials': 'Materiales / Químicas',
            'Information Technology': 'Tecnología', 'Communication Services': 'Comunicaciones', 'Utilities': 'Utilities',
            'Consumer Discretionary': 'Consumo discrecional', 'Health Care': 'Salud', 'Financials': 'Financiero', 'Real Estate': 'Inmobiliario'}

ETF_MAP = {'XLK': 'Tecnología', 'XLF': 'Financiero', 'XLE': 'Energía', 'XLV': 'Salud', 'XLY': 'Consumo discrecional', 'XLP': 'Consumo básico',
           'XLI': 'Industria', 'XLB': 'Materiales / Químicas', 'XLU': 'Utilities', 'XLC': 'Comunicaciones', 'XLRE': 'Inmobiliario',
           'VNQ': 'Inmobiliario', 'IYR': 'Inmobiliario', 'SMH': 'Semiconductores', 'SOXX': 'Semiconductores', 'XBI': 'Salud', 'IBB': 'Salud',
           'ARKG': 'Salud', 'KRE': 'Financiero', 'KBE': 'Financiero', 'KBWB': 'Financiero', 'XHB': 'Consumo discrecional', 'ITB': 'Consumo discrecional',
           'XRT': 'Consumo discrecional', 'XME': 'Materiales / Químicas', 'COPX': 'Materiales / Químicas', 'XOP': 'Energía', 'OIH': 'Energía',
           'IGV': 'Tecnología', 'CIBR': 'Tecnología', 'HACK': 'Tecnología', 'SKYY': 'Tecnología', 'BOTZ': 'Tecnología', 'ITA': 'Industria',
           'XAR': 'Industria', 'PAVE': 'Industria', 'JETS': 'Industria', 'IYT': 'Industria', 'FDN': 'Comunicaciones',
           'GLD': 'Oro (lingote)', 'IAU': 'Oro (lingote)', 'GDX': 'Metales preciosos (mineras)', 'GDXJ': 'Metales preciosos (mineras)'}

WIKI_PAGES = ('List_of_S%26P_500_companies', 'List_of_S%26P_400_companies', 'List_of_S%26P_600_companies', 'Nasdaq-100')



def _quiet(fn, *a, **kw):
    """Ejecuta fn silenciando lo que imprime (build_plan escribe un informe largo por cada versión)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)

def fetch_gics() -> Dict[str, Dict[str, str]]:
    """{símbolo: {'sector': GICS, 'sub': subindustria}} de las tablas de Wikipedia (solo valores que cotizan hoy en esos índices)."""
    import universe as U
    out: Dict[str, Dict[str, str]] = {}
    try:
        import pandas as pd
        from io import StringIO
    except Exception:
        return out
    for page in WIKI_PAGES:
        html = U._get(f'https://en.wikipedia.org/wiki/{page}')
        if not html:
            continue
        try:
            for t in pd.read_html(StringIO(html)):
                cols = [str(c) for c in t.columns]
                sym_c = next((c for c in ('Symbol', 'Ticker') if c in cols), None)
                sec_c = next((c for c in cols if 'GICS' in c and 'Sub' not in c), None)
                sub_c = next((c for c in cols if 'GICS' in c and 'Sub' in c), None)
                if not sym_c or not sec_c:
                    continue
                syms = [str(s).strip().replace('.', '-') for s in t[sym_c].tolist()]
                ok = [bool(re.fullmatch(r'[A-Z0-9\-]{1,6}', s)) for s in syms]
                if sum(ok) < 25:
                    continue
                secs = [str(x).strip() for x in t[sec_c].tolist()]
                subs = [str(x).strip() for x in t[sub_c].tolist()] if sub_c else [''] * len(syms)
                for s, a, b, o in zip(syms, secs, subs, ok):
                    if o and a and a != 'nan' and s not in out:
                        out[s] = {'sector': a, 'sub': b if b != 'nan' else ''}
                break
        except Exception:
            continue
    return out


def tags_of(sym: str, meta: Dict, gics: Dict[str, Dict[str, str]]) -> int:
    """Máscara de sectores del reloj a los que pertenece el valor (0 = sin sector conocido)."""
    if sym in ETF_MAP:
        return BIT[ETF_MAP[sym]]
    g = gics.get(sym)
    if g:
        name = GICS_MAP.get(g['sector'])
        if not name:
            return 0
        m = BIT[name]
        if name == 'Tecnología' and 'Semiconductor' in g.get('sub', ''):
            m |= BIT['Semiconductores']
        return m
    lab = (meta or {}).get('sector') or ''
    if meta and meta.get('group') in ('thematic', 'us_large', 'us_mid', 'us_small') and lab in ('Oro', 'Oro mineras (ETF 2x)', 'Oro jr (ETF 2x)', 'Plata/Oro', 'Oro/Cobre', 'Oro/Diversif'):
        return BIT['Metales preciosos (mineras)']
    return 0


def load_picks(path: str = PICKS_PATH) -> Dict:
    return json.load(open(path, encoding='utf-8'))


def pick_masks(picks: Dict) -> Dict[int, int]:
    """{año*12+mes-1: máscara de sectores recomendados ese mes}. Los meses sin dato heredan el último conocido."""
    out: Dict[int, int] = {}
    for key, v in picks['months'].items():
        y, m = int(key[:4]), int(key[5:7])
        mask = 0
        for n in list(v.get('equity', {})) + list(v.get('gold', {})):
            if n in BIT:
                mask |= BIT[n]
        out[y * 12 + m - 1] = mask
    return out


def month_index(day: np.ndarray) -> np.ndarray:
    d = np.datetime64('1970-01-01') + day.astype('timedelta64[D]')
    return d.astype('datetime64[Y]').astype(int) * 12 + 1970 * 12 + (d.astype('datetime64[M]').astype(int) % 12)


def event_picked(mi: np.ndarray, emask: np.ndarray, pm: Dict[int, int]) -> np.ndarray:
    """Para cada evento: ¿alguno de sus sectores está recomendado en su mes? Meses fuera del calendario → False."""
    keys = np.array(sorted(pm))
    vals = np.array([pm[k] for k in keys], dtype=np.int64)
    j = np.searchsorted(keys, mi, side='right') - 1                      # último mes con dato ≤ el del evento
    have = (j >= 0)
    mask = np.where(have, vals[np.clip(j, 0, len(vals) - 1)], 0)
    return (emask & mask) != 0, have


def permutation_null(mi: np.ndarray, emask: np.ndarray, pnl: np.ndarray, pm: Dict[int, int], reps: int, rng: np.random.Generator):
    """Media de la rentabilidad de los eventos «en sector elegido» cuando cada mes se eligen AL AZAR tantos sectores (de renta variable y de oro) como el reloj."""
    keys = np.array(sorted(pm))
    k_eq = np.array([bin(pm[k] & EQ_MASK).count('1') for k in keys])
    k_gd = np.array([bin(pm[k] & GOLD_MASK).count('1') for k in keys])
    j = np.searchsorted(keys, mi, side='right') - 1
    ok = (j >= 0)
    mi_, em, x, jj = mi[ok], emask[ok], pnl[ok], j[ok]
    eq_bits = np.array([BIT[n] for n in EQ], dtype=np.int64)
    gd_bits = np.array([BIT[n] for n in GOLD], dtype=np.int64)
    means = np.empty(reps)
    for r in range(reps):
        cm = np.zeros(len(keys), dtype=np.int64)
        for q in range(len(keys)):
            if k_eq[q]:
                cm[q] |= int(eq_bits[rng.choice(len(eq_bits), size=min(k_eq[q], len(eq_bits)), replace=False)].sum())
            if k_gd[q]:
                cm[q] |= int(gd_bits[rng.choice(len(gd_bits), size=min(k_gd[q], len(gd_bits)), replace=False)].sum())
        inn = (em & cm[jj]) != 0
        means[r] = x[inn].mean() if inn.any() else np.nan
    return means


def _line(lab: str, x: np.ndarray, day: np.ndarray, sym: np.ndarray) -> str:
    x_ok = np.isfinite(x)
    if x_ok.sum() < 20:
        return '    ' + lab.ljust(34) + f'{int(x_ok.sum()):7d}  (muy pocas)'
    s = RS.stats(x, day, sym)
    return '    ' + lab.ljust(34) + f"{s['n']:7d}{s['wr']:8.1f}%{s['mean']:+8.2f}%{s['pf']:6.2f} t={s['t_day']:+5.1f}"


def run(uni: Dict, data: Dict, args, out_dir: str):
    picks = load_picks()
    pm = pick_masks(picks)
    uni2, data2, row_ok, info = pit.prepare(uni, data, args)
    gics = fetch_gics() if not os.environ.get('CLOCK_FAKE_SECTORS') else {}
    EV = RS.build_events(data2, uni2, SU.SETUPS, SM.default_variants())
    rows, FL, names = RS.union_events(EV, +1)
    k = row_ok(EV, rows)
    rows, FL = rows[k], FL[k]
    last_day = int(EV['day'].max())
    snames = EV['sym_names']
    tg = np.array([tags_of(s, uni2.get(s) or uni.get(s) or {}, gics) for s in snames], dtype=np.int64)
    if os.environ.get('CLOCK_FAKE_SECTORS'):                                   # solo para la prueba con datos sintéticos
        tg = np.array([BIT[EQ[i % len(EQ)]] for i in range(len(snames))], dtype=np.int64)
    sym, day = EV['sym'][rows], EV['day'][rows]
    emask = tg[sym]
    mapped = emask != 0
    mi = month_index(day)
    picked, have = event_picked(mi, emask, pm)
    year = RS.year_of(day)
    print(f'\n######## RELOJ DE INVERSIÓN · {len(rows):,} candidatos · datos hasta {np.datetime64("1970-01-01") + np.timedelta64(last_day, "D")} · '
          f'calendario de sectores {min(picks["months"])}…{max(picks["months"])} (reloj-inversion {picks.get("commit", "")}, probabilidades de fase del mes anterior)', flush=True)
    print(f'  sector conocido: {mapped.mean() * 100:.1f}% de los candidatos ({len(gics)} valores con GICS de Wikipedia + sectores ETF)', flush=True)
    grp = EV['grp'][rows]
    for g in ('us_large', 'us_mid', 'us_small', 'thematic', 'etf', 'eu', 'crypto', 'futures'):
        if g in RS.GROUPS:
            m = grp == RS.GROUPS.index(g)
            if m.any():
                print(f'    {g:9s}: {mapped[m].mean() * 100:5.1f}% con sector · {picked[m & mapped].mean() * 100 if (m & mapped).any() else 0:4.1f}% de ellos en sector recomendado', flush=True)
    print(f'  sectores recomendados por mes (media): {np.mean([bin(v & (EQ_MASK | GOLD_MASK)).count("1") for v in pm.values()]):.1f} de {len(EQ)} de renta variable + oro', flush=True)

    vi = EV['variants'].index(ST.vname(4.0))
    pnl = EV['pnl'][rows, vi]
    base_m = mapped & have & np.isfinite(pnl)
    rng = np.random.default_rng(7)

    print('\n  A. CANDIDATOS CON SECTOR CONOCIDO (salida del plan Equilibrado: RSI(2)>70 · stop 4×ATR · ≤10 sesiones)')
    print('    ' + 'grupo'.ljust(34) + 'n'.rjust(7) + 'acierto'.rjust(9) + 'media'.rjust(9) + 'PF'.rjust(6) + '  t(día)')
    print(_line('todos (sector conocido)', pnl[base_m], day[base_m], sym[base_m]))
    print(_line('en sector recomendado', pnl[base_m & picked], day[base_m & picked], sym[base_m & picked]))
    print(_line('en sector NO recomendado', pnl[base_m & ~picked], day[base_m & ~picked], sym[base_m & ~picked]))
    for lab, gs in (('acciones EE.UU. (large+mid+small)', ('us_large', 'us_mid', 'us_small')), ('ETF', ('etf',))):
        gm = np.isin(grp, [RS.GROUPS.index(g) for g in gs])
        print(_line(f'{lab}: recomendado', pnl[base_m & picked & gm], day[base_m & picked & gm], sym[base_m & picked & gm]))
        print(_line(f'{lab}: no recomendado', pnl[base_m & ~picked & gm], day[base_m & ~picked & gm], sym[base_m & ~picked & gm]))
    act = pnl[base_m & picked].mean() if (base_m & picked).any() else np.nan
    reps = 300
    null = permutation_null(mi[base_m], emask[base_m], pnl[base_m], pm, reps, rng)
    nn = null[np.isfinite(null)]
    print(f'    contraste de permutaciones ({reps} calendarios al azar con el mismo número de sectores por mes): media real {act:+.3f}% frente a azar '
          f'{nn.mean():+.3f}% (desv {nn.std():.3f}) → la real supera al {(nn < act).mean() * 100:.0f}% de los calendarios al azar', flush=True)
    yrs = sorted(set(year[base_m]))
    print('    por año, recomendado / no recomendado (media %, n): ' + ' · '.join(
        f"{Y}: {pnl[base_m & picked & (year == Y)].mean():+.2f}/{pnl[base_m & ~picked & (year == Y)].mean():+.2f} ({int((base_m & picked & (year == Y)).sum())}/{int((base_m & ~picked & (year == Y)).sum())})"
        for Y in yrs if (base_m & picked & (year == Y)).sum() >= 30 and (base_m & ~picked & (year == Y)).sum() >= 30), flush=True)
    for Y0 in (2019, 2022):
        mm = base_m & (year >= Y0)
        print(f'    desde {Y0}: recomendado {pnl[mm & picked].mean():+.2f}% (n={int((mm & picked).sum())}) · no recomendado {pnl[mm & ~picked].mean():+.2f}% (n={int((mm & ~picked).sum())})')

    # ── B y C: plan completo ──
    exp: Dict = {}
    RS.build_plan(EV, rows, FL, names, ST.plan_of(4.0), last_day, spy=data2['SPY'], export=exp, verbose=False)
    sel = exp['sel']
    print('\n  B. ALERTAS DEL PLAN DESPLEGADO (top-3 por día, retraining incluido) FILTRADAS DESPUÉS: SOLO LAS DE SECTOR RECOMENDADO')
    print('    ' + 'grupo'.ljust(34) + 'n'.rjust(7) + 'acierto'.rjust(9) + 'media'.rjust(9) + 'PF'.rjust(6) + '  t(día)')
    s0 = sel & mapped & have & np.isfinite(pnl)
    print(_line('alertas con sector conocido', pnl[s0], day[s0], sym[s0]))
    print(_line('… en sector recomendado', pnl[s0 & picked], day[s0 & picked], sym[s0 & picked]))
    print(_line('… en sector NO recomendado', pnl[s0 & ~picked], day[s0 & ~picked], sym[s0 & ~picked]))
    print(_line('alertas SIN sector conocido', pnl[sel & ~mapped & np.isfinite(pnl)], day[sel & ~mapped & np.isfinite(pnl)], sym[sel & ~mapped & np.isfinite(pnl)]))
    print(f'    alertas totales {int(sel.sum())} · con sector {int((sel & mapped).sum())} · recomendadas {int((sel & mapped & picked).sum())}')

    print('\n  C. SISTEMA ENTERO REENTRENADO CON EL FONDO RESTRINGIDO (modelo y política recalculados)')
    print('    ' + 'versión'.ljust(40) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'PF'.rjust(6) + 'pérd.'.rjust(8) + 'alfa vs S&P'.rjust(14) + '   cartera 10×10 %: anual · caída · Sharpe · peor mes')
    arms = (('todo el fondo', np.ones(len(rows), bool)), ('solo sector conocido', mapped & have),
            ('sector conocido Y recomendado', mapped & have & picked), ('sector conocido y NO recomendado', mapped & have & ~picked),
            ('recomendado o sin sector', ~mapped | (have & picked)))
    for lab, keep in arms:
        try:
            pl = _quiet(RS.build_plan, EV, rows[keep], FL[keep], names, ST.plan_of(4.0), last_day, spy=data2['SPY'])
            o, vs, p10 = pl['stats']['oos'], pl['stats'].get('vsSpy', {}), pl['stats']['portfolio']['10x10']
            print('    ' + lab.ljust(40) + f"{o['n']:6.0f}{o['wr']:8.1f}%{o['mean']:+7.2f}%{o['pf']:6.2f}{o['avg_loss']:+7.2f}%{vs.get('alpha', float('nan')):+8.2f}% (t={vs.get('alphaT', float('nan')):.1f})   "
                  f"{p10['cagr']:+6.1f}% · {p10['dd']:4.1f}% · {p10['sharpe']:.2f} · {p10['worstMonth']:+5.1f}%", flush=True)
        except Exception as exc:                                              # política sin grupos suficientes, etc.
            print('    ' + lab.ljust(40) + f'  sin plan ({type(exc).__name__}: {exc})', flush=True)
    sys.stdout.flush()
