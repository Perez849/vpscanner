"""
Laboratorio: para cada valor detecta los patrones (engine), calcula su plan (plan) y lo vuelca como JSON que dibuja la web.

Salida (carpeta data/):
  charts/<VALOR>.json   barras (últimas NB), medias, soportes y resistencias, resumen y lista de patrones con geometría (puntos, rectas, polígonos, zonas, medidas),
                        plan (entrada, stop, objetivo, T1, R:R, obstáculos) y resultado simulado; las compresiones sin romper van como «formándose» con dos escenarios
  index.json            resumen por valor (precio, minigráfico, patrones activos) para el radar
  catalog.json          catálogo de patrones (nombre, descripción, regla del plan)
"""
from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import engine, plan as P, ta
from .catalog import CATALOG, FAMILIES, HQ
from . import shapes
from .shapes import at

NB = 340             # barras exportadas (el resto de la historia solo sirve de arranque a medias e indicadores)
ACTIVE = 12          # un patrón está «activo» si se completó en las últimas 12 sesiones y aún no ha resuelto
SPARK = 60

FORM_LABEL = {'channel_up': 'Canal ascendente', 'channel_dn': 'Canal descendente', 'sym_tri': 'Triángulo simétrico', 'asc_triangle': 'Triángulo ascendente', 'desc_triangle': 'Triángulo descendente', 'rising_wedge': 'Cuña ascendente',
              'falling_wedge': 'Cuña descendente', 'rect': 'Rectángulo'}
FORM_UP = {'channel_up': 'channel_up_up', 'channel_dn': 'channel_dn_up', 'sym_tri': 'sym_tri_up', 'asc_triangle': 'asc_triangle', 'falling_wedge': 'falling_wedge', 'rect': 'rect_up', 'desc_triangle': 'sym_tri_up', 'rising_wedge': 'sym_tri_up'}
FORM_DN = {'channel_up': 'channel_up_dn', 'channel_dn': 'channel_dn_dn', 'sym_tri': 'sym_tri_dn', 'desc_triangle': 'desc_triangle', 'rising_wedge': 'rising_wedge', 'rect': 'rect_dn', 'asc_triangle': 'sym_tri_dn', 'falling_wedge': 'sym_tri_dn'}
FORM_BIAS = {'channel_up': 1, 'channel_dn': -1, 'asc_triangle': 1, 'falling_wedge': 1, 'desc_triangle': -1, 'rising_wedge': -1}


# ── geometría que dibuja la web ───────────────────────────────────────────
def annotate(name: str, g: Dict, F: Dict[str, np.ndarray], i: int) -> Dict:
    """Puntos, rectas, polígonos, zonas y medidas de un patrón completado en la barra i (índices de barra absolutos)."""
    h, l, c = F['h'], F['l'], F['c']
    lines: List[Dict] = []
    pts: List[List] = []
    polys: List[Dict] = []
    zones: List[Dict] = []
    ann: List[Dict] = []

    def ln(a, b, k='trend', dash=False, n=None):
        d_ = {'a': [int(a[0]), P.r(a[1])], 'b': [int(b[0]), P.r(b[1])], 'k': k, 'dash': dash}
        if n:
            d_['n'] = int(n)
        lines.append(d_)

    def pt(p, tag=''):
        pts.append([int(p[0]), P.r(p[1]), tag])

    def ext(p1, p2, x):
        return p2[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (x - p2[0])

    if name in ('double_bottom', 'double_top', 'triple_bottom', 'triple_top'):
        pp, mm = g['pts'], g['mids']
        seq = []
        for k, p in enumerate(pp):
            seq.append(p)
            if k < len(mm):
                seq.append(mm[k])
        polys.append({'pts': [[int(x), P.r(y)] for x, y in seq], 'k': 'zig'})
        for k, p in enumerate(pp):
            pt(p, str(k + 1))
        for m in mm:
            pt(m, '')
        ln((pp[0][0], g['neck']), (i, g['neck']), 'neck', True)
        e_ = min(pp, key=lambda q: q[1]) if name.endswith('bottom') else max(pp, key=lambda q: q[1])
        ann.append({'k': 'height', 'x': int(pp[-1][0]), 'y0': P.r(g['neck']), 'y1': P.r(e_[1]), 't': 'altura'})
    elif name in ('hs_top', 'hs_inverse'):
        pts_ = [g['ls'], g['t1'], g['hd'], g['t2'], g['rs']]
        polys.append({'pts': [[int(p[0]), P.r(p[1])] for p in pts_], 'k': 'zig'})
        for p, t in zip(pts_, ['Hombro', '', 'Cabeza', '', 'Hombro']):
            pt(p, t)
        ln(g['t1'], (i, ext(g['t1'], g['t2'], i)), 'neck', True)
        ann.append({'k': 'height', 'x': int(g['hd'][0]), 'y0': P.r(ext(g['t1'], g['t2'], g['hd'][0])), 'y1': P.r(g['hd'][1]), 't': 'altura'})
    elif name in ('asc_triangle', 'desc_triangle', 'sym_tri_up', 'sym_tri_dn', 'rising_wedge', 'falling_wedge', 'rect_up', 'rect_dn') or name.startswith(('forming_', 'channel_')):
        U, D, x0 = g['U'], g['D'], g['x0']
        ln((x0, at(U, x0)), (i, at(U, i)), 'upper', False, len(g['tu']))
        ln((x0, at(D, x0)), (i, at(D, i)), 'lower', False, len(g['tl']))
        polys.append({'pts': [[int(x0), P.r(at(U, x0))], [int(i), P.r(at(U, i))], [int(i), P.r(at(D, i))], [int(x0), P.r(at(D, x0))]], 'k': 'fill'})
        for p in list(g['tu']) + list(g['tl']):
            pt(p, '')
        ap = g.get('apex')
        if ap is not None and i < ap <= i + 45:
            ln((i, at(U, i)), (ap, at(U, ap)), 'ext', True)
            ln((i, at(D, i)), (ap, at(D, ap)), 'ext', True)
            pt((ap, at(U, ap)), 'ápice')
        ann.append({'k': 'height', 'x': int(x0), 'y0': P.r(at(U, x0)), 'y1': P.r(at(D, x0)), 't': 'anchura' if name.startswith('channel_') else 'base'})
    elif name in ('trendline_up', 'trendline_dn'):
        L, tch = g['L'], g['tch']
        ln((tch[0][0], at(L, tch[0][0])), (i, at(L, i)), 'trend', False, len(tch))
        for k, p in enumerate(tch):
            pt(p, str(k + 1))
    elif name in ('support_bounce', 'resistance_reject'):
        ln((g['touches'][0][0], g['level']), (i, g['level']), 'level')
        for p in g['touches']:
            pt(p, '')
    elif name == 'vcp':
        zz = []
        for k, hi_ in enumerate(g['highs']):
            zz.append([hi_[0], P.r(hi_[1])])
            zz.append([g['lows'][k][0], P.r(g['lows'][k][1])])
            pt(hi_, f'{k + 1}')
        polys.append({'pts': zz + [[min(i, len(c) - 1), P.r(c[min(i, len(c) - 1)])]], 'k': 'zig'})
        ln((g['highs'][0][0], g['highs'][-1][1]), (i, g['highs'][-1][1]), 'neck', True)
    elif name == 'cup_handle':
        cf = g['fit']
        xs = np.arange(g['W'])
        ys = np.polyval(cf, xs)
        step = max(g['W'] // 28, 1)
        polys.append({'pts': [[int(g['w0'] + x), P.r(y)] for x, y in zip(xs[::step], ys[::step])] + [[int(g['w0'] + xs[-1]), P.r(ys[-1])]], 'k': 'curve'})
        rim = max(g['left'][1], g['right'][1])
        ln((g['left'][0], rim), (i, rim), 'neck', True)
        zones.append({'i0': g['right'][0], 'i1': i, 'lo': P.r(g['handle'][1]), 'hi': P.r(g['right'][1]), 'k': 'box'})
        pt(g['left'], 'Borde'); pt(g['bottom'], 'Fondo'); pt(g['right'], 'Borde'); pt(g['handle'], 'Asa')
        ann.append({'k': 'height', 'x': int(g['bottom'][0]), 'y0': P.r(rim), 'y1': P.r(g['bottom'][1]), 't': 'profundidad'})
    elif name in ('diamond_bull', 'diamond_bear'):
        hi_, lo_ = g['hi'], g['lo']
        polys.append({'pts': [[p[0], P.r(p[1])] for p in hi_] + [[p[0], P.r(p[1])] for p in lo_[::-1]], 'k': 'fill'})
        polys.append({'pts': [[p[0], P.r(p[1])] for p in hi_], 'k': 'zig'})
        polys.append({'pts': [[p[0], P.r(p[1])] for p in lo_], 'k': 'zig'})
    elif name in ('bull_flag', 'bear_flag', 'bull_pennant', 'bear_pennant'):
        ps, pe = g['ps'], g['pe']
        ln(ps, pe, 'pole')
        z0 = g['z0']
        seg_h, seg_l = h[z0:i], l[z0:i]
        xs = np.arange(z0, i)
        if len(xs) >= 3:
            uh, ul = np.polyfit(xs, seg_h, 1), np.polyfit(xs, seg_l, 1)
            if not g.get('pen'):                                       # bandera: canal paralelo que contiene toda la consolidación
                sl_ = float(np.polyfit(xs, c[z0:i], 1)[0])
                uh, ul = np.array([sl_, float((seg_h - sl_ * xs).max())]), np.array([sl_, float((seg_l - sl_ * xs).min())])
            ln((z0, np.polyval(uh, z0)), (i - 1, np.polyval(uh, i - 1)), 'upper')
            ln((z0, np.polyval(ul, z0)), (i - 1, np.polyval(ul, i - 1)), 'lower')
            polys.append({'pts': [[z0, P.r(np.polyval(uh, z0))], [i - 1, P.r(np.polyval(uh, i - 1))], [i - 1, P.r(np.polyval(ul, i - 1))], [z0, P.r(np.polyval(ul, z0))]], 'k': 'fill'})
        pt(ps, 'Asta'); pt(pe, '')
        ann.append({'k': 'label', 'x': int((ps[0] + pe[0]) / 2), 'y': P.r((ps[1] + pe[1]) / 2), 't': f'asta {abs(pe[1] - ps[1]):.2f} · {abs(pe[1] / ps[1] - 1) * 100:.1f} %'})
    elif name in ('island_bull', 'island_bear'):
        zones.append({'i0': g['a'] - 1, 'i1': i, 'lo': P.r(g['lo']), 'hi': P.r(g['hi']), 'k': 'box'})
        ln((g['a'] - 1, g['gap_from']), (i, g['gap_from']), 'level', True)
    elif name.startswith('donch'):
        ln((max(i - 40, 0), g['level']), (i, g['level']), 'level', True)
    elif name.startswith(('box_', 'base_near')):
        zones.append({'i0': g['i0'], 'i1': i - 1, 'lo': P.r(g['lo']), 'hi': P.r(g['hi']), 'k': 'box'})
        if name == 'base_near_high':
            ln((max(i - 60, 0), g['hi52']), (i, g['hi52']), 'level', True)
    elif name.startswith(('bb_squeeze', 'atr_squeeze')):
        zones.append({'i0': i - 20, 'i1': i - 1, 'lo': P.r(g['lo']), 'hi': P.r(g['hi']), 'k': 'box'})
    elif name in CATALOG and CATALOG[name].fam == 'velas':
        m = max(CATALOG[name].bars, 1)
        zones.append({'i0': i - m + 1, 'i1': i, 'lo': P.r(l[i - m + 1:i + 1].min()), 'hi': P.r(h[i - m + 1:i + 1].max()), 'k': 'candle'})
    return {'pts': pts, 'lines': lines, 'polys': polys, 'zones': zones, 'ann': ann}


def _xmin(geo: Dict, default: int) -> int:
    xs = [p[0] for p in geo['pts']] + [x for L_ in geo['lines'] for x in (L_['a'][0], L_['b'][0])] + [p[0] for pl in geo['polys'] for p in pl['pts']] + [z['i0'] for z in geo['zones']]
    return min(xs) if xs else default


def _shift(geo: Dict, s0: int) -> None:
    for p in geo['pts']:
        p[0] -= s0
    for L_ in geo['lines']:
        L_['a'][0] -= s0; L_['b'][0] -= s0
    for pl in geo['polys']:
        for p in pl['pts']:
            p[0] -= s0
    for z in geo['zones']:
        z['i0'] -= s0; z['i1'] -= s0
    for a in geo['ann']:
        a['x'] -= s0


FAM_RANK = {'figuras': 0, 'rupturas': 1, 'directrices': 2, 'rebotes': 3, 'velas': 4}


def group_trades(items: List[Dict], last: int) -> Dict:
    """Una operación por valor: las señales de un mismo valor se agrupan en operaciones para no contar varias veces el mismo movimiento.
    · La primera señal (si coinciden varias en la misma barra, la de mayor rango: figura > ruptura > directriz > rebote > vela, y a igualdad la de mayor calidad) abre la operación
      y su plan es el que se simula. Si la señal queda fuera de niveles en la apertura siguiente (resultado «na») no abre nada.
    · Mientras esa operación sigue abierta, las señales del mismo sentido son «confirmaciones» (no suman otra operación) y las del sentido contrario «contrarias» (aviso de conflicto).
    · Cada operación se cuenta una sola vez, gane o pierda; gana quien termina con ganancia neta, llegue o no al objetivo.
    `last` es la última barra (relativa) de la ventana exportada; una operación aún abierta bloquea hasta ahí."""
    done = [it for it in items if it['state'] == 'done' and it.get('out') and it['dir'] != 0]
    done.sort(key=lambda it: (it['i'], 1 if it['key'].startswith('consenso') else 0, FAM_RANK.get(it['fam'], 9), -(it['q'] if it['q'] is not None else -1), it['key']))
    out: List[Dict] = []
    cur: Optional[Dict] = None
    for it in done:
        o = it['out']
        if o['res'] == 'na':
            it['trade'] = {'role': 'na'}
            continue
        e = it['i'] + 1
        if cur is not None and e <= cur['exit']:
            same = it['dir'] == cur['dir']
            cur['conf' if same else 'contra'].append(it['label'])
            it['trade'] = {'role': 'confirma' if same else 'contraria', 'with': cur['id']}
            continue
        exit_ = o['xi'] if o['res'] in ('target', 'stop', 'timeout') and o.get('xi') is not None else last
        cur = {'id': it['id'], 'key': it['key'], 'label': it['label'], 'dir': it['dir'], 'fam': it['fam'], 'i': it['i'], 'res': o['res'], 'net': o.get('net'), 'r': o.get('r'), 
               'mfe': o.get('mfe'), 'prog': o.get('prog'), 'exit': int(exit_), 'conf': [], 'contra': []}
        out.append(cur)
        it['trade'] = {'role': 'lider'}
    closed = [t for t in out if t['res'] in ('target', 'stop', 'timeout') and t['net'] is not None]

    def tally(ts: List[Dict]) -> Dict:
        nets = [t['net'] for t in ts]
        rs = [t['r'] for t in ts if t['r'] is not None]
        return {'n': len(ts), 'won': sum(1 for x in nets if x > 0), 'lost': sum(1 for x in nets if x <= 0), 'tgt': sum(1 for t in ts if t['res'] == 'target'), 'stp': sum(1 for t in ts if t['res'] == 'stop'),
                'tmo': sum(1 for t in ts if t['res'] == 'timeout'), 'tmo_won': sum(1 for t in ts if t['res'] == 'timeout' and t['net'] > 0), 'lk2': sum(1 for t in ts if t.get('level', 0) >= 2), 'mean_net': round(float(np.mean(nets)), 3) if nets else None,
                'sum_net': round(float(np.sum(nets)), 2) if nets else 0.0, 'mean_r': round(float(np.mean(rs)), 3) if rs else None,
                'avg_win': round(float(np.mean([x for x in nets if x > 0])), 3) if any(x > 0 for x in nets) else None,
                'avg_loss': round(float(np.mean([x for x in nets if x <= 0])), 3) if any(x <= 0 for x in nets) else None}

    return {'list': out, 'open': sum(1 for t in out if t['res'] == 'open'), 'conflicts': sum(1 for t in out if t['contra']), 'all': tally(closed),
            'by_fam': {f: tally([t for t in closed if t['fam'] == f]) for f in sorted({t['fam'] for t in closed})}}


# ── análisis de un valor ──────────────────────────────────────────────────
def analyze(sym: str, bars: Dict[str, np.ndarray], name: str = '', nb: int = NB) -> Dict:
    F = ta.prepare(bars)
    n = len(F['c'])
    s0 = max(n - nb, 0)
    events, forming = engine.scan(F)
    zs = P.zones(F, s0)
    items: List[Dict] = []
    for e in events:
        key, i, g = e['name'], e['i'], e['g']
        if i < s0 or key not in CATALOG:
            continue
        pat = CATALOG[key]
        d = pat.dir
        try:
            geo = annotate(key, g, F, i)
            stop, tgt = P.rule(key, g, F, i, d)
        except Exception as exc:                          # un patrón mal formado no debe romper el valor
            print(f'  [{sym}] {key}@{i}: {type(exc).__name__}: {exc}', flush=True)
            continue
        if _xmin(geo, s0) < s0:                           # la figura empieza antes de la ventana exportada: no se dibuja cortada
            continue
        entry = float(F['c'][i])
        bx = P.box(F, i, d, entry, stop, tgt)
        stop, lock, stop_s, tgt_c = bx.pop('_stop'), bx.pop('_target'), bx.pop('_stop_s'), bx.pop('_target_c')
        note = pat.rule.replace('Objetivo genérico de 2R con stop de', 'Stop de').replace('Objetivo genérico de 2R. ', '')   # el objetivo del catálogo es del plan clásico
        if bx['stop_cap']:
            note += f' El stop estructural quedaba a {bx["risk_s_pct"]:.1f} % de la entrada (más de {P.MAX_RISK_ATR:g} ATR): se ha acortado a {bx["risk_pct"]:.1f} %.'
        if bx['stop_min']:
            note += f' El stop estructural quedaba a menos de {P.MIN_STOP:g} ATR: se ha ampliado a {P.MIN_STOP:g} ATR.'
        note += f' Escalera del operador: al cerrar una sesión por encima de +{P.LOCK_PCT:g} % el stop sube a +{P.LOCK_PCT:g} % y se deja correr; al cerrar por encima de +{P.LOCK2_PCT:g} % el stop sube a +{P.LOCK2_PCT:g} % y persigue al mejor cierre a {P.TRAIL2_ATR:g} ATR; sin objetivo fijo (la medida clásica daba +{bx["reward_c_pct"]:.1f} %).'
        if bx['risk_hi']:
            note += f' Aviso: el stop queda a {bx["risk_pct"]:.1f} % del precio, más de {P.MAX_RISK_PCT:g} %: valor muy volátil.'
        elif bx['rr_low']:
            note += f' Aviso: el stop ({bx["risk_pct"]:.1f} %) está más lejos que el +{P.LOCK_PCT:g} % que se quiere asegurar.'
        if bx['spent']:
            note += ' Aviso: la ruptura ya recorrió casi toda la medida clásica de la figura (el objetivo clásico queda a menos de 1 ATR).'
        out = P.outcome_ladder(F, i, d, stop)
        out_s = P.outcome(F, i, d, stop_s, tgt_c)                                  # el plan clásico sin límites, solo como referencia
        active = (n - 1 - i) <= ACTIVE and out['res'] == 'open'
        if active:
            bx['obs'] = P.obstacles(zs, entry, tgt_c)
        _shift(geo, s0)
        for o_ in (out, out_s):
            if o_ and 'xi' in o_:
                o_['xi'] -= s0
        items.append({'id': f'{key}@{i - s0}', 'key': key, 'label': pat.label, 'dir': d, 'fam': pat.fam, 'i': i - s0, 'state': 'done', 'q': e['q'], 'new': bool(i == n - 1), **geo,
                      'plan': bx, 'note': note, 'out': out, 'out_s': out_s, 'active': bool(active)})
    seen = {}
    for it in items:                                      # varias rupturas de Donchian el mismo día y en el mismo sentido: se queda la de mayor periodo
        if it['key'].startswith('donch'):
            k = (it['i'], it['dir'])
            per = int(it['key'][5:].split('_')[0])
            if k not in seen or per > seen[k][0]:
                seen[k] = (per, it)
    items = [it for it in items if not it['key'].startswith('donch') or seen[(it['i'], it['dir'])][1] is it]
    for it in items:                                      # otros patrones del mismo sentido que saltan en las 3 barras vecinas (no son confirmaciones independientes)
        it['conf'] = [o['label'] for o in items if o is not it and o['dir'] == it['dir'] and it['dir'] != 0 and abs(o['i'] - it['i']) <= 3 and o['key'] != it['key']][:4]
    trades = group_trades(items, n - s0 - 1)
    last = float(F['c'][-1])
    for st in forming:
        a_ = float(F['atr'][n - 1])
        if st['type'] == 'cmp':
            if st['kind'] not in FORM_LABEL:
                continue
            i = n - 1
            g = dict(U=st['U'], D=st['D'], tu=st['tu'], tl=st['tl'], x0=st['x0'], apex=st['apex'], kind=st['kind'])
            geo = annotate('forming_' + st['kind'], g, F, i)
            if _xmin(geo, s0) < s0:
                continue
            sc = {}
            bias = FORM_BIAS.get(st['kind'], 0)
            lv = {'up': at(st['U'], i + 1) + 0.15 * a_, 'dn': at(st['D'], i + 1) - 0.15 * a_}
            for side, d, key in (('up', 1, FORM_UP[st['kind']]), ('dn', -1, FORM_DN[st['kind']])):
                stop, tgt = P.rule(key, g, F, i, d, entry=lv[side])
                bx = P.box(F, i, d, lv[side], stop, tgt)
                for k_ in ('_stop', '_target', '_stop_s', '_target_c'):
                    bx.pop(k_, None)
                bx['key'] = key
                bx['label'] = 'rompe contra el sesgo habitual de la figura' if bias * d < 0 else CATALOG[key].label
                bx['note'] = CATALOG[key].rule
                bx['obs'] = P.obstacles(zs, lv[side], bx['target'])
                sc[side] = bx
            dist = min(lv['up'] - last, last - lv['dn']) / a_
            _shift(geo, s0)
            items.append({'id': f'forming_{st["kind"]}@{i - s0}', 'key': 'forming_' + st['kind'], 'label': FORM_LABEL[st['kind']], 'dir': 0, 'fam': 'figuras', 'i': i - s0, 'state': 'forming',
                          'q': shapes.compression_quality(st, 0.25, 1.15), 'bias': bias, **geo, 'plan': None, 'sc': sc,
                          'trig': {'up': P.r(lv['up']), 'dn': P.r(lv['dn']), 'dist': round(max(dist, 0.0), 2), 'pct': round(max(dist, 0.0) * a_ / last * 100, 2)},
                          'note': 'Aún no ha roto: el cierre sigue dentro de las dos rectas. Se opera la ruptura (cierre fuera de la recta) y el sentido no está decidido.'
                                  + (' Sesgo habitual: ' + ('alcista' if bias > 0 else 'bajista') + '.' if bias else ''),
                          'out': {'res': 'forming', 'bars': 0, 'ret': None}, 'out_s': None, 'active': True, 'conf': []})
        else:
            fk, d, g = st['name'], st['d'], st['g']
            if any(it['key'] == fk and it['dir'] == d and it['active'] for it in items if it['state'] == 'done'):      # la misma figura ya rota y aún viva: no se anuncia otra igual
                continue
            i = n - 1
            geo = annotate(fk, g, F, n)                      # la barra virtual n lleva el cuello o la recta hasta el próximo cierre
            if _xmin(geo, s0) < s0:
                continue
            trig, entry = st['trig'], st['trig'] + d * 0.15 * a_
            if not any(L_['k'] == 'neck' for L_ in geo['lines']):
                x_ = max(_xmin(geo, n - 1), n - 60)
                geo['lines'].append({'a': [int(x_), P.r(trig)], 'b': [int(n), P.r(trig)], 'k': 'neck', 'dash': True})
            stop, tgt = P.rule(fk, g, F, i, d, entry=entry)
            bx = P.box(F, i, d, entry, stop, tgt)
            for k_ in ('_stop', '_target', '_stop_s', '_target_c'):
                bx.pop(k_, None)
            bx['key'] = fk
            bx['label'] = CATALOG[fk].label
            bx['note'] = CATALOG[fk].rule
            bx['obs'] = P.obstacles(zs, entry, bx['target'])
            _shift(geo, s0)
            items.append({'id': f'forming_{fk}@{i - s0}', 'key': 'forming_' + fk, 'label': CATALOG[fk].label, 'dir': d, 'fam': 'figuras', 'i': i - s0, 'state': 'forming', 'q': st['q'], **geo,
                          'plan': None, 'sc': {'up' if d > 0 else 'dn': bx}, 'trig': {'level': P.r(trig), 'dist': round(st['dist'], 2), 'pct': round(abs(trig / last - 1) * 100, 2)},
                          'note': f'Figura dibujada pero sin romper: se activa con un cierre {"por encima" if d > 0 else "por debajo"} de {P.r(trig)} (a {st["dist"]:.1f} ATR del cierre). '
                                  'Si el precio se gira antes, la figura se invalida; el plan usa la entrada en ese nivel + 0,15 ATR.',
                          'out': {'res': 'forming', 'bars': 0, 'ret': None}, 'out_s': None, 'active': True, 'conf': []})
    items.sort(key=lambda x: x['i'])
    sl = slice(s0, n)
    c = F['c']
    summ = {'close': P.r(c[-1]), 'atr': P.r(F['atr'][-1]), 'atrp': round(float(F['atrp'][-1]), 2), 'ret5': round(float(F['ret5'][-1]), 2), 'ret20': round(float(F['ret20'][-1]), 2),
            'rsi14': round(float(F['rsi14'][-1]), 1), 'rsi2': round(float(F['rsi2'][-1]), 1), 'dist200': round(float(F['dist200'][-1]), 2) if np.isfinite(F['dist200'][-1]) else None,
            'pos52': round(float(F['pos52'][-1]), 3) if np.isfinite(F['pos52'][-1]) else None}
    day = lambda x: int(x // 86400)
    return {'sym': sym, 'name': name, 'asof': day(bars['t'][-1]), 'n': n - s0,
            'bars': {'t': [day(x) for x in bars['t'][sl]], 'o': [P.r(x) for x in bars['o'][sl]], 'h': [P.r(x) for x in bars['h'][sl]], 'l': [P.r(x) for x in bars['l'][sl]],
                     'c': [P.r(x) for x in bars['c'][sl]], 'v': [int(x) for x in bars['v'][sl]]},
            'sma50': [None if not np.isfinite(x) else P.r(x) for x in F['sma50'][sl]], 'sma200': [None if not np.isfinite(x) else P.r(x) for x in F['sma200'][sl]],
            'zones': zs, 'patterns': items, 'summary': summ, 'trades': trades}


def index_row(res: Dict) -> Dict:
    c = res['bars']['c']
    act = [p for p in res['patterns'] if p['active']]
    act.sort(key=lambda p: (-(p['state'] == 'forming'), -(p['q'] or 0)))          # primero lo que se está formando, luego por calidad
    return {'sym': res['sym'], 'name': res['name'], 'close': res['summary']['close'], 'chg': round((c[-1] / c[-2] - 1) * 100, 2) if len(c) > 1 else 0.0, 'atrp': res['summary']['atrp'],
            'rsi2': res['summary']['rsi2'], 'n': len(res['patterns']), 'spark': c[-SPARK:],
            'new': sum(1 for p in act if p.get('new')),
            'active': [{'id': p['id'], 'label': p['label'], 'dir': p['dir'], 'fam': p['fam'], 'q': p['q'], 'st': p['state'], 'rr': (p['plan'] or {}).get('rr'), 'new': bool(p.get('new')),
                        'dist': (p.get('trig') or {}).get('dist')} for p in act[:12]]}


def write_catalog(out_dir: str) -> None:
    cat = {k: {'label': p.label, 'dir': p.dir, 'fam': p.fam, 'desc': p.desc, 'rule': p.rule, 'bars': p.bars} for k, p in CATALOG.items()}
    json.dump({'patterns': cat, 'families': FAMILIES, 'hq': HQ}, open(os.path.join(out_dir, 'catalog.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))


def run(symbols: List[str], out_dir: str, rng: str = '3y', names: Optional[Dict[str, str]] = None, raw_dir: Optional[str] = None) -> Dict:
    """Descarga los valores, los analiza y escribe data/charts/*.json, index.json y catalog.json."""
    from . import data
    bars, failed = data.fetch_many(symbols, rng=rng, workers=4, verbose=True)
    os.makedirs(os.path.join(out_dir, 'charts'), exist_ok=True)
    if raw_dir:                                           # series completas, para afinar detectores sin red
        os.makedirs(raw_dir, exist_ok=True)
        for s_, b_ in bars.items():
            json.dump({k: [float(x) for x in b_[k]] for k in ('t', 'o', 'h', 'l', 'c', 'v')}, open(os.path.join(raw_dir, f'{s_}.json'), 'w'), separators=(',', ':'))
    index, missing = [], []
    for s in symbols:
        b = bars.get(s)
        if b is None or len(b['c']) < NB + 60:
            print(f'  {s}: sin datos suficientes', flush=True)
            missing.append(s)
            continue
        res = analyze(s, b, name=(names or {}).get(s, b.get('name', '')))
        json.dump(res, open(os.path.join(out_dir, 'charts', f'{s}.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
        index.append(index_row(res))
    keep = {f'{x["sym"]}.json' for x in index}
    for f in os.listdir(os.path.join(out_dir, 'charts')):         # un valor quitado de la lista no deja su gráfico antiguo en el repositorio
        if f.endswith('.json') and f not in keep:
            os.remove(os.path.join(out_dir, 'charts', f))
    meta = {'generated': int(time.time()), 'asof': int(max(bars[s]['t'][-1] // 86400 for s in bars)) if bars else 0, 'symbols': index, 'missing': missing}
    json.dump(meta, open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    write_catalog(out_dir)
    print(f'  laboratorio: {len(index)} valores → {out_dir} (fallidos {failed})', flush=True)
    if len(index) < 0.5 * len(symbols):
        raise SystemExit(f'solo {len(index)} de {len(symbols)} valores con datos: no se dan por buenos')
    return meta
