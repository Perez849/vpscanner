"""Ideas por coyuntura: marcador (data/ideas_track.json) y registro de casos propios (data/ideas_casos.json) a partir de data/ideas.json.

Cada idea abierta o cerrada se compara con el SPY desde el cierre del día de entrada; las que están en vigilancia, desde el precio de referencia del día en que se dieron
(solo como referencia: todavía no se ha entrado). Las ideas las escribe el analista en cada sesión (ver IDEAS.md); este módulo solo pone los precios."""
from __future__ import annotations
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import data


def fetch(sym: str):
    """Barras de un año. Los valores de otras bolsas llevan sufijo con punto (IDR.MC), que `data.fetch_one` convertiría en guion: esos se piden tal cual."""
    if '.' not in sym:
        return data.fetch_one(sym, '1y')
    url = data.URL.format(host=data.HOSTS[0], sym=urllib.parse.quote(sym, safe='.'), span=data.span('1y'))
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': data.UA, 'Accept': 'application/json'}), timeout=25) as r:
            return data.parse(json.loads(r.read().decode('utf-8')))
    except (OSError, ValueError):
        return None


def closes(sym: str) -> Dict[str, float]:
    b = fetch(sym)
    if b is None:
        return {}
    return {datetime.fromtimestamp(int(t), timezone.utc).strftime('%Y-%m-%d'): float(c) for t, c in zip(b['t'], b['c'])}


def on_or_before(cl: Dict[str, float], day: Optional[str]) -> Optional[tuple]:
    ks = [k for k in sorted(cl) if day is None or k <= day]
    return (ks[-1], cl[ks[-1]]) if ks else None


def symbols(idea: Dict) -> List[str]:
    return [v['sym'] if isinstance(v, dict) else v for v in idea.get('valores', [])]


def score(idea: Dict, cl: Dict[str, Dict[str, float]]) -> Dict:
    """Rendimiento de cada valor y del SPY desde la entrada (o la referencia, si está en vigilancia) hasta la salida o el último cierre; `media` es la de los valores a partes iguales."""
    ent, sal = idea.get('entrada'), idea.get('salida')
    por: Dict[str, Dict] = {}
    hasta = None
    for s in symbols(idea) + ['SPY']:
        c = cl.get(s, {})
        a = on_or_before(c, ent) if ent else None
        p0 = a[1] if a else (idea.get('ref') or {}).get(s)
        b = on_or_before(c, sal)
        if p0 and b:
            por[s] = {'ref': round(p0, 2), 'ahora': round(b[1], 2), 'pct': round(100 * (b[1] / p0 - 1), 2)}
            hasta = b[0]
    vals = [por[s]['pct'] for s in symbols(idea) if s in por]
    return {'estado': idea.get('estado'), 'desde': ent or idea.get('fecha'), 'hasta': hasta, 'por_valor': {s: v for s, v in por.items() if s != 'SPY'},
            'media': round(sum(vals) / len(vals), 2) if vals else None, 'spy': por['SPY']['pct'] if 'SPY' in por else None}


def bars(sym: str) -> Dict[str, tuple]:
    """Fecha → (apertura, cierre) del último año."""
    b = fetch(sym)
    if b is None:
        return {}
    return {datetime.fromtimestamp(int(t), timezone.utc).strftime('%Y-%m-%d'): (float(o), float(c)) for t, o, c in zip(b['t'], b['o'], b['c'])}


def caso_id(fecha: str, titulo: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', unicodedata.normalize('NFKD', titulo.lower()).encode('ascii', 'ignore').decode()).strip('-')
    return f'{fecha}-{slug[:48]}'


def outcome(valores: List[str], fecha: str, B: Dict[str, Dict[str, tuple]]) -> Dict[str, Optional[float]]:
    """Lo que ganó de más la cesta sobre el SPY comprando en la apertura de la sesión siguiente a `fecha` y vendiendo al cierre 5, 20 y 60 sesiones después (igual que el entrenamiento). None = aún no ha pasado ese plazo."""
    spy = B.get('SPY', {})
    days = sorted(spy)
    nxt = [d for d in days if d > fecha]
    out: Dict[str, Optional[float]] = {'e5': None, 'e20': None, 'e60': None}
    if not nxt:
        return out
    i0 = days.index(nxt[0])
    for h in (5, 20, 60):
        if i0 + h >= len(days):
            continue
        d0, dh = days[i0], days[i0 + h]
        rets = []
        for s in valores:
            b = B.get(s, {})
            ins = [d for d in sorted(b) if d >= d0]                 # otras bolsas pueden tener festivos distintos: primera sesión desde d0
            outs = [d for d in sorted(b) if d <= dh]
            if ins and outs and ins[0] <= outs[-1]:
                rets.append(b[outs[-1]][1] / b[ins[0]][0] - 1)
        if rets:
            out[f'e{h}'] = round(100 * (sum(rets) / len(rets) - (spy[dh][1] / spy[d0][0] - 1)), 2)
    return out


def registro(D: Dict, prev: Dict, B: Dict[str, Dict[str, tuple]]) -> Dict:
    """Registro permanente de casos propios: cada idea dada y cada idea descartada (con sus valores) y lo que pasó después. Es la materia con la que se revisan las lecciones.
    Los casos ya apuntados no se borran aunque desaparezcan de ideas.json (los descartes se rehacen en cada lectura), y un resultado ya medido no se recalcula."""
    casos = {c['id']: c for c in prev.get('casos', [])}
    nuevos = [(i['id'], i['fecha'], i['titulo'], i.get('tipo'), symbols(i), True) for i in D.get('ideas', [])]
    nuevos += [(caso_id(d['fecha'], d['titulo']), d['fecha'], d['titulo'], d.get('tipo'), list(d.get('valores') or []), False) for d in D.get('descartes', []) if d.get('valores')]
    for cid, fecha, titulo, tipo, vals, dada in nuevos:
        c = casos.setdefault(cid, {'id': cid, 'fecha': fecha, 'titulo': titulo, 'tipo': tipo, 'valores': vals, 'dada': dada, 'e5': None, 'e20': None, 'e60': None})
        c.update(titulo=titulo, tipo=tipo, dada=dada)
    for c in casos.values():
        if c['e60'] is None:
            o = outcome(c['valores'], c['fecha'], B)
            for k, v in o.items():
                if c.get(k) is None:
                    c[k] = v
    lst = sorted(casos.values(), key=lambda c: (c['fecha'], c['id']))

    def res(dada: bool) -> Dict:
        x = [c['e20'] for c in lst if c['dada'] == dada and c['e20'] is not None]
        return {'n': len(x), 'e20': round(sum(x) / len(x), 2) if x else None, 'aciertos': sum((v > 0) == dada for v in x)}      # una idea dada acierta si sube más que el índice; una descartada, si no lo hace
    return {'casos': lst, 'resumen': {'dadas': res(True), 'descartadas': res(False)}}


def main(out: str = 'data') -> None:
    ideas = json.load(open(f'{out}/ideas.json', encoding='utf-8'))['ideas']
    syms = sorted({s for i in ideas for s in symbols(i)} | {'SPY'})
    cl = {s: closes(s) for s in syms}
    track = {'generado': int(time.time()), 'ideas': {i['id']: score(i, cl) for i in ideas}}
    json.dump(track, open(f'{out}/ideas_track.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    D = json.load(open(f'{out}/ideas.json', encoding='utf-8'))
    pc = f'{out}/ideas_casos.json'
    prev = json.load(open(pc, encoding='utf-8')) if os.path.exists(pc) else {}
    need = sorted({v for d in D.get('descartes', []) for v in (d.get('valores') or [])} | {v for c in prev.get('casos', []) for v in c['valores']} | set(syms))
    reg = registro(D, prev, {x: bars(x) for x in need})
    reg['generado'] = int(time.time())
    json.dump(reg, open(pc, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"casos propios: {len(reg['casos'])} apuntados · con resultado a 20 sesiones: {reg['resumen']['dadas']['n']} dadas, {reg['resumen']['descartadas']['n']} descartadas")
    for i in ideas:
        r = track['ideas'][i['id']]
        if r['media'] is None or r['spy'] is None:
            print(f"{i['estado']:10s} {i['id']}: sin precios")
        else:
            print(f"{i['estado']:10s} {i['id']}: {r['media']:+.1f}% vs SPY {r['spy']:+.1f}% (desde {r['desde']}) · " + ' '.join(f"{s} {v['pct']:+.1f}%" for s, v in r['por_valor'].items()))


if __name__ == '__main__':
    main(*sys.argv[1:2])
