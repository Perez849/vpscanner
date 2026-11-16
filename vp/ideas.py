"""Ideas por coyuntura: marcador de data/ideas.json → data/ideas_track.json.

Cada idea abierta o cerrada se compara con el SPY desde el cierre del día de entrada; las que están en vigilancia, desde el precio de referencia del día en que se dieron
(solo como referencia: todavía no se ha entrado). Las ideas las escribe el analista en cada sesión (ver IDEAS.md); este módulo solo pone los precios."""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import data


def closes(sym: str) -> Dict[str, float]:
    b = data.fetch_one(sym, '1y')
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


def main(out: str = 'data') -> None:
    ideas = json.load(open(f'{out}/ideas.json', encoding='utf-8'))['ideas']
    syms = sorted({s for i in ideas for s in symbols(i)} | {'SPY'})
    cl = {s: closes(s) for s in syms}
    track = {'generado': int(time.time()), 'ideas': {i['id']: score(i, cl) for i in ideas}}
    json.dump(track, open(f'{out}/ideas_track.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for i in ideas:
        r = track['ideas'][i['id']]
        if r['media'] is None or r['spy'] is None:
            print(f"{i['estado']:10s} {i['id']}: sin precios")
        else:
            print(f"{i['estado']:10s} {i['id']}: {r['media']:+.1f}% vs SPY {r['spy']:+.1f}% (desde {r['desde']}) · " + ' '.join(f"{s} {v['pct']:+.1f}%" for s, v in r['por_valor'].items()))


if __name__ == '__main__':
    main(*sys.argv[1:2])
