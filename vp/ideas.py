"""Parte 3 · ideas por coyuntura: marcador de data/ideas.json. Cada idea abierta o cerrada se compara con el SPY desde el cierre del día de entrada; las que están en vigilancia, desde el precio de referencia del día en que se dieron."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from typing import Dict, Optional

from . import data


def _closes(sym: str) -> Dict[str, float]:
    b = data.fetch_one(sym, '1y')
    if b is None:
        return {}
    return {datetime.fromtimestamp(int(t), timezone.utc).strftime('%Y-%m-%d'): float(c) for t, c in zip(b['t'], b['c'])}


def _on_or_before(cl: Dict[str, float], day: Optional[str]) -> Optional[float]:
    ks = [k for k in sorted(cl) if day is None or k <= day]
    return cl[ks[-1]] if ks else None


def score(idea: Dict, closes: Dict[str, Dict[str, float]]) -> Dict:
    """Rendimiento medio de los valores de la idea y del SPY desde la entrada (o, si está en vigilancia, desde la referencia) hasta la salida o el último cierre."""
    ent, sal = idea.get('entrada'), idea.get('salida')
    out = {}
    for s in idea['valores'] + ['SPY']:
        cl = closes.get(s, {})
        p0 = _on_or_before(cl, ent) if ent else idea.get('ref', {}).get(s)
        p1 = _on_or_before(cl, sal)
        if p0 and p1:
            out[s] = 100 * (p1 / p0 - 1)
    vals = [out[s] for s in idea['valores'] if s in out]
    return {'por_valor': out, 'media': sum(vals) / len(vals) if vals else None, 'spy': out.get('SPY')}


def main(path: str = 'data/ideas.json') -> None:
    ideas = json.load(open(path, encoding='utf-8'))['ideas']
    syms = sorted({s for i in ideas for s in i['valores']} | {'SPY'})
    closes = {s: _closes(s) for s in syms}
    for i in ideas:
        r = score(i, closes)
        if r['media'] is None or r['spy'] is None:
            print(f"{i['fecha']}  {i['estado']:10s} {i['id']}: sin precios")
            continue
        desde = i.get('entrada') or f"ref. {i['fecha']}"
        print(f"{i['fecha']}  {i['estado']:10s} {i['id']}: {r['media']:+.1f}% vs SPY {r['spy']:+.1f}% (desde {desde}) · " + ' '.join(f'{s} {v:+.1f}%' for s, v in r['por_valor'].items() if s != 'SPY'))


if __name__ == '__main__':
    main(*sys.argv[1:2])
