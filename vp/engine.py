"""Motor: junta velas, rupturas y figuras en una única lista de eventos causales. Un evento [{name, i, q, g}] existe en el cierre de la barra i usando solo datos hasta i."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import breakouts, candles, shapes

COOLDOWN = 5        # barras mínimas entre dos señales del mismo patrón en el mismo valor


def cooldown(idx: np.ndarray, gap: int = COOLDOWN) -> List[int]:
    """Se queda con la primera señal de cada racha: ignora las que llegan a menos de `gap` barras de la última conservada."""
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            keep.append(int(i))
            last = int(i)
    return keep


def scan(F: Dict[str, np.ndarray], gap: int = COOLDOWN) -> Tuple[List[Dict], Optional[Dict]]:
    """(eventos ordenados por barra, compresión en formación en la última barra o None)."""
    events: List[Dict] = []
    for name, sig in candles.detect(F).items():
        events += [{'name': name, 'i': i, 'q': None, 'g': {}} for i in cooldown(np.flatnonzero(sig), gap)]
    for name, sig in breakouts.detect(F).items():
        events += [{'name': name, 'i': i, 'q': None, 'g': breakouts.geometry(name, F, i)} for i in cooldown(np.flatnonzero(sig), gap) if i >= 20]
    shape_ev, forming = shapes.detect(F)
    last: Dict[str, int] = {}
    for e in shape_ev:                                      # la misma cadencia de enfriamiento para las figuras
        if e['i'] - last.get(e['name'], -10 ** 9) >= gap:
            last[e['name']] = e['i']
            events.append(e)
    events.sort(key=lambda e: (e['i'], e['name']))
    return events, forming
