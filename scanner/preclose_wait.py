#!/usr/bin/env python3
"""
preclose_wait.py — decide qué hace el workflow «Scan previo al cierre» en cada ejecución.

El disparo programado de GitHub llega con 3–6,5 horas de retraso (medido en octubre de 2026), así que no puede lanzar el escaneo a las 15:20 ET.
En su lugar el workflow se programa por la mañana (UTC) y esta pieza ESPERA hasta que se abra la ventana previa al cierre (15:10–15:40 ET):

  · run   → estamos dentro de la ventana (o se ha esperado hasta ella): hay que escanear ahora.
  · skip  → fin de semana, ya pasó la ventana hoy, o ya se escaneó hoy: no hacer nada (el workflow termina en verde).
  · hop   → faltan más horas de las que cabe esperar en un trabajo de GitHub (límite 6 h): se duerme un tramo y el workflow se relanza a sí mismo.

Escribe `decision=<run|skip|hop>` en $GITHUB_OUTPUT (y en la salida estándar). Argumentos: --check-done mira si hoy ya hay un escaneo previo guardado.
Variables: FORCE=true → run sin esperar (pruebas manuales). PRECLOSE_NOW=<ISO UTC> → hora simulada (pruebas); sin dormir de verdad si PRECLOSE_NO_SLEEP=1.
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')
START_MIN, END_MIN = 15 * 60 + 10, 15 * 60 + 40        # ET; igual que scan.py (PRE_WINDOW)
MAX_WAIT_MIN = 330                                      # un trabajo de GitHub dura como mucho 6 h; se deja margen
MARGIN_S = 20                                           # segundos de más tras abrirse la ventana
HERE = os.path.dirname(os.path.abspath(__file__))
PRE_PATH = os.path.join(HERE, 'data', 'alerts_pre.json')


def decide(now_utc: datetime):
    """(decisión, segundos a dormir) para una hora dada."""
    et = now_utc.astimezone(NY)
    if et.weekday() >= 5:
        return 'skip', 0
    minute = et.hour * 60 + et.minute
    if minute > END_MIN:
        return 'skip', 0
    if minute >= START_MIN:
        return 'run', 0
    start = et.replace(hour=START_MIN // 60, minute=START_MIN % 60, second=0, microsecond=0)
    wait = (start - et).total_seconds() + MARGIN_S
    if wait <= MAX_WAIT_MIN * 60:
        return 'wait', wait
    return 'hop', (MAX_WAIT_MIN - 30) * 60


def done_today(now_utc: datetime, path: str = PRE_PATH) -> bool:
    """¿El escaneo previo de HOY (hora de Nueva York) ya está guardado?"""
    try:
        gen = json.load(open(path, encoding='utf-8')).get('generatedAt')
        g = datetime.fromisoformat(gen.replace('Z', '+00:00')).astimezone(NY)
    except Exception:
        return False
    return g.date() == now_utc.astimezone(NY).date() and START_MIN <= g.hour * 60 + g.minute <= END_MIN + 20


def now() -> datetime:
    s = os.environ.get('PRECLOSE_NOW')
    return datetime.fromisoformat(s.replace('Z', '+00:00')) if s else datetime.now(timezone.utc)


def emit(decision: str) -> None:
    print(f'decisión: {decision}', flush=True)
    out = os.environ.get('GITHUB_OUTPUT')
    if out:
        with open(out, 'a') as f:
            f.write(f'decision={decision}\n')


def sleep(sec: float) -> None:
    if os.environ.get('PRECLOSE_NO_SLEEP') == '1':
        print(f'(simulado) dormir {sec / 60:.0f} min', flush=True)
        return
    end = time.time() + sec
    while time.time() < end:
        time.sleep(min(600, max(1, end - time.time())))


def main() -> int:
    if '--check-done' in sys.argv:
        d = done_today(now())
        out = os.environ.get('GITHUB_OUTPUT')
        print(f'¿ya escaneado hoy? {d}', flush=True)
        if out:
            with open(out, 'a') as f:
                f.write(f'done={"true" if d else "false"}\n')
        return 0
    if os.environ.get('FORCE') == 'true':
        emit('run')
        return 0
    for _ in range(3):
        n = now()
        d, secs = decide(n)
        et = n.astimezone(NY)
        print(f'{n:%Y-%m-%d %H:%M} UTC = {et:%a %H:%M} ET → {d}' + (f' (esperar {secs / 60:.0f} min)' if secs else ''), flush=True)
        if d in ('run', 'skip'):
            emit(d)
            return 0
        sleep(secs)
        if d == 'hop':
            emit('hop')
            return 0
        if os.environ.get('PRECLOSE_NOW'):                       # en pruebas el reloj no avanza solo: se acaba de «dormir» hasta la ventana
            os.environ['PRECLOSE_NOW'] = datetime.fromtimestamp(n.timestamp() + secs, timezone.utc).isoformat()
    emit('skip')
    return 0


if __name__ == '__main__':
    sys.exit(main())
