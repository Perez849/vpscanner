"""
notify.py — Avisos de las alertas.

  · Telegram (opcional): define en GitHub → Settings → Secrets → Actions los secretos
    TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID. Sin ellos simplemente no envía nada.
  · Resumen en la pestaña "Actions" de cada ejecución ($GITHUB_STEP_SUMMARY).
"""
from __future__ import annotations
import json
import os
import urllib.request
from typing import Any, Dict, List


def _fmt_alert(a: Dict[str, Any]) -> str:
    lines = [f"🟢 COMPRA *{a['sym']}*  ({a['label'][:60]})"]
    for pl in a['plans']:
        lv, ex = pl['levels'], pl['exit']
        if lv.get('targetPct') is not None:
            sal = f"objetivo {lv['targetPct']:+.1f}%"
        elif lv.get('exitTrigPct') is not None:
            sal = f"salida si RSI(2)>70 (hoy ≈ cierre ≥ {lv['exitTrigPct']:+.1f}%)"
        else:
            sal = 'salida por señal'
        lines.append(f"   {pl['label']}: P={pl['p'] * 100:.0f}% · esperado {pl['ev']:+.2f}% · {sal} · stop {lv['stopPct']:+.1f}% · máx {ex['H']} ses.")
    return '\n'.join(lines)


def build_message(alerts: List[Dict[str, Any]], meta: Dict[str, Any], tracking: Dict[str, Any]) -> str:
    g = tracking['stats']['global']
    head = f"📡 *VP Scanner* · datos hasta {meta.get('dataThrough')}\n"
    mk = meta.get('market') or {}
    if mk:
        head += f"Mercado: SPY {'sobre' if mk.get('spyUp') else 'BAJO'} SMA200 · VIX {mk.get('vix')}\n"
    if g['n']:
        head += f"Real en vivo: {g['wr']}% acierto en {g['n']} operaciones cerradas ({g['mean']:+.2f}%/op)\n"
    if not alerts:
        return head + '\nHoy no hay alertas que superen el umbral de probabilidad. Mejor no forzar.'
    body = '\n\n'.join(_fmt_alert(a) for a in alerts[:12])
    more = f"\n\n… y {len(alerts) - 12} más en la web" if len(alerts) > 12 else ''
    return head + '\n' + body + more + "\n\nOperar en la apertura de la próxima sesión."


def send(alerts: List[Dict[str, Any]], meta: Dict[str, Any], tracking: Dict[str, Any]) -> None:
    token, chat = os.environ.get('TELEGRAM_BOT_TOKEN'), os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat:
        return
    text = build_message(alerts, meta, tracking)
    req = urllib.request.Request(
        f'https://api.telegram.org/bot{token}/sendMessage',
        data=json.dumps({'chat_id': chat, 'text': text[:4000], 'parse_mode': 'Markdown',
                         'disable_web_page_preview': True}).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=20).read()
        print('  Telegram: enviado', flush=True)
    except Exception as e:
        print(f'  Telegram: fallo ({e})', flush=True)


def step_summary(alerts, watch, meta, tracking) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not path:
        return
    g = tracking['stats']['global']
    lines = [f"## VP Scanner · datos hasta {meta.get('dataThrough')}",
             f"{meta['symbols']} activos · {meta['alerts']} alertas · {meta['watch']} en vigilancia · "
             f"seguimiento: {g['open']} abiertas, {g['n']} cerradas" + (f", acierto real {g['wr']}%" if g['n'] else ''), '']
    if alerts:
        lines += ['| Activo | Plan | P(acierto) | Esperado | Stop | Salida | Máx. sesiones |', '|---|---|---|---|---|---|---|']
        for a in alerts:
            for pl in a['plans']:
                lv = pl['levels']
                sal = ('%+.1f%%' % lv['targetPct']) if lv.get('targetPct') is not None else ('RSI(2)>70 ≈ cierre %+.1f%%' % lv['exitTrigPct'] if lv.get('exitTrigPct') is not None else 'señal')
                lines.append(f"| {a['sym']} | {pl['label']} | {pl['p'] * 100:.0f}% | {pl['ev']:+.2f}% | {lv['stopPct']:+.1f}% | {sal} | {pl['exit']['H']} |")
    else:
        lines.append('_Sin alertas hoy._')
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
