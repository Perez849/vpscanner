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


def _fmt_alert(a: Dict[str, Any], moc: bool = False) -> str:
    lines = [f"🟢 {'COMPRA AL CIERRE' if moc else 'COMPRA'} *{a['sym']}*  ({a['label'][:60]})"]
    for pl in a['plans']:
        lv, ex = pl['levels'], pl['exit']
        if lv.get('targetPct') is not None:
            sal = f"objetivo {lv['targetPct']:+.1f}%"
        elif lv.get('exitTrigPct') is not None:
            sal = f"salida si RSI(2)>70 (hoy ≈ cierre ≥ {lv['exitTrigPct']:+.1f}%)"
        else:
            sal = 'salida por señal'
        lines.append(f"   #{pl.get('rank', '?')} {pl['label']}: P={pl['p'] * 100:.0f}% · esperado {pl['ev']:+.2f}% · {sal} · stop {lv['stopPct']:+.1f}% · máx {ex['H']} ses.")
    return '\n'.join(lines)


def _fmt_pelotazo(a: Dict[str, Any]) -> str:
    return (f"🚀 *{a['sym']}*  ({a['label'][:55]})\n   probab. de superar +12% ≈ {a['pBig'] * 100:.0f}% · stop inicial {a['levels']['stopPct']:+.1f}% "
            f"que sube con el precio (trailing {a['exit']['T']:g}×ATR) · máx {a['exit']['H']} sesiones")


def build_message(alerts: List[Dict[str, Any]], meta: Dict[str, Any], tracking: Dict[str, Any], pelotazos: List[Dict[str, Any]] | None = None) -> str:
    g = tracking['stats']['global']
    head = f"📡 *VP Scanner* · datos hasta {meta.get('dataThrough')}\n"
    mk = meta.get('market') or {}
    if mk:
        head += f"Mercado: SPY {'sobre' if mk.get('spyUp') else 'BAJO'} SMA200 · VIX {mk.get('vix')}\n"
    if g['n']:
        head += f"Real en vivo: {g['wr']}% acierto en {g['n']} operaciones cerradas ({g['mean']:+.2f}%/op)\n"
    if not alerts:
        return head + '\nHoy no hay alertas que superen el umbral de probabilidad. Mejor no forzar.'
    new = [a for a in alerts if not a.get('pre')]
    done = [a['sym'] for a in alerts if a.get('pre')]
    note = f"\n\n⏱ Ya avisadas antes del cierre (orden MOC): {', '.join(done)}" if done else ''
    if not new:
        return head + note + '\n\nNo hay alertas nuevas para la apertura de mañana.'
    body = '\n\n'.join(_fmt_alert(a) for a in new[:12])
    more = f"\n\n… y {len(new) - 12} más en la web" if len(new) > 12 else ''
    return head + '\n' + body + more + note + "\n\nOperar en la apertura de la próxima sesión."


def build_message_pre(alerts: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    head = f"⏱ *VP Scanner · PREVIO AL CIERRE* · sesión {meta.get('sigDate')}\n"
    if not alerts:
        return head + '\nNada que comprar al cierre hoy. Mejor no forzar.'
    body = '\n\n'.join(_fmt_alert(a, moc=True) for a in alerts[:12])
    more = f"\n\n… y {len(alerts) - 12} más en la web" if len(alerts) > 12 else ''
    return (head + '\nSeñal PROVISIONAL (la vela de hoy aún no ha cerrado). Para comprar al precio de cierre: orden MOC *antes de las 15:50 ET* '
            '(21:50 en España). Si no puedes, espera al aviso de después del cierre y entra a la apertura de mañana.\n\n' + body + more)


def _telegram(text: str) -> None:
    token, chat = os.environ.get('TELEGRAM_BOT_TOKEN'), os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat:
        return
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


def send(alerts: List[Dict[str, Any]], meta: Dict[str, Any], tracking: Dict[str, Any], pelotazos: List[Dict[str, Any]] | None = None) -> None:
    _telegram(build_message(alerts, meta, tracking, pelotazos))


def send_pre(alerts: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    if alerts:                      # sin avisos no se molesta a nadie antes del cierre
        _telegram(build_message_pre(alerts, meta))


def step_summary(alerts, watch, meta, tracking, pelotazos=None) -> None:
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
    if pelotazos:
        lines += ['', '### 🚀 Pelotazos (experimental)', '| Activo | P(≥+12%) | Stop inicial | Patrón |', '|---|---|---|---|']
        for a in pelotazos:
            lines.append(f"| {a['sym']} | {a['pBig'] * 100:.0f}% | {a['levels']['stopPct']:+.1f}% | {a['label'][:60]} |")
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def step_summary_pre(alerts, sig_date) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not path:
        return
    lines = [f"## VP Scanner · previo al cierre · sesión {sig_date}", f"{len(alerts)} avisos para comprar AL CIERRE (orden MOC antes de las 15:50 ET)", '']
    if alerts:
        lines += ['| Activo | Plan | P(acierto) | Esperado | Stop | Máx. sesiones |', '|---|---|---|---|---|---|']
        for a in alerts:
            for pl in a['plans']:
                lines.append(f"| {a['sym']} | {pl['label']} | {pl['p'] * 100:.0f}% | {pl['ev']:+.2f}% | {pl['levels']['stopPct']:+.1f}% | {pl['exit']['H']} |")
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
