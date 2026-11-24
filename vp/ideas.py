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

from . import data, europa


def _raw(sym: str, rng: str = '1y'):
    """Barras tal cual las da Yahoo, incluida la de hoy aunque la sesión no haya terminado (para las alertas en vivo y los valores de otras bolsas)."""
    for host in data.HOSTS:
        url = data.URL.format(host=host, sym=urllib.parse.quote(sym, safe='.'), span=data.span(rng))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': data.UA, 'Accept': 'application/json'}), timeout=25) as r:
                return data.parse(json.loads(r.read().decode('utf-8')))
        except (OSError, ValueError):
            continue
    return None


def fetch(sym: str, vivo: bool = False):
    """Barras de un año. Los valores de otras bolsas llevan sufijo con punto (IDR.MC), que `data.fetch_one` convertiría en guion: esos se piden tal cual.
    Con `vivo`, también los de EE. UU. se piden tal cual: la barra de hoy entra con el último precio aunque Wall Street no haya cerrado."""
    if '.' not in sym and not vivo:
        return data.fetch_one(sym, '1y')
    return _raw(sym)


def sesion_abierta(now: Optional[datetime] = None) -> bool:
    """¿Está Wall Street en plena sesión (de 9:30 a 16:20, hora de Nueva York, de lunes a viernes)? Entonces las alertas usan el precio del momento."""
    from zoneinfo import ZoneInfo
    ny = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo('America/New_York'))
    return ny.weekday() < 5 and (9, 30) <= (ny.hour, ny.minute) < (16, 20)


def closes(sym: str, vivo: bool = False) -> Dict[str, float]:
    b = fetch(sym, vivo)
    if b is None:
        return {}
    return {datetime.fromtimestamp(int(t), timezone.utc).strftime('%Y-%m-%d'): float(c) for t, c in zip(b['t'], b['c'])}


def on_or_before(cl: Dict[str, float], day: Optional[str]) -> Optional[tuple]:
    ks = [k for k in sorted(cl) if day is None or k <= day]
    return (ks[-1], cl[ks[-1]]) if ks else None


SERIE = 60


def symbols(idea: Dict) -> List[str]:
    return [v['sym'] if isinstance(v, dict) else v for v in idea.get('valores', [])]


def indice(idea: Dict) -> str:
    """Índice con el que se compara la idea: el SPY salvo que la ficha diga otro (`indice`: '^STOXX50E' o '^IBEX' para las ideas europeas)."""
    return idea.get('indice') or 'SPY'


def score(idea: Dict, cl: Dict[str, Dict[str, float]]) -> Dict:
    """Rendimiento de cada valor y de su índice (el SPY, o el europeo si la ficha lo dice) desde la entrada (o la referencia, si está en vigilancia) hasta la salida o el último cierre;
    `media` es la de los valores a partes iguales. La clave `spy` guarda el índice de la idea, sea cual sea (`indice` dice cuál)."""
    ent, sal = idea.get('entrada'), idea.get('salida')
    bx = indice(idea)
    por: Dict[str, Dict] = {}
    hasta = None
    for s in symbols(idea) + [bx]:
        c = cl.get(s, {})
        a = on_or_before(c, ent) if ent else None
        p0 = a[1] if a else (idea.get('ref') or {}).get(s)
        b = on_or_before(c, sal)
        if p0 and b and (ent or b[0] >= (idea.get('fecha') or '')):          # en vigilancia no se compara con cierres anteriores al día en que se dio la idea
            por[s] = {'ref': round(p0, 2), 'ahora': round(b[1], 2), 'pct': round(100 * (b[1] / p0 - 1), 2)}
            ks = [k for k in sorted(c) if k <= b[0]][-SERIE:]                 # los últimos cierres, para el gráfico de la ficha; i0 = primer cierre desde el día del hecho
            desde = ent or idea.get('fecha') or ''
            por[s]['serie'] = [round(c[k], 2) for k in ks]
            por[s]['i0'] = next((i for i, k in enumerate(ks) if k >= desde), len(ks) - 1)
            hasta = b[0]
    vals = [por[s]['pct'] for s in symbols(idea) if s in por]
    pendiente = bool(ent) and (hasta is None or hasta < ent)                  # abierta hoy y aún sin el cierre del día de entrada: no hay nada que medir
    if pendiente:
        for v in por.values():
            v['pct'] = None
    hipotetica = idea.get('estado') == 'cerrada' and not ent                  # se cerró sin comprarse: la cifra es solo lo que habría pasado, para aprender
    return {'estado': idea.get('estado'), 'desde': ent or idea.get('fecha'), 'hasta': hasta, 'por_valor': {s: v for s, v in por.items() if s != bx},
            'media': None if pendiente else (round(sum(vals) / len(vals), 2) if vals else None), 'spy': None if pendiente else (por[bx]['pct'] if bx in por else None),
            'indice': bx, 'pendiente': pendiente, 'hipotetica': hipotetica}


def bars(sym: str) -> Dict[str, tuple]:
    """Fecha → (apertura, cierre) del último año."""
    b = fetch(sym)
    if b is None:
        return {}
    return {datetime.fromtimestamp(int(t), timezone.utc).strftime('%Y-%m-%d'): (float(o), float(c)) for t, o, c in zip(b['t'], b['o'], b['c'])}


def caso_id(fecha: str, titulo: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', unicodedata.normalize('NFKD', titulo.lower()).encode('ascii', 'ignore').decode()).strip('-')
    return f'{fecha}-{slug[:48]}'


def outcome(valores: List[str], fecha: str, B: Dict[str, Dict[str, tuple]], bench: str = 'SPY') -> Dict[str, Optional[float]]:
    """Lo que ganó de más la cesta sobre su índice (el SPY; el Euro Stoxx 50 si todos los valores son europeos) comprando en la apertura de la sesión siguiente a `fecha`
    y vendiendo al cierre 5, 20 y 60 sesiones después (igual que el entrenamiento). None = aún no ha pasado ese plazo."""
    spy = B.get(bench, {})
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
    for i in D.get('ideas', []):                                              # de las ideas dadas se guarda también la convicción y si llegó a comprarse (para medir si la convicción sirve)
        if i['id'] in casos:
            casos[i['id']].update(conviccion=i.get('conviccion'), comprada=bool(i.get('entrada')))
    for c in casos.values():
        if c['e60'] is None:
            eu = bool(c['valores']) and all(europa.is_eu(v) for v in c['valores'])
            o = outcome(c['valores'], c['fecha'], B, '^STOXX50E' if eu else 'SPY')
            for k, v in o.items():
                if c.get(k) is None:
                    c[k] = v
    lst = sorted(casos.values(), key=lambda c: (c['fecha'], c['id']))

    def res(dada: bool) -> Dict:
        x = [c['e20'] for c in lst if c['dada'] == dada and c['e20'] is not None]
        return {'n': len(x), 'e20': round(sum(x) / len(x), 2) if x else None, 'aciertos': sum((v > 0) == dada for v in x)}      # una idea dada acierta si sube más que el índice; una descartada, si no lo hace
    return {'casos': lst, 'resumen': {'dadas': res(True), 'descartadas': res(False)}, 'calibracion': calibracion(lst)}


def calibracion(lst: List[Dict]) -> Dict[str, Dict]:
    """¿Sirve la convicción? Ideas que llegaron a comprarse, agrupadas por convicción, con su resultado medio sobre su índice a 20 y 60 sesiones (el mismo que el de cada caso).
    Si las de convicción 3 no rinden más que las de 2, la convicción no está bien puesta y hay que recalibrarla (IDEAS.md, revisión mensual)."""
    out: Dict[str, Dict] = {}
    for k in sorted({c.get('conviccion') for c in lst if c.get('comprada') and c.get('conviccion')}):
        cs = [c for c in lst if c.get('comprada') and c.get('conviccion') == k]
        m = lambda h: (lambda x: {'n': len(x), 'media': round(sum(x) / len(x), 2) if x else None, 'aciertos': sum(v > 0 for v in x)})([c[h] for c in cs if c.get(h) is not None])
        out[str(k)] = {'ideas': len(cs), 'e20': m('e20'), 'e60': m('e60')}
    return out


def edicion_id(cuando: str) -> str:
    """'2026-11-16T13:22:05Z' → '2026-11-16-1322' (hora UTC de la lectura)."""
    return cuando[:10] + '-' + cuando[11:13] + cuando[14:16]


def archivar(D: Dict, track: Dict, out: str) -> Optional[str]:
    """Hemeroteca: guarda la edición completa (ideas y marcador) en data/ediciones/<id>.json y la apunta en el índice. Una edición ya guardada se reescribe (misma lectura corregida), no se duplica."""
    cuando = (D.get('sesion') or {}).get('cuando')
    if not cuando:
        return None
    eid = edicion_id(cuando)
    d = os.path.join(out, 'ediciones')
    os.makedirs(d, exist_ok=True)
    json.dump({'ideas': D, 'track': track}, open(os.path.join(d, eid + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    pi = os.path.join(d, 'index.json')
    idx = json.load(open(pi, encoding='utf-8')) if os.path.exists(pi) else []
    ses = D['sesion']
    fila = {'id': eid, 'cuando': cuando, 'tipo': ses.get('tipo'), 'titular': (ses.get('titular') or '').replace('*', ''), 'hoy': (ses.get('hoy') or {}).get('texto'), 'accion': (ses.get('hoy') or {}).get('accion'),
            'vivas': sum(i.get('estado') != 'cerrada' for i in D.get('ideas', []))}
    idx = sorted([x for x in idx if x['id'] != eid] + [fila], key=lambda x: x['id'], reverse=True)
    json.dump(idx, open(pi, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return eid


def caducadas(D: Dict, hoy: str) -> List[str]:
    """Ideas en vigilancia cuya fecha de caducidad ya ha pasado: hay que cerrarlas como «no saltó»."""
    return [i['id'] for i in D.get('ideas', []) if i.get('estado') == 'vigilancia' and i.get('caduca') and i['caduca'] < hoy]


# ── alertas: qué lecciones se activan con las condiciones de hoy ──
WATCH = ('SPY', 'QQQ', 'SMH', '^VIX', '^TNX', 'CL=F', 'ZW=F', 'ZC=F', 'GC=F')


def metricas(cl: Dict[str, Dict[str, float]]) -> Dict[str, Optional[float]]:
    """De los cierres diarios: último movimiento del S&P 500, racha, VIX de hoy y de ayer, bono, petróleo, trigo, maíz y oro."""
    def serie(s):
        c = cl.get(s, {})
        return [c[k] for k in sorted(c)]
    spy, tnx = serie('SPY'), serie('^TNX')
    r = lambda x, n=1: (x[-1] / x[-1 - n] - 1) if len(x) > n else None
    vx, sd = cl.get('^VIX', {}), max(cl.get('SPY') or {'9999': 0})
    vix = [vx[k] for k in sorted(vx) if k <= sd]                          # el VIX se empareja por fecha con la última sesión del SPY: el previo es siempre el del día anterior a esa sesión
    m: Dict[str, Optional[float]] = {'spy_1d': r(spy), 'spy_3d': r(spy, 3), 'vix': vix[-1] if vix else None, 'vix_prev': vix[-2] if len(vix) > 1 else None,
                                     'tnx_1d': (tnx[-1] - tnx[-2]) if len(tnx) > 1 else None, 'tnx_10d': (tnx[-1] - tnx[-11]) if len(tnx) > 10 else None}
    m['racha'] = sum(1 for k in range(1, min(len(spy), 6)) if spy[-k] < spy[-k - 1]) if len(spy) > 6 else None
    if m['racha'] is not None:                                            # días seguidos a la baja contando desde el último
        n = 0
        for k in range(1, len(spy)):
            if spy[-k] < spy[-k - 1]: n += 1
            else: break
        m['racha'] = n
    for s, k in (('CL=F', 'oil_1d'), ('ZW=F', 'trigo_1d'), ('ZC=F', 'maiz_1d'), ('GC=F', 'oro_1d')):
        m[k] = r(serie(s))
    eu = cl.get('^STOXX50E', {})
    if eu:                                                                # Europa: último día de su sesión y el VIX del cierre de Wall Street ANTERIOR a ese día (el miedo que había al abrir Europa)
        fe = max(eu)
        m['eu_fecha'] = fe
        m['eu_1d'] = r(serie('^STOXX50E'))
        ib = cl.get('^IBEX', {})
        m['ibex_1d'] = r([ib[k] for k in sorted(ib) if k <= fe]) if ib else None
        vx = cl.get('^VIX', {})
        prev = [k for k in sorted(vx) if k < fe]
        m['eu_vix_prev'] = vx[prev[-1]] if prev else None
    return m


def reglas(m: Dict[str, Optional[float]], movers: List[tuple]) -> List[str]:
    """Traduce las condiciones del día a las lecciones de data/ideas_playbook.json que hay que tener en cuenta. Pura: sin red."""
    out: List[str] = []
    v, vp_ = m.get('vix'), m.get('vix_prev')
    alto = vp_ is not None and vp_ >= 28
    if v is not None and v >= 28:
        out.append(f'Miedo alto (VIX {v:.1f}): revisa los descartes de «comprar la caída», «comprar al castigado» y «competidoras castigadas» (lección «miedo»). Sigue siendo «no» si la caída es por un problema propio.')
    s1 = m.get('spy_1d')
    if s1 is not None and s1 <= -0.02:
        out.append(f'S&P 500 {100 * s1:+.1f} % con el VIX previo en {vp_:.1f}: ' + ('IDEA de pánico con miedo alto → Nasdaq-100 y, mejor aún, semiconductores (lección «panico»).' if alto else 'caída del 2 % sin miedo alto: no es idea (lección «panico»).'))
    s3, ra = m.get('spy_3d'), m.get('racha')
    if s3 is not None and ra is not None and ra >= 3 and s3 <= -0.03:
        out.append(f'Racha de {ra} días a la baja ({100 * s3:+.1f} % en tres sesiones) con el VIX previo en {vp_:.1f}: ' + ('IDEA de pánico con miedo alto (lección «panico»).' if alto else 'sin miedo alto la ventaja es pequeña (lección «panico»).'))
    t1, t10 = m.get('tnx_1d'), m.get('tnx_10d')
    if t1 is not None and abs(t1) >= 0.12:
        out.append(f'Bono a 10 años {t1 * 100:+.0f} pb en un día: el beneficiario evidente (bancos, constructoras) ya lo ha recogido (lección «obvio»).')
    if t10 is not None and t10 <= -0.30:
        out.append(f'Bono a 10 años {t10 * 100:+.0f} pb en diez sesiones: pista de constructoras a 60 sesiones (lección «segundo»).')
    o = m.get('oil_1d')
    if o is not None and abs(o) >= 0.04:
        out.append(f'Petróleo {100 * o:+.1f} %: no comprar aerolíneas ni petroleras por el movimiento en sí (lección «obvio»); solo cuenta si hay un hecho duradero detrás (lección «estructural»).')
    for k, nom in (('trigo_1d', 'Trigo'), ('maiz_1d', 'Maíz')):
        x = m.get(k)
        if x is not None and x >= 0.04:
            out.append(f'{nom} {100 * x:+.1f} %: pista de fertilizantes (lección «segundo»).')
    g = m.get('oro_1d')
    if g is not None and abs(g) >= 0.02:
        out.append(f'Oro {100 * g:+.1f} %: las mineras lo recogen el mismo día (lección «obvio»).')
    for sym, ch in movers:
        if ch <= -10:
            out.append(f'{sym} {ch:+.1f} %: ' + ('con miedo alto, revisar como castigo excesivo si NO es un problema propio (lección «miedo»); mira también si sus competidoras suben ≥ 2 % hoy (lección «rival»).' if alto else 'gigante caído: no se compra (lección «caido»); mira si sus competidoras suben ≥ 2 % hoy (lección «rival»).'))
        elif ch >= 10:
            out.append(f'{sym} {ch:+.1f} %: si sus competidoras caen hoy, posible castigo excesivo (lección «contagio»); no comprarlas por contagio si están quietas.')
    return out


def reglas_europa(m: Dict, movers: List[tuple]) -> List[str]:
    """Lo mismo para Europa y España (lección «europa»): el miedo que cuenta es el VIX del cierre de Wall Street anterior a la sesión europea. Pura: sin red."""
    out: List[str] = []
    vp_ = m.get('eu_vix_prev')
    alto = vp_ is not None and vp_ >= 28
    fe = m.get('eu_fecha') or ''
    for k, nom in (('eu_1d', 'Euro Stoxx 50'), ('ibex_1d', 'Ibex 35')):
        x = m.get(k)
        if x is not None and x <= -0.02:
            out.append(f'Europa ({fe}): {nom} {100 * x:+.1f} % con el VIX previo en {"—" if vp_ is None else f"{vp_:.1f}"}: ' + (f'IDEA de pánico con miedo alto en Europa: comprar el {nom} (lección «europa»).' if alto else 'caída del 2 % sin miedo alto: no es idea (lección «europa»).'))
    s1, us_alto = m.get('spy_1d'), m.get('vix_prev') is not None and m['vix_prev'] >= 28
    if s1 is not None and s1 <= -0.02 and us_alto:
        out.append('Wall Street −2 % con el miedo alto: también es IDEA en Europa, comprando el Euro Stoxx 50 o el Ibex 35 en la apertura europea siguiente (lección «europa»).')
    for sym, ch in movers:
        if ch <= -10:
            out.append(f'Europa ({fe}): {sym} {ch:+.1f} %: ' + ('con miedo alto, revisar como castigo excesivo si NO es un problema propio, igual que en EE. UU. (lecciones «miedo» y «europa»).' if alto else 'gigante caído sin miedo alto: no se compra (lecciones «caido» y «europa»).'))
        elif ch >= 10:
            out.append(f'Europa ({fe}): {sym} {ch:+.1f} %: no se persigue el salto (lección «europa»); mira si sus competidoras caen hoy (lección «contagio»).')
    return out


def _cambio(sym: str) -> Optional[tuple]:
    """(fecha de la última barra, % frente al cierre anterior) con la barra de hoy aunque la sesión no haya terminado."""
    b = _raw(sym, '5d')
    if b is None or len(b['c']) < 2:
        return None
    return datetime.fromtimestamp(int(b['t'][-1]), timezone.utc).strftime('%Y-%m-%d'), 100 * (float(b['c'][-1]) / float(b['c'][-2]) - 1)


def movers_vivos(syms: List[str], fecha: Optional[str] = None, umbral: float = 10.0) -> List[tuple]:
    """Valores que se mueven un 10 % o más en su última sesión (si `fecha`, solo los de esa sesión: lo demás es un dato viejo)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(_cambio, syms))
    return [(s, round(x[1], 1)) for s, x in zip(syms, res) if x and (fecha is None or x[0] == fecha) and abs(x[1]) >= umbral]


def alertas(out: str = 'data', vivo: Optional[bool] = None) -> List[str]:
    """Alertas del día. Si Wall Street está en plena sesión (o con `vivo`), con el precio del momento: la barra de hoy entra aunque no haya cerrado, y las caídas o saltos del 10 % se
    miran ahora mismo en los valores del escaneo, no en los de la madrugada. Fuera de sesión, con el último cierre. Europa siempre con su última sesión (la de hoy si está abierta)."""
    vivo = sesion_abierta() if vivo is None else vivo
    cl = {s: closes(s, vivo) for s in WATCH}
    cl.update({s: closes(s, True) for s in europa.IDX})
    try:
        uni = [x['sym'] for x in json.load(open(f'{out}/index.json', encoding='utf-8'))['symbols']]
        if vivo:
            hoy = max(cl.get('SPY') or {'': 0})
            movers = movers_vivos(uni, hoy)
        else:
            movers = [(x['sym'], float(x['chg'])) for x in json.load(open(f'{out}/index.json', encoding='utf-8'))['symbols'] if x.get('chg') is not None and abs(float(x['chg'])) >= 10]
    except (OSError, ValueError, KeyError):
        movers = []
    m = metricas(cl)
    mov_eu = movers_vivos(europa.EURO + europa.ES, m.get('eu_fecha')) if m.get('eu_fecha') else []
    al = reglas(m, movers) + reglas_europa(m, mov_eu)
    from zoneinfo import ZoneInfo
    sp, hoy_ny = max(cl.get('SPY') or {'—': 0}), datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
    print('ALERTAS DE HOY (cada una se atiende en la edición: en una idea, un descarte o un cambio)')
    print(('   precios: Wall Street EN VIVO (sesión de hoy, ' + sp + ', sin cerrar: el cierre aún no es definitivo)') if vivo and sp == hoy_ny else ('   precios: último cierre de Wall Street (' + sp + ')' + (' · hoy aún no ha abierto' if vivo else '')), '· Europa: sesión del', m.get('eu_fecha') or '—')
    for a in al or ['Ninguna condición del entrenamiento activada hoy.']:
        print(' · ' + a)
    print('   métricas: ' + ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in m.items()))
    return al


def main(out: str = 'data') -> None:
    ideas = json.load(open(f'{out}/ideas.json', encoding='utf-8'))['ideas']
    syms = sorted({s for i in ideas for s in symbols(i)} | {indice(i) for i in ideas} | {'SPY', '^STOXX50E'})
    cl = {s: closes(s) for s in syms}
    track = {'generado': int(time.time()), 'ideas': {i['id']: score(i, cl) for i in ideas}}
    try:
        track['alertas'] = alertas(out)
    except Exception as e:                                                # las alertas nunca deben impedir el marcador
        print(f'alertas no disponibles: {e}')
    json.dump(track, open(f'{out}/ideas_track.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    D = json.load(open(f'{out}/ideas.json', encoding='utf-8'))
    pc = f'{out}/ideas_casos.json'
    prev = json.load(open(pc, encoding='utf-8')) if os.path.exists(pc) else {}
    need = sorted({v for d in D.get('descartes', []) for v in (d.get('valores') or [])} | {v for c in prev.get('casos', []) for v in c['valores']} | set(syms))
    reg = registro(D, prev, {x: bars(x) for x in need})
    reg['generado'] = int(time.time())
    json.dump(reg, open(pc, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    eid = archivar(D, track, out)
    print(f'hemeroteca: edición {eid} guardada')
    for cid in caducadas(D, datetime.now(timezone.utc).strftime('%Y-%m-%d')):
        print(f'¡CADUCADA! {cid}: su disparador no ha saltado a tiempo; ciérrala con motivo «no saltó»')
    print(f"casos propios: {len(reg['casos'])} apuntados · con resultado a 20 sesiones: {reg['resumen']['dadas']['n']} dadas, {reg['resumen']['descartadas']['n']} descartadas")
    for i in ideas:
        r = track['ideas'][i['id']]
        if r['media'] is None or r['spy'] is None:
            print(f"{i['estado']:10s} {i['id']}: sin precios")
        else:
            print(f"{i['estado']:10s} {i['id']}: {r['media']:+.1f}% vs {r['indice']} {r['spy']:+.1f}% (desde {r['desde']}) · " + ' '.join(f"{s} {v['pct']:+.1f}%" for s, v in r['por_valor'].items()))


if __name__ == '__main__':
    if sys.argv[1:2] == ['alertas']:                                         # python -m vp.ideas alertas [vivo|cierre] [carpeta]
        a_ = sys.argv[2:]
        alertas(next((x for x in a_ if x not in ('vivo', 'cierre')), 'data'), True if 'vivo' in a_ else False if 'cierre' in a_ else None)
    else:
        main(*sys.argv[1:2])
