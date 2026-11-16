"""Entrenamiento de las ideas por coyuntura: simula cada tipo de idea sobre hechos pasados y guarda los ensayos en data/ideas_playbook.json.

Para cada hecho (un desplome del petróleo, una grande que cae un 10 %, un caso con nombre) se compra la cesta «lógica» en la APERTURA del día siguiente —cuando de verdad se puede actuar—
y se mide lo que gana de más sobre el SPY a 5, 20 y 60 sesiones, restando lo que esa cesta hace en un día cualquiera. «d0» es lo que se movió el mismo día de la noticia (ya no se puede comprar).
Los hechos de un mismo tipo a menos de 10 sesiones del anterior se descartan para no contar dos veces el mismo episodio.

Uso: python -m vp.ideas_train [--out data] [--cache .cache/ideas]. Solo reescribe la lista «estudios» del fichero: las lecciones las redacta el analista a la vista de las cifras.
Cautelas: universo = miembros ACTUALES del S&P 1500 (faltan los excluidos y quebrados) y los casos con nombre están elegidos a mano."""
from __future__ import annotations
import argparse
import io
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
EXTRA = 'SPY QQQ CL=F GC=F NG=F HG=F ^TNX XLU KRE GDX GEO CXW S ZS NTR SCCO COIN SNAP PINS ZM PTON X CLF'.split()
ap = argparse.ArgumentParser()
ap.add_argument('--out', default='data')
ap.add_argument('--cache', default='.cache/ideas')
ARGS = ap.parse_args()
PX = os.path.join(ARGS.cache, 'px')
os.makedirs(PX, exist_ok=True)


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def universe():
    """Miembros actuales del S&P 500, 400 y 600 con su subindustria GICS (Wikipedia)."""
    p = os.path.join(ARGS.cache, 'universe.json')
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 30 * 86400:
        return json.load(open(p))
    uni = {}
    for page in ('List_of_S%26P_500_companies', 'List_of_S%26P_400_companies', 'List_of_S%26P_600_companies'):
        t = pd.read_html(io.StringIO(get('https://en.wikipedia.org/wiki/' + page).decode()))[0]
        t.columns = [str(c) for c in t.columns]
        sub = next(c for c in t.columns if 'Sub-Industry' in c or 'Sub Industry' in c)
        for _, r in t.iterrows():
            uni.setdefault(str(r['Symbol']).strip().upper().replace('.', '-'), {'sub': str(r[sub])})
    json.dump(uni, open(p, 'w'))
    return uni


def fetch(s):
    p = os.path.join(PX, s + '.json')
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 7 * 86400:
        return 1
    time.sleep(random.random() * 0.3)
    d = get(f'https://{random.choice(["query1", "query2"])}.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?period1=1167609600&period2={int(time.time()) + 86400}&interval=1d&includeAdjustedClose=true')
    try:
        r = json.loads(d)['chart']['result'][0]
        q = r['indicators']['quote'][0]
        json.dump({'t': r['timestamp'], 'o': q['open'], 'c': q['close'], 'v': q['volume'], 'a': r['indicators']['adjclose'][0]['adjclose']}, open(p, 'w'))
        return 1
    except Exception:
        return 0


uni = universe()
with ThreadPoolExecutor(max_workers=6) as ex:
    got = sum(ex.map(fetch, sorted(set(list(uni) + EXTRA))))
print(f'precios: {got} de {len(set(list(uni) + EXTRA))}', flush=True)


def load():
    A, O, DV = {}, {}, {}
    for f in os.listdir(PX):
        s = f[:-5]; j = json.load(open(os.path.join(PX, f)))
        t = pd.to_datetime(np.array(j['t'], dtype='int64'), unit='s').normalize()
        d = pd.DataFrame({'o': j['o'], 'c': j['c'], 'a': j['a'], 'v': j['v']}, index=t).astype('float64')
        d = d[~d.index.duplicated(keep='last')]; d = d[(d.c > 0)]
        A[s] = d.a; O[s] = d.o * d.a / d.c; DV[s] = d.c * d.v
    idx = A['SPY'].index
    A = pd.DataFrame(A).reindex(idx); O = pd.DataFrame(O).reindex(idx); DV = pd.DataFrame(DV).reindex(idx)
    return A, O, DV
A, O, DV = load(); idx = A.index; R = A.pct_change(fill_method=None)
R = R.where((R < 4) & (R > -0.95))
pos = {d: i for i, d in enumerate(idx)}
Av, Ov = A.values, O.values; col = {s: i for i, s in enumerate(A.columns)}
def after(i, syms, h):
    """Compra en la apertura siguiente al día i y venta al cierre h sesiones después; exceso sobre el SPY en el mismo tramo."""
    if i + 1 + h >= len(idx): return np.nan
    c = [col[s] for s in syms if s in col]
    r = np.nanmean(Av[i + h, c] / Ov[i + 1, c] - 1); m = Av[i + h, col['SPY']] / Ov[i + 1, col['SPY']] - 1
    return r - m
def day0(i, syms):
    c = [col[s] for s in syms if s in col]
    return np.nanmean(R.values[i, c]) - R.values[i, col['SPY']]
def thin(days, gap=10):
    out = []; last = -999
    for d in sorted(days):
        i = pos[d]
        if i - last >= gap: out.append(i); last = i
    return out
def summ(rows):
    x = pd.DataFrame(rows, columns=['d0', 'e5', 'e20', 'e60'])
    o = {'n': int(len(x))}
    for k in x.columns:
        v = x[k].dropna(); o[k] = round(100 * v.mean(), 2) if len(v) else None
        if k != 'd0' and len(v) > 2: o['t' + k[1:]] = round(v.mean() / v.std() * np.sqrt(len(v)), 1); o['hit' + k[1:]] = int(round(100 * (v > 0).mean()))
    return o
def base(basket):
    ii = np.arange(260 + int(np.argmax(since)), len(idx) - 62, 7)
    return {h: float(np.nanmean([after(i, basket, h) for i in ii])) for h in (5, 20, 60)}
def study(days, basket, sign=1):
    b = base(basket)
    rows = [(sign * day0(i, basket), sign * (after(i, basket, 5) - b[5]), sign * (after(i, basket, 20) - b[20]), sign * (after(i, basket, 60) - b[60])) for i in thin(days)]
    return summ(rows)
OUT = []
def add(grupo, nombre, logica, res):
    OUT.append({'grupo': grupo, 'nombre': nombre, 'logica': logica, **res})
    print(f"{grupo[:10]:10s} {nombre[:58]:58s} n={res['n']:4d}  dia0 {res['d0']:+6.2f}%  | 5s {res['e5']:+6.2f}% (t {res.get('t5',0):+.1f}, {res.get('hit5',0)}%) | 20s {res['e20']:+6.2f}% (t {res.get('t20',0):+.1f}, {res.get('hit20',0)}%) | 60s {res['e60']:+6.2f}% (t {res.get('t60',0):+.1f})", flush=True)
since = idx >= '2010-01-01'
oil = A['CL=F']; ro = R['CL=F'].where(oil.shift(1) > 20)
AIR = 'DAL UAL AAL LUV ALK JBLU'.split(); CRU = 'CCL RCL NCLH'.split(); EP = 'XOM CVX COP OXY EOG DVN'.split(); REF = 'VLO MPC PSX'.split()
BANK = 'JPM BAC WFC C KRE'.split(); HB = 'DHI LEN PHM TOL NVR'.split(); UTIL = ['XLU']; GOLDM = 'NEM GDX RGLD AEM'.split(); GAS = 'EQT AR RRC CTRA'.split(); COP_ = 'FCX SCCO'.split()
G = 'Materias primas y tipos'
for th, lab in ((-0.04, '4 %'), (-0.07, '7 %')):
    d = idx[since & (ro <= th)]
    add(G, f'Petróleo cae ≥ {lab} en un día → aerolíneas', 'El combustible es su mayor coste variable.', study(d, AIR))
    add(G, f'Petróleo cae ≥ {lab} en un día → cruceros', 'Combustible y ánimo de consumo.', study(d, CRU))
    add(G, f'Petróleo cae ≥ {lab} en un día → petroleras (comprar la caída)', 'Apostar a que la caída del crudo se da la vuelta.', study(d, EP))
d = idx[since & (ro >= 0.04)]
add(G, 'Petróleo sube ≥ 4 % en un día → petroleras', 'Cada dólar de más va a su margen.', study(d, EP))
add(G, 'Petróleo sube ≥ 4 % en un día → refino', 'Margen de refino.', study(d, REF))
add(G, 'Petróleo sube ≥ 4 % en un día → aerolíneas (comprar el castigo)', 'Apostar a que el castigo a las aerolíneas es excesivo.', study(d, AIR))
# tramos largos: petróleo −15 % en 20 sesiones
o20 = oil / oil.shift(20) - 1
d = idx[since & (o20 <= -0.15) & (oil.shift(20) > 20)]
add(G, 'Petróleo acumula −15 % en un mes → aerolíneas', 'La bajada ya no es un titular: llega a las cuentas del trimestre.', study(d, AIR))
d = idx[since & (o20 >= 0.15) & (oil.shift(20) > 20)]
add(G, 'Petróleo acumula +15 % en un mes → petroleras', 'La subida llega a las cuentas del trimestre.', study(d, EP))
ty = A['^TNX'].diff()
d = idx[since & (ty >= 0.12)]
add(G, 'El bono a 10 años sube ≥ 12 pb en un día → bancos', 'Más margen de intereses.', study(d, BANK))
add(G, 'El bono a 10 años sube ≥ 12 pb en un día → constructoras (comprar el castigo)', 'Hipotecas más caras: apostar a que el castigo es excesivo.', study(d, HB))
d = idx[since & (ty <= -0.12)]
add(G, 'El bono a 10 años baja ≥ 12 pb en un día → constructoras', 'Hipotecas más baratas.', study(d, HB))
add(G, 'El bono a 10 años baja ≥ 12 pb en un día → eléctricas', 'Su dividendo compite mejor con el bono.', study(d, UTIL))
d = idx[since & (R['GC=F'] >= 0.02)]
add(G, 'Oro sube ≥ 2 % en un día → mineras de oro', 'Apalancamiento operativo al precio del oro.', study(d, GOLDM))
d = idx[since & (R['NG=F'] >= 0.08)]
add(G, 'Gas natural sube ≥ 8 % en un día → productoras de gas', 'Precio de venta más alto.', study(d, GAS))
d = idx[since & (R['HG=F'] >= 0.03)]
add(G, 'Cobre sube ≥ 3 % en un día → mineras de cobre', 'Precio de venta más alto.', study(d, COP_))
G = 'Pánico de mercado'
d = idx[since & (R['SPY'] <= -0.03)]
rows = []
for i in thin(d):
    f = lambda h: Av[i + h, col['QQQ']] / Ov[i + 1, col['QQQ']] - 1 if i + 1 + h < len(idx) else np.nan
    rows.append((R.values[i, col['SPY']], f(5), f(20), f(60)))
base = {h: float(np.nanmean(Av[h:, col['QQQ']] / Av[:-h, col['QQQ']] - 1)) for h in (5, 20, 60)}
res = summ(rows); res['nota'] = f"rendimiento absoluto del Nasdaq-100 (no exceso); en un tramo cualquiera hace {100*base[5]:+.1f} % / {100*base[20]:+.1f} % / {100*base[60]:+.1f} %"
add(G, 'El S&P 500 cae ≥ 3 % en un día → comprar el Nasdaq-100', 'El pánico vende de más.', res)
# líder que tropieza / se dispara
G = 'Tropiezo o salto de una empresa grande'
subs = {}
for s, u in uni.items():
    if s in col and u.get('sub'): subs.setdefault(u['sub'], []).append(s)
rank = DV.rolling(60, min_periods=30).median().rank(axis=1, ascending=False)
Rv = R.values; rk = rank.values; spy = Rv[:, col['SPY']]
evs = {'caida': [], 'salto': []}
for s in uni:
    if s not in col or not uni[s].get('sub'): continue
    peers = [p for p in subs[uni[s]['sub']] if p != s]
    if len(peers) < 2: continue
    j = col[s]; pc = [col[p] for p in peers]
    for i in np.flatnonzero(since & (np.abs(Rv[:, j]) >= 0.10)):
        if i < 61 or not rk[i - 1, j] <= 300 or abs(spy[i]) > 0.02: continue
        pm = np.nanmean(Rv[i, pc]) - spy[i]
        kind = 'caida' if Rv[i, j] < 0 else 'salto'
        evs[kind].append((i, s, peers, Rv[i, j], pm))
_ub = {}
def ubase(h):
    if h not in _ub:
        ii = np.arange(int(np.argmax(since)) + 61, len(idx) - 62, 5); v = []
        for i in ii:
            c = np.flatnonzero(rk[i - 1] <= 300)
            v.append(np.nanmean(Av[i + h, c] / Ov[i + 1, c] - 1) - (Av[i + h, col['SPY']] / Ov[i + 1, col['SPY']] - 1))
        _ub[h] = float(np.nanmean(v))
    return _ub[h]
def peer_study(sel, target):
    rows = []
    print('   (control: una grande cualquiera frente al SPY: 5s %+.2f%% · 20s %+.2f%% · 60s %+.2f%%)' % tuple(100 * ubase(h) for h in (5, 20, 60))) if not _ub else None
    for i, s, peers, r, pm in sel:
        b = peers if target == 'peers' else [s]
        rows.append((pm if target == 'peers' else r - spy[i], after(i, b, 5) - ubase(5), after(i, b, 20) - ubase(20), after(i, b, 60) - ubase(60)))
    return summ(rows)
c = evs['caida']; u_ = evs['salto']
add(G, 'Una grande cae ≥ 10 % en un día → sus competidoras', 'Lo que pierde una lo ganan las otras.', peer_study(c, 'peers'))
add(G, '   …cuando las competidoras subieron ese día (el mercado las ve ganadoras)', 'Sustitución: clientes y cuota cambian de manos.', peer_study([e for e in c if e[4] > 0.005], 'peers'))
add(G, '   …cuando las competidoras cayeron con ella (contagio)', 'El problema se lee como de todo el sector: ¿rebotan?', peer_study([e for e in c if e[4] < -0.01], 'peers'))
add(G, 'Una grande cae ≥ 10 % en un día → comprar la propia caída', 'El castigo es excesivo y el negocio sigue.', peer_study(c, 'self'))
add(G, '   …caída ≥ 20 %', 'Castigos extremos.', peer_study([e for e in c if e[3] <= -0.20], 'self'))
add(G, 'Una grande sube ≥ 10 % en un día → sus competidoras', 'Las buenas noticias de una anticipan las del sector.', peer_study(u_, 'peers'))
add(G, '   …cuando las competidoras apenas se movieron ese día (< 1 %)', 'El mercado aún no ha extendido la noticia.', peer_study([e for e in u_ if abs(e[4]) < 0.01], 'peers'))
add(G, 'Una grande sube ≥ 10 % en un día → seguir comprándola', 'La noticia buena tarda en digerirse.', peer_study(u_, 'self'))
# casos con nombre
G = 'Casos con nombre'
CASES = [
 ('2016-11-09', 'Trump gana (2016) → bancos y defensa', 'JPM BAC GS LMT NOC RTX'),
 ('2018-03-01', 'Aranceles al acero (2018) → acereras de EE. UU.', 'NUE STLD CLF X'),
 ('2020-11-09', 'Vacuna de Pfizer (2020) → aerolíneas y cruceros', 'DAL UAL AAL CCL RCL NCLH'),
 ('2022-02-24', 'Rusia invade Ucrania → defensa', 'LMT NOC RTX GD'),
 ('2022-02-24', 'Rusia invade Ucrania → fertilizantes y energía', 'MOS CF NTR XOM CVX'),
 ('2023-03-10', 'Quiebra de SVB → bancos grandes (refugio de depósitos)', 'JPM BAC WFC C'),
 ('2023-03-13', 'Quiebra de SVB → regionales (comprar el pánico)', 'KRE'),
 ('2023-05-25', 'Nvidia dispara su guía de IA (2023) → cadena de la IA', 'AVGO AMD MRVL ANET SMCI MU'),
 ('2023-10-09', 'Ataque de Hamás → defensa', 'LMT NOC RTX GD'),
 ('2024-07-19', 'Apagón de CrowdStrike → competidoras', 'S PANW ZS FTNT'),
 ('2024-07-19', 'Apagón de CrowdStrike → comprar la propia CrowdStrike', 'CRWD'),
 ('2024-11-06', 'Trump gana (2024) → bancos, prisiones privadas y cripto', 'JPM GS GEO CXW COIN'),
 ('2025-01-27', 'DeepSeek hunde a Nvidia → comprar Nvidia', 'NVDA'),
 ('2025-01-27', 'DeepSeek (IA barata) → software que usa IA', 'CRM NOW MSFT ADBE PLTR'),
 ('2025-04-09', 'Pausa de los aranceles (2025) → los más castigados', 'AAPL NKE BBY RH DECK'),
 ('2022-02-03', 'Meta se hunde un 26 % → comprar Meta', 'META'),
 ('2022-02-03', 'Meta se hunde → otras redes de publicidad', 'GOOGL SNAP PINS'),
 ('2019-03-11', 'Segundo accidente del 737 MAX → comprar Boeing', 'BA'),
 ('2020-03-16', 'Confinamientos → quedarse en casa', 'ZM NFLX AMZN CLX PTON'),
 ('2017-08-28', 'Huracán Harvey → reformas del hogar', 'HD LOW'),
 ('2022-09-29', 'Huracán Ian → reformas y generadores', 'HD LOW GNRC'),
]
for d, name, b in CASES:
    d = pd.Timestamp(d)
    if d not in pos: d = idx[idx.searchsorted(d)]
    i = pos[d]; b = b.split()
    r = {'n': 1, 'd0': round(100 * day0(i, b), 2), 'e5': round(100 * after(i, b, 5), 2), 'e20': round(100 * after(i, b, 20), 2), 'e60': round(100 * after(i, b, 60), 2), 'valores': b, 'fecha': str(d.date())}
    add(G, name, '', r)
cs = [o for o in OUT if o['grupo'] == G]
print(f"casos: media dia0 {np.mean([c['d0'] for c in cs]):+.2f}% · 5s {np.mean([c['e5'] for c in cs]):+.2f}% · 20s {np.mean([c['e20'] for c in cs]):+.2f}% ({sum(c['e20']>0 for c in cs)}/{len(cs)} positivos) · 60s {np.mean([c['e60'] for c in cs]):+.2f}% ({sum(c['e60']>0 for c in cs)}/{len(cs)})")
for o in OUT:
    o['nombre'] = '↳ ' + o['nombre'].strip() if o['nombre'].startswith('   ') else o['nombre'].strip()
pb = os.path.join(ARGS.out, 'ideas_playbook.json')
P = json.load(open(pb, encoding='utf-8')) if os.path.exists(pb) else {'lecciones': [], 'metodo': '', 'cautelas': '', 'resumen': ''}
P['estudios'] = OUT
P['generado'] = time.strftime('%Y-%m-%d')
json.dump(P, open(pb, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
