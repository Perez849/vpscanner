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
EXTRA = ('SPY QQQ IWM CL=F GC=F NG=F HG=F ZW=F ^TNX ^VIX BTC-USD XLU KRE GDX VNQ TAN GEO CXW S ZS NTR SCCO COIN MSTR MARA RIOT SNAP PINS ZM PTON X CLF '
         'XLK XLF XLE XLV XLY XLP XLI XLB XLRE XLC NVO BNTX EADSY ONON SEDG RUN '
         'BP SAN DB ING LYG BCS LVS WYNN BABA JD BIDU IBB XBI VST CEG WBA Z CHGG BHC EEM HYG UUP SHOP EBAY ETSY').split()
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
    f = lambda k: 'n/d' if res.get(k) is None else f'{res[k]:+6.2f}%'
    print(f"{grupo[:10]:10s} {nombre[:60]:60s} n={res['n']:4d}  dia0 {f('d0')} | 5s {f('e5')} (t {res.get('t5', 0):+.1f}) | 20s {f('e20')} (t {res.get('t20', 0):+.1f}, {res.get('hit20', 0)}%) | 60s {f('e60')} (t {res.get('t60', 0):+.1f}, {res.get('hit60', 0)}%)", flush=True)
since = idx >= '2008-01-01'
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
REIT = ['VNQ']; BTCB = 'COIN MSTR MARA RIOT'.split(); FERT = 'MOS CF NTR'.split(); AGRI = 'ADM BG DE'.split(); CHEM = 'DOW LYB CE'.split(); TRUCK = 'JBHT ODFL KNX'.split(); RETAIL = 'WMT TGT DG COST'.split(); SOLAR = 'FSLR ENPH TAN'.split()
d = idx[since & (ro <= -0.04)]
add(G, 'Petróleo cae ≥ 4 % en un día → transporte por carretera', 'Gasóleo más barato.', study(d, TRUCK))
add(G, 'Petróleo cae ≥ 4 % en un día → grandes superficies', 'El consumidor gasta menos en gasolina y más en la tienda (segundo efecto).', study(d, RETAIL))
add(G, 'Petróleo cae ≥ 4 % en un día → refino', 'Materia prima más barata.', study(d, REF))
d = idx[since & (ro >= 0.04)]
add(G, 'Petróleo sube ≥ 4 % en un día → energía solar', 'La energía cara hace más atractiva la alternativa (segundo efecto).', study(d, SOLAR))
d = idx[since & (o20 <= -0.15) & (oil.shift(20) > 20)]
add(G, 'Petróleo acumula −15 % en un mes → grandes superficies', 'El ahorro en gasolina llega al gasto del trimestre.', study(d, RETAIL))
d = idx[since & (ty >= 0.12)]
add(G, 'El bono a 10 años sube ≥ 12 pb en un día → inmobiliarias cotizadas (comprar el castigo)', 'Compiten con el bono: apostar a que el castigo es excesivo.', study(d, REIT))
add(G, 'El bono a 10 años sube ≥ 12 pb en un día → eléctricas (comprar el castigo)', 'Compiten con el bono: apostar a que el castigo es excesivo.', study(d, UTIL))
d = idx[since & (ty <= -0.12)]
add(G, 'El bono a 10 años baja ≥ 12 pb en un día → inmobiliarias cotizadas', 'Su dividendo compite mejor con el bono.', study(d, REIT))
add(G, 'El bono a 10 años baja ≥ 12 pb en un día → mineras de oro', 'Tipos más bajos, oro más atractivo (segundo efecto).', study(d, GOLDM))
d = idx[since & (R['GC=F'] <= -0.02)]
add(G, 'Oro cae ≥ 2 % en un día → mineras de oro (comprar la caída)', 'Apostar a que el castigo es excesivo.', study(d, GOLDM))
rb = R['BTC-USD']
add(G, 'Bitcoin sube ≥ 8 % en un día → bolsas y mineras de cripto', 'Más volumen y más valor en balance.', study(idx[since & (rb >= 0.08)], BTCB))
add(G, 'Bitcoin cae ≥ 8 % en un día → bolsas y mineras de cripto (comprar la caída)', 'Apostar a que el castigo es excesivo.', study(idx[since & (rb <= -0.08)], BTCB))
d = idx[since & (R['ZW=F'] >= 0.04)]
add(G, 'Trigo sube ≥ 4 % en un día → fertilizantes', 'El agricultor gana más y abona más (segundo efecto).', study(d, FERT))
add(G, 'Trigo sube ≥ 4 % en un día → comercializadoras y maquinaria agrícola', 'Más margen y más inversión en el campo.', study(d, AGRI))
d = idx[since & (R['NG=F'] <= -0.08)]
add(G, 'Gas natural cae ≥ 8 % en un día → químicas', 'El gas es su materia prima y su energía.', study(d, CHEM))
# castigo o euforia de un sector entero frente al mercado
G = 'Un sector entero frente al mercado'
SECT = 'XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC'.split()
for sgn_, lab, logic in ((-1, 'Un sector cae ≥ 3 % más que el mercado en un día → comprar ese sector', 'El castigo a un sector entero suele ser excesivo.'), (1, 'Un sector sube ≥ 3 % más que el mercado en un día → comprar ese sector', 'La noticia buena de un sector tarda en digerirse.')):
    rows = []
    for e in SECT:
        if e not in col: continue
        rel = R[e] - R['SPY']; b_ = base([e])
        for i in thin(idx[since & ((rel <= -0.03) if sgn_ < 0 else (rel >= 0.03))]):
            rows.append((rel.values[i], after(i, [e], 5) - b_[5], after(i, [e], 20) - b_[20], after(i, [e], 60) - b_[60]))
    add(G, lab, logic, summ(rows))
G = 'Pánico de mercado'
d = idx[since & (R['SPY'] <= -0.03)]
rows = []
for i in thin(d):
    f = lambda h: Av[i + h, col['QQQ']] / Ov[i + 1, col['QQQ']] - 1 if i + 1 + h < len(idx) else np.nan
    rows.append((R.values[i, col['SPY']], f(5), f(20), f(60)))
base_q = {h: float(np.nanmean(Av[h:, col['QQQ']] / Av[:-h, col['QQQ']] - 1)) for h in (5, 20, 60)}
res = summ(rows); res['nota'] = f"rendimiento absoluto del Nasdaq-100 (no exceso); en un tramo cualquiera hace {100*base_q[5]:+.1f} % / {100*base_q[20]:+.1f} % / {100*base_q[60]:+.1f} %"
add(G, 'El S&P 500 cae ≥ 3 % en un día → comprar el Nasdaq-100', 'El pánico vende de más.', res)
def absol(days, sym, nota_sym):
    rows = []
    for i in thin(days):
        f = lambda h: Av[i + h, col[sym]] / Ov[i + 1, col[sym]] - 1 if i + 1 + h < len(idx) else np.nan
        rows.append((R.values[i, col['SPY']], f(5), f(20), f(60)))
    b_ = {h: float(np.nanmean(Av[h:, col[sym]] / Av[:-h, col[sym]] - 1)) for h in (5, 20, 60)}
    r_ = summ(rows); r_['nota'] = f"rendimiento absoluto de {nota_sym} (no exceso); en un tramo cualquiera hace {100*b_[5]:+.1f} % / {100*b_[20]:+.1f} % / {100*b_[60]:+.1f} %"
    return r_
add(G, 'El S&P 500 cae ≥ 2 % en un día → comprar el Nasdaq-100', 'Un mal día normal, no un pánico.', absol(idx[since & (R['SPY'] <= -0.02) & (R['SPY'] > -0.03)], 'QQQ', 'el Nasdaq-100'))
add(G, 'El S&P 500 cae ≥ 3 % en un día → comprar las pequeñas (Russell 2000)', 'Las pequeñas caen más en el pánico.', absol(idx[since & (R['SPY'] <= -0.03)], 'IWM', 'el Russell 2000'))
add(G, 'El índice del miedo (VIX) sube ≥ 30 % en un día → comprar el Nasdaq-100', 'El miedo se dispara de golpe.', absol(idx[since & (R['^VIX'] >= 0.30)], 'QQQ', 'el Nasdaq-100'))
add(G, 'El S&P 500 sube ≥ 3 % en un día → comprar el Nasdaq-100 (perseguir la euforia)', 'Apuntarse a un gran día alcista.', absol(idx[since & (R['SPY'] >= 0.03)], 'QQQ', 'el Nasdaq-100'))
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
# esperar a la prueba: entrar 20 sesiones después de la caída, según lo que haya hecho la acción entretanto
def wait_study(sel, cond):
    rows = []
    for i, s_, peers, r, pm in sel:
        j = col[s_]
        if i + 21 >= len(idx) or not np.isfinite(Av[i + 20, j]) or not np.isfinite(Av[i, j]): continue
        up = Av[i + 20, j] >= Av[i, j]
        if cond is None or cond == up:
            rows.append((r - spy[i], np.nan, after(i + 20, [s_], 20) - ubase(20), after(i + 20, [s_], 60) - ubase(60)))
    return summ(rows)
add(G, 'Una grande cae ≥ 10 % → esperar 20 sesiones y comprar entonces', 'No tener prisa: dejar pasar los titulares.', wait_study(c, None))
add(G, '   …si en esas 20 sesiones NO ha perdido el precio del día de la caída', 'El mercado ha dado por buena la empresa: es la «prueba».', wait_study(c, True))
add(G, '   …si en esas 20 sesiones ha seguido cayendo', 'Sin prueba: el problema sigue vivo.', wait_study(c, False))
# las 50 mayores, con umbral del 7 %
mega = {'caida': [], 'salto': []}
for s_ in uni:
    if s_ not in col or not uni[s_].get('sub'): continue
    peers = [p for p in subs[uni[s_]['sub']] if p != s_]
    if len(peers) < 2: continue
    j = col[s_]
    for i in np.flatnonzero(since & (np.abs(Rv[:, j]) >= 0.07)):
        if i < 61 or not rk[i - 1, j] <= 50 or abs(spy[i]) > 0.02: continue
        mega['caida' if Rv[i, j] < 0 else 'salto'].append((i, s_, peers, Rv[i, j], np.nanmean(Rv[i, [col[p] for p in peers]]) - spy[i]))
add(G, 'Una de las 50 mayores cae ≥ 7 % en un día → comprar la propia caída', 'Los gigantes de verdad no desaparecen por un mal día.', peer_study(mega['caida'], 'self'))
add(G, 'Una de las 50 mayores cae ≥ 7 % en un día → sus competidoras', 'Lo que pierde el gigante lo ganan las otras.', peer_study(mega['caida'], 'peers'))
add(G, 'Una de las 50 mayores sube ≥ 7 % en un día → sus competidoras', 'Lo que anuncia el gigante vale para el sector.', peer_study(mega['salto'], 'peers'))
add(G, 'Una de las 50 mayores sube ≥ 7 % en un día → seguir comprándola', 'La noticia buena tarda en digerirse.', peer_study(mega['salto'], 'self'))
# casos con nombre
# ── afinar: en qué condiciones funciona cada tipo de idea ──
G = 'Afinar · el pánico según el contexto'
SPYc = A['SPY']; above = (SPYc.shift(1) > SPYc.rolling(200).mean().shift(1)).fillna(False).values
vixp = A['^VIX'].shift(1).values; rs = R['SPY'].values
def absol2(mask, sym, nota_sym, at_close=False):
    rows = []; j = col[sym]
    for i in thin(idx[mask]):
        f = lambda h: (Av[i + h, j] / (Av[i, j] if at_close else Ov[i + 1, j]) - 1) if i + 1 + h < len(idx) else np.nan
        rows.append((rs[i], f(5), f(20), f(60)))
    b_ = {h: float(np.nanmean(Av[h:, j] / Av[:-h, j] - 1)) for h in (5, 20, 60)}
    r_ = summ(rows); r_['nota'] = f"rendimiento absoluto de {nota_sym}; un tramo cualquiera hace {100*b_[5]:+.1f} % / {100*b_[20]:+.1f} % / {100*b_[60]:+.1f} %"
    return r_
pan = since & (rs <= -0.02)
add(G, 'S&P 500 −2 % o peor, con el mercado en tendencia alcista → Nasdaq-100', 'El índice estaba por encima de su media de 200 sesiones: susto dentro de una subida.', absol2(pan & above, 'QQQ', 'el Nasdaq-100'))
add(G, 'S&P 500 −2 % o peor, con el mercado en tendencia bajista → Nasdaq-100', 'El índice ya estaba por debajo de su media de 200 sesiones: caída dentro de una bajada.', absol2(pan & ~above, 'QQQ', 'el Nasdaq-100'))
add(G, 'S&P 500 −2 % o peor viniendo de calma (VIX < 20) → Nasdaq-100', 'El golpe pilla al mercado tranquilo.', absol2(pan & (vixp < 20), 'QQQ', 'el Nasdaq-100'))
add(G, 'S&P 500 −2 % o peor con el miedo ya alto (VIX ≥ 28) → Nasdaq-100', 'Otra caída más en plena tormenta.', absol2(pan & (vixp >= 28), 'QQQ', 'el Nasdaq-100'))
add(G, 'S&P 500 −2 % o peor → comprar el Nasdaq-100 al CIERRE de ese mismo día', 'Adelantarse a la apertura siguiente.', absol2(pan, 'QQQ', 'el Nasdaq-100', at_close=True))
add(G, 'S&P 500 −2 % o peor → comprar el Nasdaq-100 en la APERTURA siguiente', 'Esperar a que el cierre confirme la caída.', absol2(pan, 'QQQ', 'el Nasdaq-100'))
add(G, 'S&P 500 entre −1,5 % y −2 % → Nasdaq-100', 'Un mal día que no llega al umbral.', absol2(since & (rs <= -0.015) & (rs > -0.02), 'QQQ', 'el Nasdaq-100'))
r5 = (SPYc / SPYc.shift(5) - 1).values
add(G, 'S&P 500 pierde ≥ 5 % en cinco sesiones → Nasdaq-100', 'Una mala semana entera.', absol2(since & (r5 <= -0.05), 'QQQ', 'el Nasdaq-100'))
dd = (SPYc / SPYc.rolling(60).max() - 1).values; ddp = np.r_[np.nan, dd[:-1]]
add(G, 'S&P 500 entra en corrección (−10 % desde su máximo de 60 sesiones) → Nasdaq-100', 'El primer día que cruza el −10 %.', absol2(since & (dd <= -0.10) & (ddp > -0.10), 'QQQ', 'el Nasdaq-100'))
vx = A['^VIX'].values
add(G, 'El VIX cierra por encima de 30 por primera vez en 60 sesiones → Nasdaq-100', 'El miedo llega a niveles de crisis.', absol2(since & (vx > 30) & (pd.Series(vx).shift(1).rolling(60).max().values <= 30), 'QQQ', 'el Nasdaq-100'))
add(G, 'Los bonos basura caen ≥ 1,2 % en un día → Nasdaq-100', 'El crédito avisa antes que la bolsa.', absol2(since & (R['HYG'].values <= -0.012), 'QQQ', 'el Nasdaq-100'))

G = 'Afinar · tropiezos y saltos según el contexto'
OFF = (A.shift(1) / A.rolling(252, min_periods=120).max().shift(1) - 1).values
off = lambda e: OFF[e[0], col[e[1]]]
add(G, 'Cae ≥ 10 % estando cerca de máximos (a menos de un 10 %) → comprar la caída', 'Estaba cotizada para la perfección.', peer_study([e for e in c if off(e) > -0.10], 'self'))
add(G, 'Cae ≥ 10 % cuando ya venía hundida (más de un 30 % bajo máximos) → comprar la caída', 'Otra mala noticia sobre una acción ya castigada.', peer_study([e for e in c if off(e) < -0.30], 'self'))
add(G, 'Cae ≥ 10 % estando cerca de máximos → sus competidoras', 'El tropiezo del que iba primero.', peer_study([e for e in c if off(e) > -0.10], 'peers'))
add(G, 'Cae entre 10 % y 15 % → sus competidoras', 'Tropiezo moderado.', peer_study([e for e in c if e[3] > -0.15], 'peers'))
add(G, 'Cae entre 15 % y 25 % → sus competidoras', 'Tropiezo serio.', peer_study([e for e in c if -0.25 < e[3] <= -0.15], 'peers'))
add(G, 'Cae más de un 25 % → sus competidoras', 'Desplome.', peer_study([e for e in c if e[3] <= -0.25], 'peers'))
add(G, 'Cae ≥ 10 % en un día en que el mercado sube → sus competidoras', 'El problema es solo suyo.', peer_study([e for e in c if spy[e[0]] > 0.003], 'peers'))
add(G, 'Cae ≥ 10 % en un día en que el mercado baja → sus competidoras', 'El problema se mezcla con el mal día.', peer_study([e for e in c if spy[e[0]] < -0.003], 'peers'))
add(G, 'Sube ≥ 10 % y marca máximo anual → seguir comprándola', 'Ruptura con noticia.', peer_study([e for e in u_ if Av[e[0], col[e[1]]] >= np.nanmax(Av[max(0, e[0] - 252):e[0], col[e[1]]])], 'self'))
add(G, 'Sube ≥ 10 % viniendo hundida (más de un 30 % bajo máximos) → seguir comprándola', 'El rebote del castigado.', peer_study([e for e in u_ if off(e) < -0.30], 'self'))
add(G, 'Sube ≥ 15 % en un día → sus competidoras quietas (< 1 %)', 'Noticia muy buena que el sector aún no recoge.', peer_study([e for e in u_ if e[3] >= 0.15 and abs(e[4]) < 0.01], 'peers'))

G = 'Afinar · movimientos de varios días'
def acc(sym, n): return (A[sym] / A[sym].shift(n) - 1)
o5 = acc('CL=F', 5).where(oil.shift(5) > 20)
add(G, 'Petróleo acumula −10 % en cinco sesiones → aerolíneas', 'Bajada de una semana, no de un día.', study(idx[since & (o5 <= -0.10)], AIR))
add(G, 'Petróleo acumula −10 % en cinco sesiones → cruceros', 'Combustible y consumo.', study(idx[since & (o5 <= -0.10)], CRU))
add(G, 'Petróleo acumula +10 % en cinco sesiones → petroleras', 'Subida de una semana.', study(idx[since & (o5 >= 0.10)], EP))
add(G, 'Petróleo acumula +10 % en cinco sesiones → refino', 'Margen de refino.', study(idx[since & (o5 >= 0.10)], REF))
add(G, 'Petróleo acumula +10 % en cinco sesiones → aerolíneas (comprar el castigo)', 'Apostar a que el castigo es excesivo.', study(idx[since & (o5 >= 0.10)], AIR))
add(G, 'Oro acumula +5 % en diez sesiones → mineras de oro', 'Subida sostenida.', study(idx[since & (acc('GC=F', 10) >= 0.05)], GOLDM))
t10 = A['^TNX'].diff(10)
add(G, 'El bono a 10 años sube ≥ 30 pb en diez sesiones → bancos', 'Tipos al alza de forma sostenida.', study(idx[since & (t10 >= 0.30)], BANK))
add(G, 'El bono a 10 años sube ≥ 30 pb en diez sesiones → constructoras (comprar el castigo)', 'Hipotecas más caras.', study(idx[since & (t10 >= 0.30)], HB))
add(G, 'El bono a 10 años baja ≥ 30 pb en diez sesiones → constructoras', 'Hipotecas más baratas.', study(idx[since & (t10 <= -0.30)], HB))
add(G, 'El bono a 10 años baja ≥ 30 pb en diez sesiones → inmobiliarias cotizadas', 'Dividendo frente a bono.', study(idx[since & (t10 <= -0.30)], REIT))
add(G, 'Cobre acumula +7 % en diez sesiones → mineras de cobre', 'Subida sostenida.', study(idx[since & (acc('HG=F', 10) >= 0.07)], COP_))
add(G, 'Gas natural acumula +15 % en cinco sesiones → productoras de gas', 'Subida de una semana.', study(idx[since & (acc('NG=F', 5) >= 0.15)], GAS))
add(G, 'Trigo acumula +10 % en diez sesiones → fertilizantes', 'El campo gana más durante semanas.', study(idx[since & (acc('ZW=F', 10) >= 0.10)], FERT))
add(G, 'Bitcoin acumula +15 % en cinco sesiones → bolsas y mineras de cripto', 'Subida de una semana.', study(idx[since & (acc('BTC-USD', 5) >= 0.15)], BTCB))
add(G, 'El dólar sube ≥ 2 % en cinco sesiones → emergentes (comprar el castigo)', 'Dólar fuerte castiga a los emergentes.', study(idx[since & (acc('UUP', 5) >= 0.02)], ['EEM']))

G = 'Cuando se mueve un líder con nombre'
def lider(sym, thr, basket, nombre, logica):
    r_ = R[sym].values
    m = since & ((r_ >= thr) if thr > 0 else (r_ <= thr)) & (np.abs(rs) < 0.02)
    add(G, nombre, logica, study(idx[m], basket))
SEMI = 'AMD AVGO MU MRVL'.split()
lider('NVDA', 0.07, SEMI, 'Nvidia sube ≥ 7 % en un día → otros fabricantes de chips', 'Lo que anuncia el líder vale para el sector.')
lider('NVDA', -0.07, SEMI, 'Nvidia cae ≥ 7 % en un día → otros fabricantes de chips', '¿Contagio que se recupera o aviso para todos?')
lider('NVDA', -0.07, ['NVDA'], 'Nvidia cae ≥ 7 % en un día → comprar Nvidia', 'El líder castigado.')
lider('AAPL', -0.04, 'QCOM SWKS AVGO QRVO'.split(), 'Apple cae ≥ 4 % en un día → sus proveedores', 'Arrastra a quien le vende piezas.')
lider('AAPL', 0.04, 'QCOM SWKS AVGO QRVO'.split(), 'Apple sube ≥ 4 % en un día → sus proveedores', 'Buenas noticias aguas arriba.')
lider('TSLA', 0.08, ['GM', 'F'], 'Tesla sube ≥ 8 % en un día → fabricantes tradicionales', 'Lo que gana una lo pierden las otras, o al revés.')
lider('TSLA', -0.08, ['GM', 'F'], 'Tesla cae ≥ 8 % en un día → fabricantes tradicionales', 'Rival del que tropieza.')
lider('JPM', -0.04, ['KRE'], 'JPMorgan cae ≥ 4 % en un día → bancos regionales', 'Si tose el grande, ¿se resfrían los pequeños?')
lider('WMT', -0.05, ['TGT', 'COST'], 'Walmart cae ≥ 5 % en un día → Target y Costco', 'Aviso sobre el consumidor o problema propio.')
lider('AMZN', 0.07, ['SHOP', 'EBAY', 'ETSY'], 'Amazon sube ≥ 7 % en un día → otro comercio electrónico', 'Marea que sube todos los barcos.')
lider('META', 0.08, ['GOOGL', 'SNAP', 'PINS'], 'Meta sube ≥ 8 % en un día → otra publicidad digital', 'El mercado publicitario va bien para todos.')
lider('META', -0.08, ['GOOGL', 'SNAP', 'PINS'], 'Meta cae ≥ 8 % en un día → otra publicidad digital', 'Rival o contagio.')
lider('UNH', -0.05, ['ELV', 'CI', 'HUM'], 'UnitedHealth cae ≥ 5 % en un día → otras aseguradoras', 'Suele ser un problema del sector entero.')
lider('LLY', 0.06, ['NVO'], 'Lilly sube ≥ 6 % en un día → Novo Nordisk', 'El mismo mercado, el otro jugador.')
lider('BA', -0.05, ['EADSY'], 'Boeing cae ≥ 5 % en un día → Airbus', 'Duopolio: lo que pierde uno lo gana el otro.')
lider('MSFT', 0.05, ['ORCL', 'CRM', 'NOW'], 'Microsoft sube ≥ 5 % en un día → otro software', 'El gasto en software va bien.')
lider('HD', -0.05, ['LOW'], 'Home Depot cae ≥ 5 % en un día → Lowe\'s', 'Mismo negocio, mismo cliente.')
lider('XOM', -0.04, REF, 'Exxon cae ≥ 4 % en un día → refino', 'Petróleo abajo, materia prima más barata.')

G = 'Casos con nombre'
TIPOS = {'E': 'Hecho consumado que cambia el mundo (guerra, paz, vacuna, elecciones, fármaco)', 'D': 'Decisión anunciada (banco central, ley firmada, plan de empresa)', 'S': 'Susto o anuncio de un día', 'N': 'Sector castigado por una norma o sanción → comprar el castigo', 'V': 'Víctimas de la novedad de otro → comprar el castigo', 'C': 'Comprar al gigante caído', 'R': 'Competencia del que tropieza', 'B': 'Buena noticia que se contagia'}
CASES = [
 # E · cambios duraderos
 ('2014-11-28', 'La OPEP no recorta (petróleo barato durante años) → aerolíneas', 'DAL UAL AAL LUV ALK', 'D'),
 ('2016-11-09', 'Trump gana (2016) → bancos y defensa', 'JPM BAC GS LMT NOC GD', 'E'),
 ('2016-11-09', 'Trump gana (2016) → infraestructuras y acero', 'VMC MLM NUE X', 'E'),
 ('2020-03-16', 'Confinamientos → quedarse en casa', 'ZM NFLX AMZN CLX PTON', 'E'),
 ('2020-11-09', 'Vacuna de Pfizer (2020) → aerolíneas y cruceros', 'DAL UAL AAL CCL RCL NCLH', 'E'),
 ('2020-11-09', 'Vacuna de Pfizer (2020) → energía y bancos', 'XOM CVX JPM BAC', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → defensa', 'LMT NOC RTX GD', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → fertilizantes y energía', 'MOS CF NTR XOM CVX', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → comercializadoras y maquinaria agrícola', 'ADM BG DE', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → ciberseguridad', 'CRWD PANW FTNT', 'E'),
 ('2022-07-28', 'Acuerdo para la ley del clima de EE. UU. (2022) → solar', 'ENPH FSLR SEDG RUN', 'D'),
 ('2022-11-30', 'Sale ChatGPT → Microsoft y Nvidia', 'MSFT NVDA', 'D'),
 ('2023-08-08', 'El fármaco de la obesidad protege el corazón → Lilly y Novo', 'LLY NVO', 'E'),
 ('2024-09-18', 'Primera bajada de tipos de la Fed (2024) → constructoras y pequeñas', 'DHI LEN IWM', 'D'),
 ('2024-11-06', 'Trump gana (2024) → bancos, prisiones privadas y cripto', 'JPM GS GEO CXW COIN', 'E'),
 ('2024-11-06', 'Trump gana (2024) → Tesla', 'TSLA', 'E'),
 ('2024-11-06', 'Trump gana (2024) → pequeñas y bancos regionales', 'IWM KRE', 'E'),
 ('2025-01-22', 'Anuncio de Stargate (2025) → infraestructura de IA', 'ORCL VRT ANET', 'D'),
 ('2025-06-24', 'Alto el fuego entre Israel e Irán (2025) → aerolíneas y cruceros', 'DAL UAL AAL CCL RCL', 'E'),
 # S · sustos y anuncios de un día
 ('2011-03-14', 'Fukushima → solar', 'FSLR', 'S'),
 ('2012-10-31', 'Huracán Sandy → reformas del hogar', 'HD LOW', 'S'),
 ('2016-06-24', 'Brexit → mineras de oro', 'NEM GDX', 'S'),
 ('2017-08-28', 'Huracán Harvey → reformas del hogar', 'HD LOW', 'S'),
 ('2018-03-01', 'Aranceles al acero (2018) → acereras de EE. UU.', 'NUE STLD CLF X', 'S'),
 ('2019-09-16', 'Ataque a la refinería saudí de Abqaiq → petroleras', 'XOM CVX COP OXY', 'S'),
 ('2020-04-20', 'El petróleo cotiza en negativo → comprar petroleras', 'XOM CVX COP', 'S'),
 ('2021-11-26', 'Variante ómicron → fabricantes de vacunas', 'MRNA PFE BNTX', 'S'),
 ('2021-11-26', 'Variante ómicron → comprar las aerolíneas castigadas', 'DAL UAL AAL', 'S'),
 ('2022-09-29', 'Huracán Ian → reformas y generadores', 'HD LOW GNRC', 'S'),
 ('2023-03-10', 'Quiebra de SVB → bancos grandes (refugio de depósitos)', 'JPM BAC WFC C', 'S'),
 ('2023-03-13', 'Quiebra de SVB → regionales (comprar el pánico)', 'KRE', 'S'),
 ('2023-10-09', 'Ataque de Hamás → defensa', 'LMT NOC RTX GD', 'S'),
 ('2023-10-09', 'Ataque de Hamás → petroleras', 'XOM CVX', 'S'),
 ('2025-04-09', 'Pausa de los aranceles (2025) → los más castigados', 'AAPL NKE BBY RH DECK', 'S'),
 ('2025-05-12', 'Tregua comercial EE. UU.–China (2025) → importadores', 'NKE BBY AAPL RH', 'S'),
 ('2025-06-13', 'Israel ataca Irán (2025) → defensa y petróleo', 'LMT NOC RTX XOM CVX', 'S'),
 # C · comprar al gigante caído
 ('2010-04-16', 'La SEC acusa a Goldman Sachs → comprar Goldman', 'GS', 'C'),
 ('2015-11-02', 'Brote de E. coli en Chipotle → comprar Chipotle', 'CMG', 'C'),
 ('2016-09-08', 'Cuentas falsas de Wells Fargo → comprar Wells Fargo', 'WFC', 'C'),
 ('2017-04-11', 'United arrastra a un pasajero → comprar United', 'UAL', 'C'),
 ('2017-09-08', 'Robo de datos en Equifax → comprar Equifax', 'EFX', 'C'),
 ('2018-03-19', 'Cambridge Analytica → comprar Facebook', 'META', 'C'),
 ('2018-07-26', 'Facebook se hunde un 19 % → comprar Facebook', 'META', 'C'),
 ('2018-12-14', 'Talco de Johnson & Johnson → comprar J&J', 'JNJ', 'C'),
 ('2019-03-11', 'Segundo accidente del 737 MAX → comprar Boeing', 'BA', 'C'),
 ('2022-02-02', 'PayPal se hunde un 25 % → comprar PayPal', 'PYPL', 'C'),
 ('2022-02-03', 'Meta se hunde un 26 % → comprar Meta', 'META', 'C'),
 ('2022-04-20', 'Netflix se hunde un 35 % → comprar Netflix', 'NFLX', 'C'),
 ('2022-05-18', 'Target se hunde un 25 % → comprar Target', 'TGT', 'C'),
 ('2022-10-27', 'Meta se hunde un 25 % (otoño de 2022) → comprar Meta', 'META', 'C'),
 ('2023-02-08', 'Error de Bard en su presentación → comprar Alphabet', 'GOOGL', 'C'),
 ('2024-01-08', 'Boeing pierde una puerta en vuelo → comprar Boeing', 'BA', 'C'),
 ('2024-06-28', 'Nike se hunde un 20 % → comprar Nike', 'NKE', 'C'),
 ('2024-07-19', 'Apagón de CrowdStrike → comprar la propia CrowdStrike', 'CRWD', 'C'),
 ('2024-08-02', 'Intel se hunde un 26 % → comprar Intel', 'INTC', 'C'),
 ('2025-01-27', 'DeepSeek hunde a Nvidia → comprar Nvidia', 'NVDA', 'C'),
 ('2025-04-17', 'UnitedHealth se hunde un 22 % → comprar UnitedHealth', 'UNH', 'C'),
 # R · competencia del que tropieza
 ('2015-11-02', 'Brote en Chipotle → otras cadenas', 'MCD YUM DPZ', 'R'),
 ('2016-09-08', 'Cuentas falsas de Wells Fargo → otros bancos', 'JPM BAC USB', 'R'),
 ('2017-09-08', 'Robo de datos en Equifax → TransUnion', 'TRU', 'R'),
 ('2019-03-11', 'Segundo accidente del 737 MAX → Airbus', 'EADSY', 'R'),
 ('2022-02-03', 'Meta se hunde → otras redes de publicidad', 'GOOGL SNAP PINS', 'R'),
 ('2022-04-20', 'Netflix se hunde → Disney', 'DIS', 'R'),
 ('2022-05-18', 'Target se hunde → Walmart y Costco', 'WMT COST', 'R'),
 ('2024-01-08', 'Boeing pierde una puerta → Airbus', 'EADSY', 'R'),
 ('2024-06-28', 'Nike se hunde → Deckers y On', 'DECK ONON', 'R'),
 ('2024-07-19', 'Apagón de CrowdStrike → competidoras', 'S PANW ZS FTNT', 'R'),
 ('2024-08-02', 'Intel se hunde → AMD', 'AMD', 'R'),
 ('2025-04-17', 'UnitedHealth se hunde → otras aseguradoras', 'ELV CI HUM', 'R'),
 # B · buena noticia que se contagia
 ('2023-02-02', 'Meta anuncia el «año de la eficiencia» (+23 %) → otras redes', 'GOOGL SNAP PINS', 'B'),
 ('2023-05-25', 'Nvidia dispara su guía de IA (2023) → cadena de la IA', 'AVGO AMD MRVL ANET SMCI MU', 'B'),
 ('2024-02-22', 'Resultados de Nvidia (feb-2024) → cadena de la IA', 'AMD AVGO SMCI MRVL', 'B'),
 ('2025-01-27', 'DeepSeek (IA barata) → software que usa IA', 'CRM NOW MSFT ADBE PLTR', 'B'),
 # ── ampliación ──
 ('2008-10-03', 'Se firma el rescate bancario (TARP) → bancos', 'JPM BAC WFC C', 'D'),
 ('2010-03-23', 'Se firma la reforma sanitaria de Obama → hospitales', 'THC UHS', 'D'),
 ('2012-07-26', 'Draghi: «lo que haga falta» → bancos europeos', 'SAN DB ING', 'D'),
 ('2012-09-13', 'La Fed lanza la tercera ronda de compras → constructoras', 'DHI LEN PHM', 'D'),
 ('2013-05-22', 'La Fed avisa de que retirará estímulos → bancos', 'JPM BAC WFC', 'D'),
 ('2015-12-16', 'Primera subida de tipos en nueve años → bancos', 'JPM BAC WFC', 'D'),
 ('2016-11-30', 'La OPEP recorta producción → petroleras', 'XOM CVX COP OXY', 'D'),
 ('2017-12-20', 'El Congreso aprueba la rebaja fiscal → pequeñas y bancos regionales', 'IWM KRE', 'D'),
 ('2019-01-04', 'Powell se declara «paciente» → Nasdaq y pequeñas', 'QQQ IWM', 'D'),
 ('2020-03-23', 'La Fed compra sin límite → Nasdaq', 'QQQ', 'D'),
 ('2020-03-27', 'Se firma el rescate por la pandemia → aerolíneas', 'DAL UAL AAL', 'D'),
 ('2021-01-06', 'Los demócratas ganan el Senado → solar', 'TAN FSLR ENPH', 'E'),
 ('2021-11-05', 'La píldora de Pfizer contra la covid funciona → aerolíneas y cruceros', 'DAL UAL CCL', 'E'),
 ('2022-03-16', 'Primera subida de tipos de 2022 → bancos', 'JPM BAC WFC', 'D'),
 ('2022-08-09', 'Se firma la ley de chips → fabricantes y equipos', 'INTC MU AMAT LRCX', 'D'),
 ('2022-12-27', 'China levanta la cuarentena → casinos de Macao', 'LVS WYNN', 'E'),
 ('2023-11-14', 'La inflación cede (nov-2023) → pequeñas y constructoras', 'IWM DHI LEN', 'D'),
 ('2023-12-13', 'La Fed anuncia bajadas para 2024 → pequeñas, regionales y constructoras', 'IWM KRE DHI', 'D'),
 ('2014-03-03', 'Rusia ocupa Crimea → defensa', 'LMT NOC GD', 'S'),
 ('2015-11-16', 'Atentados de París → defensa', 'LMT NOC GD', 'S'),
 ('2018-09-14', 'Huracán Florence → reformas del hogar', 'HD LOW', 'S'),
 ('2020-01-03', 'EE. UU. mata a Soleimani → defensa y petróleo', 'LMT NOC XOM CVX', 'S'),
 ('2021-02-16', 'Ola de frío en Texas → generadores', 'GNRC', 'S'),
 ('2021-05-10', 'Ciberataque al oleoducto Colonial → ciberseguridad', 'CRWD PANW FTNT', 'S'),
 ('2021-08-30', 'Huracán Ida → reformas y generadores', 'HD LOW GNRC', 'S'),
 ('2022-03-08', 'EE. UU. veta el petróleo ruso → petroleras', 'XOM CVX OXY', 'S'),
 ('2023-05-01', 'Cae First Republic → regionales (comprar el pánico)', 'KRE', 'S'),
 ('2024-04-15', 'Irán ataca Israel (abr-2024) → defensa y petróleo', 'LMT NOC XOM', 'S'),
 ('2024-09-24', 'China anuncia estímulos → tecnológicas chinas y cobre', 'BABA JD BIDU FCX', 'S'),
 ('2024-10-09', 'Huracán Milton → reformas y generadores', 'HD LOW GNRC', 'S'),
 ('2025-02-03', 'Aranceles a Canadá y México → comprar las automovilísticas castigadas', 'GM F', 'S'),
 ('2015-09-21', 'Clinton amenaza los precios de los fármacos → comprar biotecnología', 'IBB XBI', 'N'),
 ('2019-04-16', 'Miedo a la sanidad pública universal → comprar aseguradoras', 'UNH ELV HUM CI', 'N'),
 ('2019-05-20', 'Veto a Huawei → comprar a sus proveedores de chips', 'QCOM MU SWKS QRVO', 'N'),
 ('2021-07-26', 'Pekín reprime a sus tecnológicas → comprar las chinas cotizadas en EE. UU.', 'BABA JD BIDU', 'N'),
 ('2022-10-07', 'Primer veto de chips avanzados a China → comprar chips', 'NVDA AMD LRCX AMAT', 'N'),
 ('2023-10-17', 'Segundo veto de chips a China → comprar chips', 'NVDA AMD', 'N'),
 ('2025-04-16', 'Veto al chip H20 de Nvidia para China → comprar chips', 'NVDA AMD', 'N'),
 ('2017-06-16', 'Amazon compra Whole Foods → comprar los supermercados castigados', 'KR COST WMT TGT', 'V'),
 ('2018-06-28', 'Amazon compra PillPack → comprar las farmacias castigadas', 'CVS WBA', 'V'),
 ('2023-10-11', 'El fármaco de la obesidad protege el riñón → comprar diálisis castigada', 'DVA BAX', 'V'),
 ('2025-01-27', 'DeepSeek (IA que gasta menos) → comprar las eléctricas castigadas', 'VST CEG', 'V'),
 ('2010-04-21', 'Explota la plataforma Deepwater Horizon → comprar BP', 'BP', 'C'),
 ('2011-10-25', 'Netflix se hunde un 35 % (2011) → comprar Netflix', 'NFLX', 'C'),
 ('2012-05-11', 'La «ballena de Londres» de JPMorgan → comprar JPMorgan', 'JPM', 'C'),
 ('2013-01-24', 'Apple cae un 12 % tras resultados (2013) → comprar Apple', 'AAPL', 'C'),
 ('2015-10-21', 'Informe contra Valeant → comprar Valeant', 'BHC', 'C'),
 ('2017-11-13', 'General Electric recorta el dividendo → comprar GE', 'GE', 'C'),
 ('2018-09-28', 'La SEC demanda a Musk → comprar Tesla', 'TSLA', 'C'),
 ('2018-11-16', 'Nvidia se hunde un 19 % (resaca cripto) → comprar Nvidia', 'NVDA', 'C'),
 ('2019-01-03', 'Apple avisa de menos ventas → comprar Apple', 'AAPL', 'C'),
 ('2020-07-24', 'Intel retrasa sus chips de 7 nm → comprar Intel', 'INTC', 'C'),
 ('2021-11-03', 'Zillow abandona la compra de casas → comprar Zillow', 'Z', 'C'),
 ('2022-01-21', 'Netflix se hunde un 22 % (ene-2022) → comprar Netflix', 'NFLX', 'C'),
 ('2022-04-29', 'Amazon cae un 14 % tras resultados → comprar Amazon', 'AMZN', 'C'),
 ('2022-05-17', 'Walmart cae un 11 % → comprar Walmart', 'WMT', 'C'),
 ('2022-09-15', 'Adobe compra Figma y cae un 17 % → comprar Adobe', 'ADBE', 'C'),
 ('2022-11-09', 'Disney cae un 13 % tras resultados → comprar Disney', 'DIS', 'C'),
 ('2023-05-02', 'Chegg admite que ChatGPT le quita clientes → comprar Chegg', 'CHGG', 'C'),
 ('2024-02-21', 'Palo Alto se hunde un 28 % → comprar Palo Alto', 'PANW', 'C'),
 ('2024-04-25', 'Meta cae un 10 % por su gasto en IA → comprar Meta', 'META', 'C'),
 ('2024-05-01', 'Starbucks se hunde un 16 % → comprar Starbucks', 'SBUX', 'C'),
 ('2025-05-13', 'UnitedHealth pierde a su consejero delegado → comprar UnitedHealth', 'UNH', 'C'),
 ('2010-04-21', 'Explota la plataforma de BP → otras petroleras', 'XOM CVX', 'R'),
 ('2012-05-11', 'La «ballena de Londres» → otros bancos', 'BAC WFC C', 'R'),
 ('2018-03-19', 'Cambridge Analytica → Alphabet y Snap', 'GOOGL SNAP', 'R'),
 ('2018-07-26', 'Facebook se hunde un 19 % → Alphabet y Snap', 'GOOGL SNAP', 'R'),
 ('2018-09-28', 'La SEC demanda a Musk → GM y Ford', 'GM F', 'R'),
 ('2020-07-24', 'Intel retrasa sus chips → AMD', 'AMD', 'R'),
 ('2022-01-21', 'Netflix se hunde (ene-2022) → Disney', 'DIS', 'R'),
 ('2022-10-27', 'Meta se hunde (otoño de 2022) → Alphabet, Pinterest y Snap', 'GOOGL PINS SNAP', 'R'),
 ('2024-02-21', 'Palo Alto se hunde → otras de ciberseguridad', 'CRWD ZS FTNT', 'R'),
 ('2024-05-01', 'Starbucks se hunde → McDonald\'s y Chipotle', 'MCD CMG', 'R'),
 ('2023-08-24', 'Resultados de Nvidia (ago-2023) → cadena de la IA', 'AMD AVGO MRVL', 'B'),
 ('2024-05-23', 'Resultados de Nvidia (may-2024) → cadena de la IA', 'AMD AVGO MRVL SMCI', 'B'),
 ('2025-09-10', 'Oracle dispara su cartera de pedidos de IA → cadena de la IA', 'AVGO NVDA MU', 'B'),
]
done = []
for d, name, b, tp in CASES:
    d = pd.Timestamp(d)
    if d not in pos: d = idx[idx.searchsorted(d)]
    i = pos[d]; b = [x for x in b.split() if x in col and np.isfinite(Av[i, col[x]])]
    if not b: print('  sin datos:', name); continue
    v = lambda x: None if not np.isfinite(x) else round(100 * x, 2)
    done.append({'n': 1, 'd0': v(day0(i, b)), 'e5': v(after(i, b, 5)), 'e20': v(after(i, b, 20)), 'e60': v(after(i, b, 60)), 'valores': b, 'fecha': str(d.date()), 'tipo': tp, 'nombre': name})
G = 'Casos con nombre · resumen por tipo'
for tp, lab in TIPOS.items():
    cs = [c for c in done if c['tipo'] == tp]
    m = lambda k: (lambda x: round(float(np.mean(x)), 2) if x else None)([c[k] for c in cs if c[k] is not None])
    h = lambda k: (lambda x: int(round(100 * np.mean([y > 0 for y in x]))) if x else None)([c[k] for c in cs if c[k] is not None])
    add(G, lab, f'{len(cs)} casos elegidos a mano', {'n': len(cs), 'd0': m('d0'), 'e5': m('e5'), 'e20': m('e20'), 'e60': m('e60'), 'hit20': h('e20'), 'hit60': h('e60'), 'tipo': tp})
G = 'Casos con nombre'
for c in sorted(done, key=lambda c: ('EDSNVCRB'.index(c['tipo']), c['fecha'])):
    add(G, c['nombre'], TIPOS[c['tipo']].split(' (')[0], {k: v for k, v in c.items() if k != 'nombre'})
for o in OUT:
    o['nombre'] = '↳ ' + o['nombre'].strip() if o['nombre'].startswith('   ') else o['nombre'].strip()
pb = os.path.join(ARGS.out, 'ideas_playbook.json')
P = json.load(open(pb, encoding='utf-8')) if os.path.exists(pb) else {'lecciones': [], 'metodo': '', 'cautelas': '', 'resumen': ''}
P['estudios'] = OUT
P['generado'] = time.strftime('%Y-%m-%d')
json.dump(P, open(pb, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
