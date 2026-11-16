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
         'XLK XLF XLE XLV XLY XLP XLI XLB XLRE XLC NVO BNTX EADSY ONON SEDG RUN').split()
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
base = {h: float(np.nanmean(Av[h:, col['QQQ']] / Av[:-h, col['QQQ']] - 1)) for h in (5, 20, 60)}
res = summ(rows); res['nota'] = f"rendimiento absoluto del Nasdaq-100 (no exceso); en un tramo cualquiera hace {100*base[5]:+.1f} % / {100*base[20]:+.1f} % / {100*base[60]:+.1f} %"
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
G = 'Casos con nombre'
TIPOS = {'E': 'Cambio duradero (guerra, paz, vacuna, gobierno, ley, tecnología)', 'S': 'Susto o anuncio de un día', 'C': 'Comprar al gigante caído', 'R': 'Competencia del que tropieza', 'B': 'Buena noticia que se contagia'}
CASES = [
 # E · cambios duraderos
 ('2014-11-28', 'La OPEP no recorta (petróleo barato durante años) → aerolíneas', 'DAL UAL AAL LUV ALK', 'E'),
 ('2016-11-09', 'Trump gana (2016) → bancos y defensa', 'JPM BAC GS LMT NOC RTX', 'E'),
 ('2016-11-09', 'Trump gana (2016) → infraestructuras y acero', 'VMC MLM NUE X', 'E'),
 ('2020-03-16', 'Confinamientos → quedarse en casa', 'ZM NFLX AMZN CLX PTON', 'E'),
 ('2020-11-09', 'Vacuna de Pfizer (2020) → aerolíneas y cruceros', 'DAL UAL AAL CCL RCL NCLH', 'E'),
 ('2020-11-09', 'Vacuna de Pfizer (2020) → energía y bancos', 'XOM CVX JPM BAC', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → defensa', 'LMT NOC RTX GD', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → fertilizantes y energía', 'MOS CF NTR XOM CVX', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → comercializadoras y maquinaria agrícola', 'ADM BG DE', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → ciberseguridad', 'CRWD PANW FTNT', 'E'),
 ('2022-07-28', 'Acuerdo para la ley del clima de EE. UU. (2022) → solar', 'ENPH FSLR SEDG RUN', 'E'),
 ('2022-11-30', 'Sale ChatGPT → Microsoft y Nvidia', 'MSFT NVDA', 'E'),
 ('2023-08-08', 'El fármaco de la obesidad protege el corazón → Lilly y Novo', 'LLY NVO', 'E'),
 ('2024-09-18', 'Primera bajada de tipos de la Fed (2024) → constructoras y pequeñas', 'DHI LEN IWM', 'E'),
 ('2024-11-06', 'Trump gana (2024) → bancos, prisiones privadas y cripto', 'JPM GS GEO CXW COIN', 'E'),
 ('2024-11-06', 'Trump gana (2024) → Tesla', 'TSLA', 'E'),
 ('2024-11-06', 'Trump gana (2024) → pequeñas y bancos regionales', 'IWM KRE', 'E'),
 ('2025-01-22', 'Anuncio de Stargate (2025) → infraestructura de IA', 'ORCL VRT ANET', 'E'),
 ('2025-06-24', 'Alto el fuego entre Israel e Irán (2025) → aerolíneas y cruceros', 'DAL UAL AAL CCL RCL', 'E'),
 # S · sustos y anuncios de un día
 ('2011-03-14', 'Fukushima → solar', 'FSLR', 'S'),
 ('2012-10-31', 'Huracán Sandy → reformas del hogar', 'HD LOW', 'S'),
 ('2016-06-24', 'Brexit → mineras de oro', 'NEM GDX', 'S'),
 ('2017-06-16', 'Amazon compra Whole Foods → comprar los supermercados castigados', 'KR COST WMT TGT', 'S'),
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
for c in sorted(done, key=lambda c: ('ESCRB'.index(c['tipo']), c['fecha'])):
    add(G, c['nombre'], TIPOS[c['tipo']].split(' (')[0], {k: v for k, v in c.items() if k != 'nombre'})
for o in OUT:
    o['nombre'] = '↳ ' + o['nombre'].strip() if o['nombre'].startswith('   ') else o['nombre'].strip()
pb = os.path.join(ARGS.out, 'ideas_playbook.json')
P = json.load(open(pb, encoding='utf-8')) if os.path.exists(pb) else {'lecciones': [], 'metodo': '', 'cautelas': '', 'resumen': ''}
P['estudios'] = OUT
P['generado'] = time.strftime('%Y-%m-%d')
json.dump(P, open(pb, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
