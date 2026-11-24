"""Entrenamiento de las ideas por coyuntura: simula cada tipo de idea sobre hechos pasados y guarda los ensayos en data/ideas_playbook.json.

Para cada hecho (un desplome del petróleo, una grande que cae un 10 %, un caso con nombre) se compra la cesta «lógica» en la APERTURA del día siguiente —cuando de verdad se puede actuar—
y se mide lo que gana de más sobre el SPY a 5, 20 y 60 sesiones, restando lo que esa cesta hace en un día cualquiera. «d0» es lo que se movió el mismo día de la noticia (ya no se puede comprar).
Los hechos de un mismo tipo a menos de 10 sesiones del anterior se descartan para no contar dos veces el mismo episodio.

Uso: python -m vp.ideas_train [--out data] [--cache .cache/ideas]. Solo reescribe la lista «estudios» del fichero: las lecciones las redacta el analista a la vista de las cifras.
Cautelas: universo = miembros ACTUALES del S&P 1500 (faltan los excluidos y quebrados) y los casos con nombre están elegidos a mano."""
from __future__ import annotations
import argparse
import re
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
         'BP SAN DB ING LYG BCS LVS WYNN BABA JD BIDU IBB XBI VST CEG WBA Z CHGG BHC EEM HYG UUP SHOP EBAY ETSY '
         'SMH JETS ITA XHB XOP XRT IYT FXI CHKP DOCU BOX EA TTWO FANG XPO TM GDRX DLTR '
         'ROST BURL KSS M DDS BBY CVX MPC KO PFE MRK ABBV BMY AZN VZ T CMCSA CHTR UPS FDX CSX UNP NSC '
         'KC=F CC=F SB=F ZC=F LE=F SBUX HSY MDLZ TSN HRL CAG XYZ IEP RBLX CVNA APP HOOD EME DASH WSM JBL BLDR KKR GDDY DELL PLTR '
         'BIIB ZIM MATX LUV TXN PSX HON CRM NFLX CMG HPQ MSI CL KMB '
         'TSEM CPRI ACI NXPI WTW QCOM BRK-B NUE STLD BA CAT DE RH DECK').split()
from .europa import EURO as EU_EURO, ES as EU_ES, IDX as EU_IDX, MACRO as EU_MACRO, SECTOR as EU_SECTOR, MISMA as EU_MISMA, is_eu        # una sola lista para el entrenamiento y las alertas
SIN_EU = bool(os.environ.get('VP_SIN_EUROPA'))                       # solo para comprobar que Europa no cambia nada de EE. UU.: entrena sin ningún valor europeo
if not SIN_EU:
    EXTRA += EU_EURO + EU_ES + EU_IDX + EU_MACRO
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


DESDE = 946684800                                                    # 1-ene-2000: los precios se piden desde aquí


def fetch(s):
    p = os.path.join(PX, s + '.json')
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 7 * 86400:
        try:
            if json.load(open(p)).get('desde', 10 ** 12) <= DESDE:          # ya descargado con el historial completo
                return 1
        except Exception:
            pass
    time.sleep(random.random() * 0.3)
    d = get(f'https://{random.choice(["query1", "query2"])}.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?period1={DESDE}&period2={int(time.time()) + 86400}&interval=1d&includeAdjustedClose=true')
    try:
        r = json.loads(d)['chart']['result'][0]
        q = r['indicators']['quote'][0]
        json.dump({'desde': DESDE, 't': r['timestamp'], 'o': q['open'], 'c': q['close'], 'v': q['volume'], 'a': r['indicators']['adjclose'][0]['adjclose']}, open(p, 'w'))
        return 1
    except Exception:
        return 0


uni = universe()


def join_dates():
    """Fecha en que cada miembro actual entró en el S&P 500 (columna «Date added» de Wikipedia). Vacío si falla: el ensayo se salta."""
    p = os.path.join(ARGS.cache, 'sp500_added.json')
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 30 * 86400:
        return json.load(open(p))
    try:
        t = pd.read_html(io.StringIO(get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').decode()))[0]
        t.columns = [str(c) for c in t.columns]
        dc = next(c for c in t.columns if 'Date added' in c or 'Date first added' in c)
        out = {}
        for _, r in t.iterrows():
            m = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(r[dc]))
            if m: out[str(r['Symbol']).strip().upper().replace('.', '-')] = m.group(0)
        json.dump(out, open(p, 'w'))
        return out
    except Exception as e:
        print('  fechas de entrada en el S&P 500: no disponibles', e)
        return {}


with ThreadPoolExecutor(max_workers=6) as ex:
    got = sum(ex.map(fetch, sorted(set(list(uni) + EXTRA))))
print(f'precios: {got} de {len(set(list(uni) + EXTRA))}', flush=True)


def limpia_eu(d):
    """Valores europeos: ni el cierre ajustado ni el sin ajustar de Yahoo son fiables siempre. El ajustado se rompe a veces (negativo en Inditex, Endesa, UniCredit,
    AB InBev, Acciona, Indra, Meliá…); el sin ajustar trae los dividendos extraordinarios como caídas (Endesa 2014, −48 %). Así que la serie se rehace día a día:
    la variación del ajustado si es coherente; la del sin ajustar si el ajustado es negativo o se dispara él solo. Después se quitan los errores de dato:
    los saltos con forma exacta de split no aplicado (×2, ÷2, ×3…: Deutsche Börse 2007), todo lo anterior al último salto imposible para una grande (×4 o ÷4 en un día:
    Bankinter 2002–2005) y los tramos entre un salto de un día de ±40 % y su vuelta al nivel previo (Bankinter 2007 y 2011). Devuelve cierre, apertura y lo que se ha tocado."""
    d = d[~((d.v.fillna(0) <= 0) & (d.c == d.c.shift(1)))]                  # sin negociación (antes de cotizar, festivos que Yahoo rellena con el cierre anterior): no es un día
    z = (d.c.diff() == 0).astype(int).values                              # precio congelado la mitad de un mes (Sabadell antes de cotizar en abril de 2001, tramos sin datos):
    fin = np.flatnonzero(pd.Series(z).rolling(20).sum().values >= 10)       # esas semanas no son precios de verdad y se anulan, con el día siguiente
    quieto = np.zeros(len(d) + 1, bool)
    for e in fin: quieto[max(0, e - 19):e + 2] = True
    quieto = quieto[:len(d)]
    a, c, o = d.a.values, d.c.values, d.o.values
    with np.errstate(all='ignore'):
        rc = c[1:] / c[:-1] - 1
        ra = np.where((a[1:] > 0) & (a[:-1] > 0), a[1:] / a[:-1] - 1, np.nan)
    r = np.where(np.isfinite(ra) & ~((np.abs(ra - rc) > 0.2) & (np.abs(ra) > np.abs(rc))), ra, rc)
    r = np.where(np.isfinite(r), r, rc)
    sp = 0
    for k in range(len(r)):                                               # split no aplicado: el salto es casi exactamente una razón de split
        x = 1 + r[k]
        if x > 0 and any(abs(x / q - 1) <= 0.03 for q in (1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2, 3, 4, 5, 10)):
            r[k] = 0.0; sp += 1
    lc = np.log(np.concatenate([[1.0], np.cumprod(1 + r)]))
    cl = np.exp(lc - lc[-1]) * c[-1]                                      # anclado al último cierre real
    q = o / c
    op = cl * np.where((q > 0.67) & (q < 1.5), q, 1.0)                     # la apertura guarda su distancia con el cierre del mismo día; si es absurda (sin ajustar a un split), vale el cierre
    rr = np.diff(lc, prepend=lc[0])
    bad = np.zeros(len(cl), bool)
    imp = np.flatnonzero(np.abs(rr) >= np.log(4))
    if len(imp): bad[:imp[-1]] = True
    k = 1
    while k < len(lc):
        if not bad[k] and abs(rr[k]) >= np.log(1.4):
            for m in range(k + 1, min(k + 60, len(lc))):
                if np.sign(rr[m]) == -np.sign(rr[k]) and abs(rr[m]) >= 0.8 * abs(rr[k]) and abs(lc[m] - lc[k - 1]) <= np.log(1.15):
                    bad[k:m] = True; k = m; break
        k += 1
    bad |= quieto
    cl, op = pd.Series(cl, index=d.index), pd.Series(op, index=d.index)
    cl[bad] = np.nan; op[bad] = np.nan
    return cl, op, int(bad.sum()) + sp
LIMPIEZA = {}


def load():
    A, O, DV = {}, {}, {}
    for f in os.listdir(PX):
        s = f[:-5]
        if SIN_EU and is_eu(s): continue
        j = json.load(open(os.path.join(PX, f)))
        t = pd.to_datetime(np.array(j['t'], dtype='int64'), unit='s').normalize()
        d = pd.DataFrame({'o': j['o'], 'c': j['c'], 'a': j['a'], 'v': j['v']}, index=t).astype('float64')
        d = d[~d.index.duplicated(keep='last')]; d = d[(d.c > 0)]
        if s in EU_MACRO:                                                # gas y euro: tal cual (sus saltos grandes pueden ser reales); fuera del ranking de volumen
            A[s] = d.c; O[s] = d.o; DV[s] = d.c * np.nan
            continue
        if is_eu(s):                                                     # Europa: cierre sin ajustar y sin errores de dato (ver limpia_eu); EE. UU. no cambia
            c_, o_, nb = limpia_eu(d)
            if nb: LIMPIEZA[s] = nb
            A[s] = c_; O[s] = o_; DV[s] = d.c * d.v
            continue
        A[s] = d.a; O[s] = d.o * d.a / d.c; DV[s] = d.c * d.v
    idx = A['SPY'].index
    A = pd.DataFrame(A).reindex(idx); O = pd.DataFrame(O).reindex(idx); DV = pd.DataFrame(DV).reindex(idx)
    return A, O, DV
A, O, DV = load(); idx = A.index; R = A.pct_change(fill_method=None)
if LIMPIEZA: print('Europa · sesiones anuladas por errores de dato:', LIMPIEZA, flush=True)
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
since = idx >= '2000-01-01'
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
DV.loc[:, [c_ for c_ in DV.columns if is_eu(c_)]] = np.nan                  # las europeas no compiten en el ranking de las grandes de EE. UU. (volumen en otra moneda): sus columnas quedan, vacías, para no descuadrar índices
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

# ── más variedad: industrias, China, elecciones, opas ──
G = 'Más variedad · industrias, China, elecciones y opas'
IND = 'SMH XBI KRE XHB JETS ITA GDX TAN XOP XRT IYT'.split()
for sg, lab, logic in ((-1, 'Una industria cae ≥ 4 % más que el mercado en un día → comprar esa industria', 'Castigo a una industria concreta (chips, biotecnología, bancos regionales, aerolíneas…).'), (1, 'Una industria sube ≥ 4 % más que el mercado en un día → comprar esa industria', 'Noticia buena para una industria concreta.')):
    rows = []
    for e in IND:
        if e not in col: continue
        rel = R[e] - R['SPY']; b_ = base([e])
        for i in thin(idx[since & ((rel <= -0.04) if sg < 0 else (rel >= 0.04))]):
            rows.append((rel.values[i], after(i, [e], 5) - b_[5], after(i, [e], 20) - b_[20], after(i, [e], 60) - b_[60]))
    add(G, lab, logic, summ(rows))
for e, nom in (('SMH', 'los semiconductores'), ('XBI', 'la biotecnología'), ('KRE', 'los bancos regionales'), ('JETS', 'las aerolíneas'), ('XOP', 'las petroleras independientes')):
    if e not in col: continue
    rel = R[e] - R['SPY']
    add(G, f'↳ {nom.capitalize()} caen ≥ 4 % más que el mercado → comprarlos', 'El mismo ensayo, industria por industria.', study(idx[since & (rel <= -0.04)], [e]))
rf = R['FXI']
add(G, 'La bolsa china sube ≥ 4 % en un día → casinos de Macao y cobre', 'China gasta más.', study(idx[since & (rf >= 0.04)], ['LVS', 'WYNN', 'FCX']))
add(G, 'La bolsa china cae ≥ 4 % en un día → casinos de Macao y cobre (comprar el castigo)', 'Apostar a que el castigo es excesivo.', study(idx[since & (rf <= -0.04)], ['LVS', 'WYNN', 'FCX']))
add(G, 'La bolsa china sube ≥ 4 % en un día → chinas cotizadas en EE. UU.', 'Seguir el impulso.', study(idx[since & (rf >= 0.04)], ['BABA', 'JD', 'BIDU']))
ELEC = ['2000-11-07', '2002-11-05', '2004-11-02', '2006-11-07', '2008-11-04', '2010-11-02', '2012-11-06', '2014-11-04', '2016-11-08', '2018-11-06', '2020-11-03', '2022-11-08', '2024-11-05']
em = np.isin(idx.strftime('%Y-%m-%d'), ELEC)
add(G, 'Día de elecciones en EE. UU. → comprar el Nasdaq-100 al día siguiente', 'Se despeja la incertidumbre.', absol2(em, 'QQQ', 'el Nasdaq-100'))
add(G, 'Día de elecciones en EE. UU. → comprar las pequeñas al día siguiente', 'Se despeja la incertidumbre.', absol2(em, 'IWM', 'el Russell 2000'))
opa = [e for e in u_ if e[3] >= 0.20]
add(G, 'Una grande sube ≥ 20 % en un día (opa o noticia extraordinaria) → sus competidoras', '¿Quién será la siguiente?', peer_study(opa, 'peers'))
add(G, 'Una grande sube ≥ 20 % en un día → seguir comprándola', 'Tras el salto.', peer_study(opa, 'self'))
vxr = R['^VIX'].values
add(G, 'El VIX se desploma ≥ 15 % en un día → Nasdaq-100', 'El miedo se va de golpe.', absol2(since & (vxr <= -0.15), 'QQQ', 'el Nasdaq-100'))
rq = (R['QQQ'] - R['SPY']).values
add(G, 'El Nasdaq-100 cae ≥ 1,5 % más que el S&P 500 en un día → Nasdaq-100', 'Rotación contra la tecnología.', absol2(since & (rq <= -0.015), 'QQQ', 'el Nasdaq-100'))

# ── los «no»: ¿hay condiciones en que sí? ──
G = 'Más variedad · sorpresas de la noche y rachas'
gap = (O['SPY'] / A['SPY'].shift(1) - 1).values
add(G, 'El S&P 500 abre ≥ 1,5 % abajo por una noticia de la noche → Nasdaq-100 al cierre de ese día', 'Susto nocturno: ¿se recupera en la sesión o sigue?', absol2(since & (gap <= -0.015), 'QQQ', 'el Nasdaq-100', at_close=True))
add(G, 'El S&P 500 abre ≥ 1,5 % arriba por una noticia de la noche → Nasdaq-100 al cierre de ese día', 'Alegría nocturna.', absol2(since & (gap >= 0.015), 'QQQ', 'el Nasdaq-100', at_close=True))
d3 = (SPYc / SPYc.shift(3) - 1).values; dn3 = (pd.Series(rs) < 0).rolling(3).sum().values == 3
add(G, 'El S&P 500 cae tres días seguidos y pierde ≥ 3 % en total → Nasdaq-100', 'Racha de ventas sin un día de pánico.', absol2(since & dn3 & (d3 <= -0.03), 'QQQ', 'el Nasdaq-100'))
rach = since & dn3 & (d3 <= -0.03)
add(G, '↳ …con el miedo bajo (VIX previo < 20)', 'La racha llega desde la calma.', absol2(rach & (vixp < 20), 'QQQ', 'el Nasdaq-100'))
add(G, '↳ …con el miedo medio (VIX previo entre 20 y 28)', 'La racha llega con nervios.', absol2(rach & (vixp >= 20) & (vixp < 28), 'QQQ', 'el Nasdaq-100'))
add(G, '↳ …con el miedo alto (VIX previo ≥ 28)', 'La racha llega en plena tormenta.', absol2(rach & (vixp >= 28), 'QQQ', 'el Nasdaq-100'))
add(G, '↳ …comprando al CIERRE del tercer día', 'Adelantarse a la apertura siguiente.', absol2(rach, 'QQQ', 'el Nasdaq-100', at_close=True))
dn5 = (pd.Series(rs) < 0).rolling(5).sum().values == 5
add(G, 'El S&P 500 cae cinco días seguidos → Nasdaq-100', 'Una semana entera en rojo.', absol2(since & dn5, 'QQQ', 'el Nasdaq-100'))
add(G, 'El bono a 10 años baja ≥ 15 pb en un día → Nasdaq-100', 'Tipos más bajos favorecen a las tecnológicas.', absol2(since & (ty.values <= -0.15), 'QQQ', 'el Nasdaq-100'))
add(G, 'El bono a 10 años sube ≥ 15 pb en un día → Nasdaq-100 (comprar el castigo)', 'Tipos más altos castigan a las tecnológicas.', absol2(since & (ty.values >= 0.15), 'QQQ', 'el Nasdaq-100'))
ru = R['UUP'].values
add(G, 'El dólar sube ≥ 1 % en un día → emergentes (comprar el castigo)', 'Dólar fuerte de golpe.', study(idx[since & (ru >= 0.01)], ['EEM']))
add(G, 'El dólar cae ≥ 1 % en un día → emergentes', 'Dólar débil de golpe.', study(idx[since & (ru <= -0.01)], ['EEM']))
add(G, 'El oro cae ≥ 3 % en un día → mineras de oro (comprar el castigo)', 'Caída fuerte del oro.', study(idx[since & (R['GC=F'] <= -0.03)], GOLDM))
add(G, 'El petróleo cae ≥ 4 % con el dólar subiendo ≥ 0,5 % → petroleras (comprar el castigo)', 'Caída del crudo por dólar fuerte, no por oferta.', study(idx[since & (ro <= -0.04) & (ru >= 0.005)], EP))

G = 'Más variedad · materias primas agrícolas'
def soft(sym, thr, basket, nombre, logica):
    if sym not in col: return
    r_ = R[sym].where(A[sym].shift(1) > 0).values
    m = since & ((r_ >= thr) if thr > 0 else (r_ <= thr))
    add(G, nombre, logica, study(idx[m], [b for b in basket if b in col]))
soft('KC=F', 0.06, ['SBUX'], 'El café sube ≥ 6 % en un día → Starbucks (comprar el castigo)', 'Su materia prima se encarece.')
soft('KC=F', -0.06, ['SBUX'], 'El café cae ≥ 6 % en un día → Starbucks', 'Su materia prima se abarata.')
soft('CC=F', 0.06, ['HSY', 'MDLZ'], 'El cacao sube ≥ 6 % en un día → Hershey y Mondelez (comprar el castigo)', 'El chocolate se encarece.')
soft('CC=F', -0.06, ['HSY', 'MDLZ'], 'El cacao cae ≥ 6 % en un día → Hershey y Mondelez', 'El chocolate se abarata.')
soft('SB=F', 0.05, ['KO', 'PEP', 'HSY'], 'El azúcar sube ≥ 5 % en un día → refrescos y chocolate (comprar el castigo)', 'Coste de una materia prima menor.')
soft('ZC=F', 0.04, ['MOS', 'CF', 'NTR'], 'El maíz sube ≥ 4 % en un día → fertilizantes', 'El agricultor gana más y abona más.')
soft('ZC=F', 0.04, ['TSN', 'HRL'], 'El maíz sube ≥ 4 % en un día → cárnicas (comprar el castigo)', 'El pienso se encarece.')
soft('ZC=F', 0.04, ['ADM', 'BG'], 'El maíz sube ≥ 4 % en un día → comercializadoras de grano', 'Más valor en lo que mueven.')
soft('LE=F', 0.03, ['TSN', 'HRL'], 'El ganado sube ≥ 3 % en un día → cárnicas (comprar el castigo)', 'La carne se encarece.')
soft('LE=F', 0.03, ['MCD', 'CMG'], 'El ganado sube ≥ 3 % en un día → cadenas de comida rápida (comprar el castigo)', 'La carne se encarece.')

G = 'Más variedad · entrar en el S&P 500'
JD = join_dates()
inc = []
for s_, d_ in JD.items():
    if s_ not in col: continue
    t_ = pd.Timestamp(d_)
    if t_ < pd.Timestamp('2009-06-01') or t_ > idx[-62]: continue
    i = int(idx.searchsorted(t_))
    if i < 30 or i >= len(idx) - 62: continue
    inc.append((i, s_, [], Rv[i, col[s_]], 0.0))
if inc:
    def inc_study(sel, lag, nota):
        rows = []
        for i, s_, _, r, _ in sel:
            j = i + lag
            if j + 61 < len(idx): rows.append((Av[j, col[s_]] / Av[j - 20, col[s_]] - 1 - (Av[j, col['SPY']] / Av[j - 20, col['SPY']] - 1), after(j, [s_], 5) - ubase(5), after(j, [s_], 20) - ubase(20), after(j, [s_], 60) - ubase(60)))
        r_ = summ(rows); r_['nota'] = nota; return r_
    add(G, 'Una empresa entra en el S&P 500 → comprarla el día que entra', 'Los fondos indexados tienen que comprarla.', inc_study(inc, 0, '«el mismo día» es lo que subió de más en las 20 sesiones anteriores a entrar'))
    add(G, 'Una empresa entra en el S&P 500 → comprarla 5 sesiones antes de entrar', 'Adelantarse a los fondos (el anuncio suele llegar una semana antes).', inc_study(inc, -5, '«el mismo día» es lo que subió de más en las 20 sesiones anteriores'))

G = 'Más variedad · líderes de otros sectores'
lider('KO', -0.04, ['PEP'], 'Coca-Cola cae ≥ 4 % en un día → PepsiCo', 'Rival directo.')
lider('PFE', -0.05, ['MRK', 'ABBV', 'BMY'], 'Pfizer cae ≥ 5 % en un día → otras farmacéuticas', '¿Problema propio o del sector?')
lider('MRK', -0.05, ['PFE', 'ABBV', 'BMY'], 'Merck cae ≥ 5 % en un día → otras farmacéuticas', '¿Problema propio o del sector?')
lider('LLY', -0.06, ['NVO'], 'Lilly cae ≥ 6 % en un día → Novo Nordisk', 'El otro jugador de la obesidad.')
lider('VZ', -0.04, ['T'], 'Verizon cae ≥ 4 % en un día → AT&T', 'Rival directo.')
lider('CMCSA', -0.05, ['CHTR'], 'Comcast cae ≥ 5 % en un día → Charter', 'Mismo negocio de cable.')
lider('FDX', -0.06, ['UPS'], 'FedEx cae ≥ 6 % en un día → UPS', 'Duopolio de paquetería.')
lider('FDX', 0.06, ['UPS'], 'FedEx sube ≥ 6 % en un día → UPS', 'Duopolio de paquetería.')
lider('UNP', -0.04, ['CSX', 'NSC'], 'Union Pacific cae ≥ 4 % en un día → otras ferroviarias', 'Mismo negocio.')
lider('XOM', 0.04, ['CVX', 'COP'], 'Exxon sube ≥ 4 % en un día → otras petroleras', 'Contagio dentro del petróleo.')
lider('TGT', -0.08, ['WMT', 'COST', 'DG'], 'Target cae ≥ 8 % en un día → otras cadenas', '¿Aviso sobre el consumidor o problema propio?')
lider('TGT', 0.08, ['KSS', 'M', 'DDS'], 'Target sube ≥ 8 % en un día → grandes almacenes', 'El consumidor va mejor de lo creído.')
lider('BBY', -0.08, ['AMZN', 'WMT'], 'Best Buy cae ≥ 8 % en un día → Amazon y Walmart', 'Quien se queda con sus clientes.')
lider('GS', -0.04, ['MS', 'JPM'], 'Goldman cae ≥ 4 % en un día → Morgan Stanley y JPMorgan', 'Banca de inversión.')
lider('AMD', 0.08, ['NVDA'], 'AMD sube ≥ 8 % en un día → Nvidia', 'El rival le quita negocio, o el sector va bien.')
lider('INTC', -0.07, ['AMD'], 'Intel cae ≥ 7 % en un día → AMD', 'Lo que pierde uno lo gana el otro.')

G = 'Revisar los «no» · el beneficiario evidente'
def study_if(days, basket, cond):
    b = base(basket); rows = []
    for i in thin(days):
        d = day0(i, basket)
        if cond(d): rows.append((d, after(i, basket, 5) - b[5], after(i, basket, 20) - b[20], after(i, basket, 60) - b[60]))
    return summ(rows) if rows else {'n': 0, 'd0': None, 'e5': None, 'e20': None, 'e60': None}
od = idx[since & (ro <= -0.04)]; ou = idx[since & (ro >= 0.04)]
add(G, 'Petróleo −4 % y las aerolíneas NO reaccionan ese día (< +0,3 %) → aerolíneas', 'El mercado aún no lo ha recogido.', study_if(od, AIR, lambda d: d < 0.003))
add(G, 'Petróleo −4 % y las aerolíneas ya suben ≥ 1,5 % ese día → aerolíneas', 'Ya está recogido.', study_if(od, AIR, lambda d: d >= 0.015))
add(G, 'Petróleo +4 % y las petroleras NO reaccionan ese día (< +0,5 %) → petroleras', 'El mercado aún no lo ha recogido.', study_if(ou, EP, lambda d: d < 0.005))
gd = idx[since & (R['GC=F'] >= 0.02)]
add(G, 'Oro +2 % y las mineras NO reaccionan ese día (< +1 %) → mineras de oro', 'El mercado aún no lo ha recogido.', study_if(gd, GOLDM, lambda d: d < 0.01))
add(G, 'Oro +2 % y las mineras ya suben ≥ 4 % ese día → mineras de oro', 'Ya está recogido.', study_if(gd, GOLDM, lambda d: d >= 0.04))
yu = idx[since & (ty >= 0.12)]
add(G, 'Bono +12 pb y los bancos NO reaccionan ese día (< +0,3 %) → bancos', 'El mercado aún no lo ha recogido.', study_if(yu, BANK, lambda d: d < 0.003))
add(G, 'Petróleo cae ≥ 10 % en un día → aerolíneas', 'El caso extremo.', study(idx[since & (ro <= -0.10)], AIR))
add(G, 'Petróleo cae ≥ 4 % con el mercado subiendo (S&P 500 > +0,5 %) → aerolíneas', 'Bajada del crudo por oferta, no por miedo.', study(idx[since & (ro <= -0.04) & (rs > 0.005)], AIR))
add(G, 'Petróleo cae ≥ 4 % con el mercado cayendo (S&P 500 < −1 %) → aerolíneas', 'Bajada del crudo por miedo a la economía.', study(idx[since & (ro <= -0.04) & (rs < -0.01)], AIR))

G = 'Revisar los «no» · el gigante caído'
def delayed(sel, cond):
    """Entra dos sesiones después, según lo que hizo la acción al día siguiente de la caída."""
    rows = []
    for i, s_, peers, r, pm in sel:
        j = col[s_]
        if i + 2 >= len(idx) or not np.isfinite(Rv[i + 1, j]): continue
        if cond(Rv[i + 1, j] - spy[i + 1]): rows.append((r - spy[i], after(i + 1, [s_], 5) - ubase(5), after(i + 1, [s_], 20) - ubase(20), after(i + 1, [s_], 60) - ubase(60)))
    return summ(rows)
chi = [e for e in c if off(e) > -0.10]
add(G, 'Cae ≥ 10 % desde máximos y al día siguiente REBOTA ≥ 3 % → comprar entonces', 'El mercado rectifica enseguida.', delayed(chi, lambda x: x >= 0.03))
add(G, 'Cae ≥ 10 % desde máximos y al día siguiente SIGUE cayendo ≥ 3 % → comprar entonces', 'Segunda oleada de ventas.', delayed(chi, lambda x: x <= -0.03))
add(G, 'Cae entre 10 % y 15 % desde máximos → comprar la caída', 'Tropiezo moderado del que iba primero.', peer_study([e for e in chi if e[3] > -0.15], 'self'))
add(G, 'Cae más de un 20 % desde máximos → comprar la caída', 'Desplome del que iba primero.', peer_study([e for e in chi if e[3] <= -0.20], 'self'))
add(G, 'Cae ≥ 10 % desde máximos y sus competidoras caen con ella (≥ 2 %) → comprar la caída', 'Problema que se lee como de todo el sector.', peer_study([e for e in chi if e[4] <= -0.02], 'self'))
add(G, 'Cae ≥ 10 % desde máximos y sus competidoras suben → comprar la caída', 'Problema solo suyo.', peer_study([e for e in chi if e[4] > 0.005], 'self'))
mhi = [e for e in mega['caida'] if off(e) > -0.10]
add(G, 'Una de las 50 mayores cae ≥ 7 % desde máximos → comprar la caída', 'Los gigantes de verdad.', peer_study(mhi, 'self'))
def wait60(sel):
    rows = []
    for i, s_, peers, r, pm in sel:
        if i + 61 < len(idx): rows.append((r - spy[i], np.nan, after(i + 60, [s_], 20) - ubase(20), after(i + 60, [s_], 60) - ubase(60)))
    return summ(rows)
add(G, 'Cae ≥ 10 % desde máximos → esperar 60 sesiones y comprar entonces', 'Dejar que pase un trimestre entero y unos resultados.', wait60(chi))
mom = (A.shift(1) / A.shift(127) - 1).values
add(G, 'Cae ≥ 10 % tras haber subido más de un 50 % en seis meses → comprar la caída', 'Pinchazo de una subida vertical.', peer_study([e for e in c if mom[e[0], col[e[1]]] > 0.5], 'self'))

G = 'Revisar los «no» · el líder y su sector'
add(G, 'Una grande sube ≥ 10 % y sus competidoras CAEN ese día (≥ 1 %) → sus competidoras', 'El mercado decide que el líder les quita negocio: ¿exagera?', peer_study([e for e in u_ if e[4] <= -0.01], 'peers'))
add(G, 'Una grande sube ≥ 10 % y sus competidoras suben con ella (≥ 2 %) → sus competidoras', 'Buena noticia para todo el sector, ya recogida.', peer_study([e for e in u_ if e[4] >= 0.02], 'peers'))
add(G, 'Una de las 50 mayores sube ≥ 7 % y sus competidoras quietas (< 1 %) → sus competidoras', 'Lo mismo, con los gigantes de verdad.', peer_study([e for e in mega['salto'] if abs(e[4]) < 0.01], 'peers'))
add(G, 'Una grande cae ≥ 10 % y sus competidoras suben ≥ 2 % ese día → sus competidoras', 'Trasvase claro de negocio.', peer_study([e for e in c if e[4] >= 0.02], 'peers'))

# ── combinaciones: cruzar dos condiciones y exigir que funcionen en las dos mitades del periodo ──
G = 'Combinaciones (validadas en dos mitades)'
CUT = int(idx.searchsorted(pd.Timestamp('2014-01-01')))
COMBOS = []
def combo(nombre, logica, rows):
    """rows: [(i, d0, e5, e20, e60)] ya en exceso. Se resume entero y por mitades (antes y después de 2014)."""
    rows = [r for r in rows if r is not None]
    if not rows: return
    allr = summ([r[1:] for r in rows]); a = summ([r[1:] for r in rows if r[0] < CUT]); b = summ([r[1:] for r in rows if r[0] >= CUT])
    allr['mitad1'] = {k: a.get(k) for k in ('n', 'e20', 'e60', 't60')}; allr['mitad2'] = {k: b.get(k) for k in ('n', 'e20', 'e60', 't60')}
    ok = all((m.get('e60') or -1) > 0 and (m.get('n') or 0) >= 5 for m in (allr['mitad1'], allr['mitad2'])) and (allr.get('t60') or 0) >= 2
    allr['robusta'] = bool(ok)
    allr['nota'] = f"2000–2013: {'n/d' if a.get('e60') is None else f'{a[chr(101)+chr(54)+chr(48)]:+.1f} %'} a 60 sesiones ({a['n']} casos) · 2014–2026: {'n/d' if b.get('e60') is None else f'{b[chr(101)+chr(54)+chr(48)]:+.1f} %'} ({b['n']} casos){' · funciona en las dos mitades' if ok else ''}"
    COMBOS.append(nombre); add(G, nombre, logica, allr)
_qb = {}
def qbase(sym, h, half):
    k = (sym, h, half)
    if k not in _qb:
        j = col[sym]; ii = np.arange(int(np.argmax(since)) + 1, len(idx) - 62, 3)
        ii = ii[ii < CUT] if half == 0 else ii[ii >= CUT]
        _qb[k] = float(np.nanmean([Av[i + h, j] / Ov[i + 1, j] - 1 for i in ii]))
    return _qb[k]
def idx_rows(mask, sym):
    out = []; j = col[sym]
    for i in thin(idx[mask]):
        if i + 61 >= len(idx) or not np.isfinite(Ov[i + 1, j]): continue
        hf = 0 if i < CUT else 1
        out.append((i, rs[i], *[Av[i + h, j] / Ov[i + 1, j] - 1 - qbase(sym, h, hf) for h in (5, 20, 60)]))
    return out
def stock_rows(sel, who):
    out = []
    for e in sel:
        i, s_, peers, r, pm = e
        if i + 61 >= len(idx): continue
        b_ = peers if who == 'peers' else [s_]
        out.append((i, pm if who == 'peers' else r - spy[i], *[after(i, b_, h) - ubase(h) for h in (5, 20, 60)]))
    return out
pan2 = since & (rs <= -0.02)
combo('Pánico con miedo alto (VIX ≥ 28) → Nasdaq-100', 'La regla de referencia, para comparar.', idx_rows(pan2 & (vixp >= 28), 'QQQ'))
combo('Pánico con miedo alto y mercado en tendencia alcista → Nasdaq-100', 'Susto fuerte dentro de una subida.', idx_rows(pan2 & (vixp >= 28) & above, 'QQQ'))
combo('Pánico con miedo alto y mercado en tendencia bajista → Nasdaq-100', 'Susto fuerte dentro de una bajada.', idx_rows(pan2 & (vixp >= 28) & ~above, 'QQQ'))
combo('Pánico con miedo extremo (VIX ≥ 35) → Nasdaq-100', 'Solo las tormentas de verdad.', idx_rows(pan2 & (vixp >= 35), 'QQQ'))
combo('Pánico con miedo alto → pequeñas (Russell 2000)', 'Lo más castigado en el pánico.', idx_rows(pan2 & (vixp >= 28), 'IWM'))
combo('Pánico con miedo alto → semiconductores', 'Lo que más cae, ¿más rebota?', idx_rows(pan2 & (vixp >= 28), 'SMH') if 'SMH' in col else [])
combo('Pánico con miedo alto y el bono a 10 años bajando ≥ 10 pb → Nasdaq-100', 'Huida a los bonos: los tipos bajan a favor del rebote.', idx_rows(pan2 & (vixp >= 28) & (ty.values <= -0.10), 'QQQ'))
combo('Racha de tres días (−3 %) con miedo alto y tendencia alcista → Nasdaq-100', 'Mala semana dentro de una subida.', idx_rows(since & dn3 & (d3 <= -0.03) & (vixp >= 28) & above, 'QQQ'))
combo('Hueco de apertura ≥ 1,5 % a la baja con miedo alto → Nasdaq-100 al cierre', 'Susto nocturno en plena tormenta.', [(i, d0, e5, e20, e60) for (i, d0, e5, e20, e60) in idx_rows(since & (gap <= -0.015) & (vixp >= 28), 'QQQ')])
vxs = lambda e: vixp[e[0]]
combo('Una grande cae ≥ 10 % con el miedo alto (VIX ≥ 28) → comprar la caída', 'La caída de una empresa en plena tormenta: ¿exceso de ventas?', stock_rows([e for e in c if vxs(e) >= 28], 'self'))
combo('Una grande cae ≥ 10 % con el miedo bajo (VIX < 18) → comprar la caída', 'La caída de una empresa en plena calma: problema propio.', stock_rows([e for e in c if vxs(e) < 18], 'self'))
combo('Una grande ya hundida (−30 %) cae otro 10 % con el miedo alto → comprar la caída', 'Capitulación.', stock_rows([e for e in c if off(e) < -0.30 and vxs(e) >= 28], 'self'))
combo('Cae ≥ 10 % desde máximos con el miedo alto → comprar la caída', 'El que iba primero, en plena tormenta.', stock_rows([e for e in c if off(e) > -0.10 and vxs(e) >= 28], 'self'))
combo('Tropiezo con competidoras subiendo ≥ 2 % y miedo bajo (VIX < 20) → sus competidoras', 'Trasvase de negocio sin ruido de mercado.', stock_rows([e for e in c if e[4] >= 0.02 and vxs(e) < 20], 'peers'))
combo('Tropiezo con competidoras subiendo ≥ 2 % y caída ≥ 20 % → sus competidoras', 'Trasvase tras un desplome.', stock_rows([e for e in c if e[4] >= 0.02 and e[3] <= -0.20], 'peers'))
combo('Tropiezo con competidoras subiendo ≥ 2 % desde máximos → sus competidoras', 'Tropieza el que iba primero y los demás lo celebran.', stock_rows([e for e in c if e[4] >= 0.02 and off(e) > -0.10], 'peers'))
peer_off = lambda e: np.nanmean([OFF[e[0], col[p]] for p in e[2] if p in col])
combo('El líder sube ≥ 10 % y castigan a competidoras ya hundidas (−25 %) → sus competidoras', 'Castigo sobre castigo.', stock_rows([e for e in u_ if e[4] <= -0.01 and peer_off(e) < -0.25], 'peers'))
combo('El líder sube ≥ 10 % y castigan a competidoras con el miedo alto → sus competidoras', 'Castigo en plena tormenta.', stock_rows([e for e in u_ if e[4] <= -0.01 and vxs(e) >= 25], 'peers'))
combo('El líder sube ≥ 15 % y castigan a sus competidoras → sus competidoras', 'Gran noticia del líder, castigo a los demás.', stock_rows([e for e in u_ if e[4] <= -0.01 and e[3] >= 0.15], 'peers'))
rzc = R['ZC=F'].values if 'ZC=F' in col else np.zeros(len(idx))
combo('Trigo +4 % y maíz +2 % el mismo día → fertilizantes', 'Todo el grano sube a la vez.', [(i, d0, *x) for i, d0, *x in [(i, day0(i, FERT), after(i, FERT, 5) - base(FERT)[5], after(i, FERT, 20) - base(FERT)[20], after(i, FERT, 60) - base(FERT)[60]) for i in thin(idx[since & (R['ZW=F'].values >= 0.04) & (rzc >= 0.02)]) if i + 61 < len(idx)]])
combo('Petróleo −4 % con el miedo alto → aerolíneas', 'Crudo barato y mercado asustado.', [(i, day0(i, AIR), after(i, AIR, 5) - base(AIR)[5], after(i, AIR, 20) - base(AIR)[20], after(i, AIR, 60) - base(AIR)[60]) for i in thin(idx[since & (ro.values <= -0.04) & (vixp >= 28)]) if i + 61 < len(idx)])
combo('Bono a 10 años −30 pb en diez sesiones con miedo alto → constructoras', 'Hipotecas más baratas en plena tormenta.', [(i, day0(i, HB), after(i, HB, 5) - base(HB)[5], after(i, HB, 20) - base(HB)[20], after(i, HB, 60) - base(HB)[60]) for i in thin(idx[since & (t10.values <= -0.30) & (vixp >= 28)]) if i + 61 < len(idx)])
print(f'combinaciones probadas: {len(COMBOS)} · por azar cabe esperar que alguna salga bien en las dos mitades', flush=True)

G = 'Casos con nombre'
# ── Europa: las mismas preguntas con valores europeos y españoles, medidos frente a la media de su propio grupo (no frente al SPY) ──
EUc = [s_ for s_ in EU_EURO + EU_ES if s_ in col and s_ not in EU_MISMA]   # grandes europeas con precios (incluye las del Ibex; ArcelorMittal una sola vez)
ESc = [s_ for s_ in EU_ES if s_ in col]
BANK_ES = [s_ for s_ in 'SAN.MC BBVA.MC CABK.MC SAB.MC BKT.MC UNI.MC'.split() if s_ in col]
def win(i, syms, h):
    """Rendimiento medio de la cesta comprando en la apertura siguiente al día i y vendiendo al cierre h sesiones después."""
    if i + 1 + h >= len(idx): return np.nan
    c_ = [col[s_] for s_ in syms if s_ in col]
    with np.errstate(all='ignore'):
        v = Av[i + h, c_] / Ov[i + 1, c_] - 1
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
def after_eu(i, syms, h, U=None):
    """Como after(), pero el exceso es sobre la media a partes iguales de las grandes europeas (U), que cotizan en el mismo horario."""
    return win(i, syms, h) - win(i, U or EUc, h)
def day0_eu(i, syms, U=None):
    with np.errstate(all='ignore'):
        a_ = R.values[i, [col[s_] for s_ in syms if s_ in col]]; b_ = R.values[i, [col[s_] for s_ in (U or EUc)]]
    return float(np.nanmean(a_) - np.nanmean(b_)) if np.isfinite(a_).any() and np.isfinite(b_).any() else np.nan
_eb = {}
def base_eu(basket, U=None):
    k = (tuple(basket), tuple(U or EUc))
    if k not in _eb:
        ii = np.arange(260 + int(np.argmax(since)), len(idx) - 62, 7)
        with np.errstate(all='ignore'):
            _eb[k] = {h: float(np.nanmean([after_eu(i, basket, h, U) for i in ii])) for h in (5, 20, 60)}
    return _eb[k]
def eu_study(days, basket, U=None):
    b_ = base_eu(basket, U)
    return summ([(day0_eu(i, basket, U), after_eu(i, basket, 5, U) - b_[5], after_eu(i, basket, 20, U) - b_[20], after_eu(i, basket, 60, U) - b_[60]) for i in thin(days) if i + 61 < len(idx)])
def eu_idx_rows(mask, sym, d0s=None):
    """Índice europeo comprado en la apertura siguiente: rendimiento absoluto menos lo que hace en un tramo cualquiera de su misma mitad del periodo."""
    out = []; j = col[sym]
    for i in thin(idx[mask]):
        if i + 61 >= len(idx) or not np.isfinite(Ov[i + 1, j]) or not np.isfinite(Av[i + 60, j]): continue
        hf = 0 if i < CUT else 1
        out.append((i, R.values[i, col[d0s or sym]], *[Av[i + h, j] / Ov[i + 1, j] - 1 - qbase(sym, h, hf) for h in (5, 20, 60)]))
    return out
if EUc and '^STOXX50E' in col and '^IBEX' in col:
    print(f'Europa: {len(EUc)} grandes europeas con precios, {len(ESc)} del Ibex', flush=True)
    rE, rI = R['^STOXX50E'].values, R['^IBEX'].values
    with np.errstate(all='ignore'):
        ewE = np.nanmean(R.values[:, [col[s_] for s_ in EUc]], axis=1); ewS = np.nanmean(R.values[:, [col[s_] for s_ in ESc]], axis=1)
    G = 'Europa · pánico y miedo (validadas en dos mitades)'
    combo('Euro Stoxx 50 −2 % con el miedo alto (VIX previo ≥ 28) → Euro Stoxx 50', 'El pánico con miedo alto, en Europa. Datos del índice desde 2007.', eu_idx_rows(since & (rE <= -0.02) & (vixp >= 28), '^STOXX50E'))
    combo('Euro Stoxx 50 −2 % viniendo de calma (VIX < 20) → Euro Stoxx 50', 'El mismo golpe con el mercado tranquilo.', eu_idx_rows(since & (rE <= -0.02) & (vixp < 20), '^STOXX50E'))
    combo('Ibex 35 −2 % con el miedo alto (VIX previo ≥ 28) → Ibex 35', 'El pánico con miedo alto, en España.', eu_idx_rows(since & (rI <= -0.02) & (vixp >= 28), '^IBEX'))
    combo('Ibex 35 −2 % viniendo de calma (VIX < 20) → Ibex 35', 'El mismo golpe con el mercado tranquilo.', eu_idx_rows(since & (rI <= -0.02) & (vixp < 20), '^IBEX'))
    combo('Wall Street −2 % con el miedo alto → Euro Stoxx 50 en la apertura siguiente', 'Europa abre después del golpe de Nueva York: ¿llega tarde o a tiempo?', eu_idx_rows(pan2 & (vixp >= 28), '^STOXX50E', d0s='SPY'))
    combo('Wall Street −2 % con el miedo alto → Ibex 35 en la apertura siguiente', 'Lo mismo, con la bolsa española.', eu_idx_rows(pan2 & (vixp >= 28), '^IBEX', d0s='SPY'))
    if BANK_ES:
        bb = base_eu(BANK_ES, ESc)
        combo('Ibex 35 −2 % con el miedo alto → bancos españoles (frente al resto del Ibex)', 'Lo que más cae en el pánico, ¿más rebota?', [(i, day0_eu(i, BANK_ES, ESc), *[after_eu(i, BANK_ES, h, ESc) - bb[h] for h in (5, 20, 60)]) for i in thin(idx[since & (rI <= -0.02) & (vixp >= 28)]) if i + 61 < len(idx)])
    G = 'Europa · grandes valores (frente a la media de su grupo)'
    eu_ev = {'caida': [], 'salto': []}
    for s_ in EUc:
        j = col[s_]; U = ESc if s_ in ESc else EUc; mk = ewS if s_ in ESc else ewE
        for i in np.flatnonzero(since & (np.abs(R.values[:, j]) >= 0.10)):
            r0, r1 = R.values[i, j], R.values[i + 1, j] if i + 1 < len(idx) else np.nan
            if i < 61 or i + 61 >= len(idx) or not abs(mk[i]) <= 0.02: continue          # día del propio valor, no de todo el mercado
            if np.isfinite(r1) and np.sign(r1) != np.sign(r0) and abs(r1) >= 0.8 * abs(r0): continue      # se deshace al día siguiente: casi siempre un error de dato de Yahoo
            eu_ev['caida' if r0 < 0 else 'salto'].append((i, s_, U, r0, mk[i]))
    def eu_rows(sel):
        return [(i, r - mk, *[after_eu(i, [s_], h, U) for h in (5, 20, 60)]) for i, s_, U, r, mk in sel]
    nt = 'exceso sobre la media a partes iguales de las grandes de su grupo (europeas, o del Ibex para las españolas)'
    def eu_add(nombre, logica, rows):
        r_ = summ([x[1:] for x in rows]); r_['nota'] = nt; add(G, nombre, logica, r_)
    ce, se = eu_ev['caida'], eu_ev['salto']
    eu_add('Una grande europea cae ≥ 10 % en un día → comprar la propia caída', 'El gigante caído, en Europa.', eu_rows(ce))
    eu_add('Una grande española cae ≥ 10 % en un día → comprar la propia caída', 'Lo mismo, solo el Ibex.', eu_rows([e for e in ce if e[1] in ESc]))
    eu_add('Una grande europea sube ≥ 10 % en un día → seguir comprándola', 'La buena noticia tarda en digerirse.', eu_rows(se))
    vxe = lambda e: vixp[e[0]]
    G = 'Europa · competidoras y contagio (sectores puestos a mano)'
    peers_of = lambda s_: [p_ for p_ in EUc if p_ != s_ and EU_SECTOR.get(p_) == EU_SECTOR.get(s_) and EU_MISMA.get(p_) != s_ and EU_MISMA.get(s_) != p_]
    pe = []                                                              # (i, valor, competidoras, movimiento, movimiento medio de las competidoras sobre la media europea)
    for e in ce + se:
        i, s_ = e[0], e[1]; pp = peers_of(s_)
        if len(pp) < 2: continue
        with np.errstate(all='ignore'):
            pmv = np.nanmean(R.values[i, [col[p_] for p_ in pp]]) - ewE[i]
        if np.isfinite(pmv): pe.append((i, s_, pp, e[3], pmv))
    def peer_rows(sel):
        return [(i, pmv, *[after_eu(i, pp, h) for h in (5, 20, 60)]) for i, s_, pp, r, pmv in sel]
    def pe_add(nombre, logica, rows, nota='exceso de las competidoras sobre la media de las grandes europeas; «el mismo día» es lo que hicieron ellas'):
        r_ = summ([x[1:] for x in rows]); r_['nota'] = nota; add(G, nombre, logica, r_)
    pe_add('Una grande europea cae ≥ 10 % → sus competidoras', 'Lo que pierde una lo ganan las otras.', peer_rows([e for e in pe if e[3] < 0]))
    pe_add('Una grande europea cae ≥ 10 % y sus competidoras suben ≥ 2 % ese día → sus competidoras', 'Trasvase de negocio (lección «rival»).', peer_rows([e for e in pe if e[3] < 0 and e[4] >= 0.02]))
    pe_add('Una grande europea cae ≥ 10 % y sus competidoras caen con ella (≥ 1 %) → sus competidoras', 'Contagio del sector: ¿rebotan?', peer_rows([e for e in pe if e[3] < 0 and e[4] <= -0.01]))
    pe_add('Una grande europea sube ≥ 10 % → sus competidoras', 'La buena noticia de una, ¿vale para el sector?', peer_rows([e for e in pe if e[3] > 0]))
    pe_add('Una grande europea sube ≥ 10 % y sus competidoras CAEN ese día (≥ 1 %) → sus competidoras', 'El mercado decide que el líder les quita negocio (lección «contagio»).', peer_rows([e for e in pe if e[3] > 0 and e[4] <= -0.01]))
    secs = {}
    for s_ in EUc: secs.setdefault(EU_SECTOR.get(s_), []).append(s_)
    rows = []
    for sec, vv in secs.items():
        if len(vv) < 3: continue
        with np.errstate(all='ignore'):
            rel = np.nanmean(R.values[:, [col[v_] for v_ in vv]], axis=1) - ewE
        for i in thin(idx[since & (rel <= -0.03)]):
            if i + 61 < len(idx): rows.append((i, rel[i], *[after_eu(i, vv, h) for h in (5, 20, 60)]))
    pe_add('Un sector europeo cae ≥ 3 % más que el resto en un día → comprar ese sector', 'El castigo a un sector entero suele ser excesivo.', rows, 'exceso del sector sobre la media de las grandes europeas')
    rows = []
    for sec, vv in secs.items():
        if len(vv) < 3: continue
        with np.errstate(all='ignore'):
            rel = np.nanmean(R.values[:, [col[v_] for v_ in vv]], axis=1) - ewE
        for i in thin(idx[since & (rel >= 0.03)]):
            if i + 61 < len(idx): rows.append((i, rel[i], *[after_eu(i, vv, h) for h in (5, 20, 60)]))
    pe_add('Un sector europeo sube ≥ 3 % más que el resto en un día → seguir comprando ese sector', 'La buena noticia de un sector tarda en digerirse.', rows, 'exceso del sector sobre la media de las grandes europeas')
    G = 'Europa · grandes valores (validadas en dos mitades)'
    combo('Una grande europea cae ≥ 10 % con el miedo alto (VIX ≥ 28) → comprar la caída', 'La combinación más fuerte de EE. UU., en Europa.', eu_rows([e for e in ce if vxe(e) >= 28]))
    combo('Una grande europea cae ≥ 10 % con el miedo bajo (VIX < 18) → comprar la caída', 'En calma: problema propio.', eu_rows([e for e in ce if vxe(e) < 18]))
    offe = lambda e: OFF[e[0], col[e[1]]]
    combo('Una grande europea ya hundida (−30 %) cae otro 10 % → comprar la caída', 'Capitulación.', eu_rows([e for e in ce if offe(e) < -0.30]))
    combo('Una grande europea cae ≥ 10 % estando cerca de máximos → comprar la caída', 'Estaba cotizada para la perfección (lección «caido»).', eu_rows([e for e in ce if offe(e) > -0.10]))
    combo('Una grande europea sube ≥ 10 % y marca máximo del año → seguir comprándola', 'Ruptura con noticia.', eu_rows([e for e in se if offe(e) > -0.02]))
    dE3 = (pd.Series(rE) < 0).rolling(3).sum().values == 3; sE3 = (A['^STOXX50E'] / A['^STOXX50E'].shift(3) - 1).values
    dI3 = (pd.Series(rI) < 0).rolling(3).sum().values == 3; sI3 = (A['^IBEX'] / A['^IBEX'].shift(3) - 1).values
    G = 'Europa · rachas con miedo (validadas en dos mitades)'
    combo('Racha de tres días (−3 %) con el miedo alto → Euro Stoxx 50', 'Mala semana en plena tormenta.', eu_idx_rows(since & dE3 & (sE3 <= -0.03) & (vixp >= 28), '^STOXX50E'))
    combo('Racha de tres días (−3 %) con el miedo alto → Ibex 35', 'Lo mismo, con la bolsa española.', eu_idx_rows(since & dI3 & (sI3 <= -0.03) & (vixp >= 28), '^IBEX'))
    G = 'Europa · contagio desde Wall Street (comprando en la apertura europea siguiente)'
    SEMI_EU = [x for x in 'ASML.AS IFX.DE'.split() if x in col]; BANK_EU = [x for x in 'BNP.PA SAN.MC BBVA.MC ISP.MI UCG.MI INGA.AS DBK.DE'.split() if x in col]
    rsm = R['SMH'].values - R['SPY'].values if 'SMH' in col else None; rxf = R['XLF'].values - R['SPY'].values if 'XLF' in col else None
    if rsm is not None and SEMI_EU:
        add(G, 'Los chips de EE. UU. caen ≥ 4 % más que el S&P 500 → ASML e Infineon', '¿Se contagia el castigo al día siguiente en Europa?', {**eu_study(idx[since & (rsm <= -0.04)], SEMI_EU), 'nota': nt})
        add(G, 'Los chips de EE. UU. suben ≥ 4 % más que el S&P 500 → ASML e Infineon', '¿Se contagia la buena noticia?', {**eu_study(idx[since & (rsm >= 0.04)], SEMI_EU), 'nota': nt})
    if rxf is not None and BANK_EU:
        add(G, 'Los bancos de EE. UU. caen ≥ 3 % más que el S&P 500 → bancos europeos', '¿Se contagia el miedo a la banca?', {**eu_study(idx[since & (rxf <= -0.03)], BANK_EU), 'nota': nt})
        add(G, 'Los bancos de EE. UU. suben ≥ 3 % más que el S&P 500 → bancos europeos', '¿Se contagia la buena noticia?', {**eu_study(idx[since & (rxf >= 0.03)], BANK_EU), 'nota': nt})
    G = 'Europa · materias primas'
    ro_ = ro.values
    AIR_EU = [x for x in 'IAG.MC LHA.DE AF.PA'.split() if x in col]; OIL_EU = [x for x in 'TTE.PA ENI.MI REP.MC'.split() if x in col]
    UTIL_EU = [x for x in 'ENEL.MI ENGI.PA RWE.DE EOAN.DE IBE.MC ELE.MC NTGY.MC'.split() if x in col]; CHEM_EU = [x for x in 'BAS.DE'.split() if x in col]
    EXPO_EU = [x for x in 'MBG.DE BMW.DE VOW3.DE MC.PA RMS.PA KER.PA ASML.AS SAP.DE AIR.PA RACE.MI'.split() if x in col]
    if AIR_EU:
        add(G, 'Petróleo cae ≥ 4 % en un día → aerolíneas europeas (IAG, Lufthansa, Air France-KLM)', 'El combustible es su mayor coste variable.', {**eu_study(idx[since & (ro_ <= -0.04)], AIR_EU), 'nota': nt})
        add(G, 'Petróleo sube ≥ 4 % en un día → aerolíneas europeas (comprar el castigo)', 'Apostar a que el castigo es excesivo.', {**eu_study(idx[since & (ro_ >= 0.04)], AIR_EU), 'nota': nt})
    if OIL_EU:
        add(G, 'Petróleo sube ≥ 4 % en un día → petroleras europeas (Total, Eni, Repsol)', 'Cada dólar de más va a su margen.', {**eu_study(idx[since & (ro_ >= 0.04)], OIL_EU), 'nota': nt})
        add(G, 'Petróleo cae ≥ 4 % en un día → petroleras europeas (comprar la caída)', 'Apostar a que el castigo es excesivo.', {**eu_study(idx[since & (ro_ <= -0.04)], OIL_EU), 'nota': nt})
    if 'TTF=F' in col:
        rg = R['TTF=F'].values
        if CHEM_EU:
            add(G, 'Gas europeo sube ≥ 10 % en un día → BASF (comprar el castigo)', 'El gas es su materia prima y su energía. Datos del gas desde 2017.', {**eu_study(idx[since & (rg >= 0.10)], CHEM_EU), 'nota': nt})
            add(G, 'Gas europeo cae ≥ 10 % en un día → BASF', 'Alivio de costes. Datos del gas desde 2017.', {**eu_study(idx[since & (rg <= -0.10)], CHEM_EU), 'nota': nt})
        if UTIL_EU:
            add(G, 'Gas europeo sube ≥ 10 % en un día → eléctricas europeas', 'La luz se paga al precio del gas; quien produce con renovables gana más. Datos desde 2017.', {**eu_study(idx[since & (rg >= 0.10)], UTIL_EU), 'nota': nt})
    if 'EURUSD=X' in col and EXPO_EU:
        rx = R['EURUSD=X'].values
        add(G, 'El euro sube ≥ 1 % frente al dólar en un día → exportadoras europeas (comprar el castigo)', 'Un euro caro resta a lo que venden fuera. Datos del euro desde 2003.', {**eu_study(idx[since & (rx >= 0.01)], EXPO_EU), 'nota': nt})
        add(G, 'El euro cae ≥ 1 % frente al dólar en un día → exportadoras europeas', 'Un euro barato suma a lo que venden fuera.', {**eu_study(idx[since & (rx <= -0.01)], EXPO_EU), 'nota': nt})
    if 'IAG.MC' in col:
        add(G, 'Petróleo cae ≥ 4 % en un día → IAG (Iberia, British Airways)', 'El combustible es su mayor coste variable.', {**eu_study(idx[since & (ro_ <= -0.04)], ['IAG.MC']), 'nota': nt})
    if 'REP.MC' in col:
        add(G, 'Petróleo sube ≥ 4 % en un día → Repsol', 'Cada dólar de más va a su margen.', {**eu_study(idx[since & (ro_ >= 0.04)], ['REP.MC'], ESc), 'nota': nt})
        add(G, 'Petróleo cae ≥ 4 % en un día → Repsol (comprar la caída)', 'Apostar a que el castigo es excesivo.', {**eu_study(idx[since & (ro_ <= -0.04)], ['REP.MC'], ESc), 'nota': nt})
    if BANK_ES:
        add(G, 'El bono de EE. UU. sube ≥ 12 pb en un día → bancos españoles', 'Más tipos, más margen (¿se contagia a Europa?).', {**eu_study(idx[since & (ty.values >= 0.12)], BANK_ES, ESc), 'nota': nt})
else:
    print('Europa: sin precios suficientes, se omite', flush=True)

TIPOS = {'J': 'Cambio de consejero delegado → comprar la empresa', 'X': 'Informe de un bajista → comprar el castigo', 'P': 'Anuncio de división de acciones (split) → comprar', 'I': 'Anuncio de entrada en el S&P 500 → comprar', 'F': 'Aprobación de un fármaco clave → comprar al laboratorio', 'H': 'Huelga, bloqueo o escasez → el beneficiado o el castigado', 'W': 'Entra un inversor activista → comprar la empresa', 'E': 'Hecho consumado que cambia el mundo (guerra, paz, vacuna, elecciones, fármaco)', 'D': 'Decisión anunciada (banco central, ley firmada, plan de empresa)', 'S': 'Susto o anuncio de un día', 'N': 'Sector castigado por una norma o sanción → comprar el castigo', 'V': 'Víctimas de la novedad de otro → comprar el castigo', 'A': 'Ataque o avería que obliga a gastar en protección', 'M': 'Gran compra de empresa → las siguientes candidatas', 'K': 'El canario avisa: comprar a las arrastradas por el aviso de otra', 'L': 'Demanda antimonopolio contra un gigante → comprar el gigante', 'C': 'Comprar al gigante caído', 'R': 'Competencia del que tropieza', 'B': 'Buena noticia que se contagia',
         'O': 'Fusión u OPA que se rompe → comprar a la que iba a ser comprada', 'T': 'Aranceles anunciados → comprar a los castigados', 'U': 'Multa o norma de Bruselas contra un gigante → comprar el gigante', 'Y': 'Gran recompra de acciones anunciada → comprar'}
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
 # ── 2000-2007 ──
 ('2000-04-14', 'Desplome del Nasdaq (abril de 2000) → comprar el Nasdaq', 'QQQ', 'S'),
 ('2001-09-17', 'Reapertura tras el 11-S → comprar el Nasdaq', 'QQQ', 'S'),
 ('2001-09-17', 'Reapertura tras el 11-S → comprar aerolíneas castigadas', 'LUV', 'S'),
 ('2001-09-17', 'Reapertura tras el 11-S → defensa', 'LMT NOC GD', 'E'),
 ('2001-09-17', 'Reapertura tras el 11-S → seguridad y vigilancia', 'LHX', 'A'),
 ('2003-03-20', 'Empieza la guerra de Irak → defensa', 'LMT NOC GD', 'E'),
 ('2003-03-20', 'Empieza la guerra de Irak → comprar el Nasdaq', 'QQQ', 'E'),
 ('2003-04-02', 'Alerta de la OMS por el SARS → comprar aerolíneas castigadas', 'LUV', 'S'),
 ('2004-03-11', 'Atentados de Madrid → defensa', 'LMT NOC GD', 'S'),
 ('2005-07-07', 'Atentados de Londres → defensa', 'LMT NOC GD', 'S'),
 ('2005-08-29', 'Huracán Katrina → refino', 'VLO', 'S'),
 ('2005-08-29', 'Huracán Katrina → reformas del hogar', 'HD LOW', 'S'),
 ('2007-02-27', 'Desplome de la bolsa de Shanghái → comprar el Nasdaq', 'QQQ', 'S'),
 ('2007-08-09', 'BNP congela fondos: estalla la crisis hipotecaria → comprar bancos', 'JPM BAC WFC', 'S'),
 ('2008-03-17', 'Cae Bear Stearns → comprar bancos', 'JPM BAC WFC', 'S'),
 ('2008-09-15', 'Quiebra Lehman Brothers → comprar bancos', 'JPM BAC WFC', 'S'),
 ('2008-09-29', 'El Congreso rechaza el primer rescate → comprar el Nasdaq', 'QQQ', 'S'),
 ('2004-11-03', 'Bush reelegido (2004) → farmacéuticas y defensa', 'PFE MRK LMT', 'E'),
 ('2006-11-08', 'Los demócratas ganan el Congreso (2006) → farmacéuticas (comprar el castigo)', 'PFE MRK', 'E'),
 ('2008-11-05', 'Obama gana (2008) → solar', 'FSLR', 'E'),
 ('2001-01-03', 'La Fed recorta los tipos por sorpresa (2001) → Nasdaq', 'QQQ', 'D'),
 ('2003-05-28', 'Se firma la rebaja de impuestos sobre dividendos → eléctricas', 'XLU', 'D'),
 ('2004-06-30', 'La Fed empieza a subir tipos (2004) → bancos', 'JPM BAC WFC', 'D'),
 ('2007-09-18', 'La Fed recorta 50 pb (2007) → constructoras', 'DHI LEN PHM', 'D'),
 ('2008-01-22', 'Recorte de emergencia de 75 pb → bancos', 'JPM BAC WFC', 'D'),
 ('2008-09-08', 'Intervención de Fannie Mae y Freddie Mac → bancos', 'JPM BAC WFC', 'D'),
 ('2008-10-08', 'Recorte de tipos coordinado mundial → Nasdaq', 'QQQ', 'D'),
 ('2000-09-22', 'Intel avisa de menos ventas (2000) → comprar Intel', 'INTC', 'C'),
 ('2000-09-29', 'Apple se hunde un 52 % (2000) → comprar Apple', 'AAPL', 'C'),
 ('2004-09-30', 'Merck retira el Vioxx → comprar Merck', 'MRK', 'C'),
 ('2000-09-22', 'Intel avisa → AMD', 'AMD', 'R'),
 ('2004-09-30', 'Merck retira el Vioxx → Pfizer', 'PFE', 'R'),
 ('2000-04-03', 'Un juez falla contra Microsoft por monopolio → comprar Microsoft', 'MSFT', 'L'),
 ('2000-01-10', 'AOL compra Time Warner → otros medios', 'DIS', 'M'),
 ('2005-01-28', 'Procter & Gamble compra Gillette → otras de consumo', 'CL KMB', 'M'),
 ('2006-07-24', 'AMD compra ATI → Nvidia', 'NVDA', 'M'),
 ('2007-01-09', 'Apple presenta el iPhone → comprar a la castigada Motorola', 'MSI', 'V'),
 ('2007-02-08', 'HSBC avisa de pérdidas hipotecarias → comprar otros bancos', 'JPM BAC WFC', 'K'),
 ('2000-11-27', 'General Electric nombra a Jeff Immelt', 'GE', 'J'),
 ('2005-02-09', 'Hewlett-Packard echa a Carly Fiorina', 'HPQ', 'J'),
 ('2005-03-14', 'Disney nombra a Bob Iger', 'DIS', 'J'),
 ('2003-01-17', 'Microsoft anuncia un split de 2 por 1', 'MSFT', 'P'),
 ('2005-02-14', 'Apple anuncia un split de 2 por 1', 'AAPL', 'P'),
 ('2005-11-21', 'Amazon entrará en el S&P 500', 'AMZN', 'I'),
 ('2006-03-27', 'Google entrará en el S&P 500', 'GOOGL', 'I'),
 ('2003-01-27', 'Gusano informático Slammer → ciberseguridad', 'CHKP', 'A'),
 ('2002-10-01', 'Cierre de los puertos de la costa oeste → ferroviarias', 'UNP CSX', 'H'),
 # ── tipos nuevos (3) ──
 ('2014-02-04', 'Microsoft nombra consejero delegado a Satya Nadella', 'MSFT', 'J'),
 ('2016-10-13', 'Dimite el jefe de Wells Fargo tras el escándalo de las cuentas', 'WFC', 'J'),
 ('2018-10-01', 'General Electric echa a su jefe y ficha a Larry Culp', 'GE', 'J'),
 ('2019-12-23', 'Boeing despide a su consejero delegado', 'BA', 'J'),
 ('2020-02-26', 'Iger deja Disney y llega Chapek', 'DIS', 'J'),
 ('2021-02-03', 'Bezos anuncia que deja de dirigir Amazon', 'AMZN', 'J'),
 ('2022-11-21', 'Disney recupera a Bob Iger', 'DIS', 'J'),
 ('2024-03-25', 'El jefe de Boeing anuncia su salida', 'BA', 'J'),
 ('2024-08-01', 'Boeing nombra a Kelly Ortberg', 'BA', 'J'),
 ('2024-08-13', 'Starbucks ficha a Brian Niccol', 'SBUX', 'J'),
 ('2024-09-20', 'Nike recupera a Elliott Hill', 'NKE', 'J'),
 ('2024-12-02', 'Intel echa a Pat Gelsinger', 'INTC', 'J'),
 ('2023-03-23', 'Hindenburg publica contra Block', 'XYZ', 'X'),
 ('2023-05-02', 'Hindenburg publica contra Icahn Enterprises', 'IEP', 'X'),
 ('2024-08-27', 'Hindenburg publica contra Super Micro', 'SMCI', 'X'),
 ('2024-10-08', 'Hindenburg publica contra Roblox', 'RBLX', 'X'),
 ('2025-01-02', 'Hindenburg publica contra Carvana', 'CVNA', 'X'),
 ('2020-07-31', 'Apple anuncia un split de 4 por 1', 'AAPL', 'P'),
 ('2020-08-12', 'Tesla anuncia un split de 5 por 1', 'TSLA', 'P'),
 ('2021-05-24', 'Nvidia anuncia un split de 4 por 1', 'NVDA', 'P'),
 ('2022-02-02', 'Alphabet anuncia un split de 20 por 1', 'GOOGL', 'P'),
 ('2022-03-10', 'Amazon anuncia un split de 20 por 1', 'AMZN', 'P'),
 ('2022-03-28', 'Tesla anuncia otro split', 'TSLA', 'P'),
 ('2024-01-30', 'Walmart anuncia un split de 3 por 1', 'WMT', 'P'),
 ('2024-03-20', 'Chipotle anuncia un split de 50 por 1', 'CMG', 'P'),
 ('2024-05-23', 'Nvidia anuncia un split de 10 por 1', 'NVDA', 'P'),
 ('2024-06-13', 'Broadcom anuncia un split de 10 por 1', 'AVGO', 'P'),
 ('2020-11-17', 'Tesla entrará en el S&P 500', 'TSLA', 'I'),
 ('2023-12-04', 'Uber, Jabil y Builders FirstSource entrarán en el S&P 500', 'UBER JBL BLDR', 'I'),
 ('2024-03-04', 'Super Micro entrará en el S&P 500', 'SMCI', 'I'),
 ('2024-06-10', 'CrowdStrike, KKR y GoDaddy entrarán en el S&P 500', 'CRWD KKR GDDY', 'I'),
 ('2024-09-09', 'Palantir y Dell entrarán en el S&P 500', 'PLTR DELL', 'I'),
 ('2025-03-10', 'DoorDash y Williams-Sonoma entrarán en el S&P 500', 'DASH WSM', 'I'),
 ('2025-05-13', 'Coinbase entrará en el S&P 500', 'COIN', 'I'),
 ('2025-09-08', 'AppLovin, Robinhood y Emcor entrarán en el S&P 500', 'APP HOOD EME', 'I'),
 ('2021-06-07', 'La FDA aprueba Aduhelm contra el alzhéimer', 'BIIB', 'F'),
 ('2021-06-07', 'La FDA aprueba Wegovy contra la obesidad', 'NVO', 'F'),
 ('2023-11-09', 'La FDA aprueba Zepbound contra la obesidad', 'LLY', 'F'),
 ('2024-03-11', 'La FDA aprueba Wegovy para el corazón', 'NVO', 'F'),
 ('2024-07-03', 'La FDA aprueba Kisunla contra el alzhéimer', 'LLY', 'F'),
 ('2023-01-09', 'La FDA aprueba Leqembi contra el alzhéimer', 'BIIB', 'F'),
 ('2022-09-15', 'Se evita la huelga del ferrocarril → ferroviarias', 'UNP CSX NSC', 'H'),
 ('2023-05-02', 'Empieza la huelga de guionistas → comprar Disney y Netflix', 'DIS NFLX', 'H'),
 ('2023-07-25', 'UPS evita la huelga → UPS', 'UPS', 'H'),
 ('2023-12-18', 'Ataques en el mar Rojo desvían a las navieras → navieras', 'ZIM MATX', 'H'),
 ('2024-09-13', 'Huelga de los mecánicos de Boeing → comprar Boeing', 'BA', 'H'),
 ('2024-10-01', 'Huelga en los puertos de la costa este → ferroviarias', 'UNP CSX NSC', 'H'),
 ('2023-01-12', 'Trian (Peltz) va a por Disney', 'DIS', 'W'),
 ('2023-01-23', 'Elliott entra en Salesforce', 'CRM', 'W'),
 ('2023-11-29', 'Elliott entra en Phillips 66', 'PSX', 'W'),
 ('2024-05-28', 'Elliott entra en Texas Instruments', 'TXN', 'W'),
 ('2024-06-10', 'Elliott entra en Southwest', 'LUV', 'W'),
 ('2024-10-07', 'Starboard entra en Pfizer', 'PFE', 'W'),
 ('2024-11-12', 'Elliott entra en Honeywell', 'HON', 'W'),
 ('2025-09-02', 'Elliott entra en PepsiCo', 'PEP', 'W'),
 # ── más variedad (2) ──
 ('2009-03-10', 'Citi dice que vuelve a ganar dinero → bancos', 'JPM BAC WFC C', 'E'),
 ('2010-05-10', 'Europa crea su fondo de rescate de 750.000 millones → bancos europeos', 'SAN DB ING', 'D'),
 ('2011-08-08', 'EE. UU. pierde la triple A → comprar el Nasdaq', 'QQQ', 'S'),
 ('2013-06-20', 'Bernanke anuncia el fin de las compras («taper tantrum») → comprar constructoras', 'DHI LEN PHM', 'S'),
 ('2014-10-15', 'Pánico relámpago de octubre de 2014 → comprar pequeñas', 'IWM', 'S'),
 ('2015-08-24', 'China devalúa y el mercado se hunde → comprar el Nasdaq', 'QQQ', 'S'),
 ('2016-02-11', 'Petróleo en mínimos de 13 años → comprar petroleras', 'XOM CVX COP', 'S'),
 ('2018-02-05', 'Estalla la apuesta contra la volatilidad → comprar el Nasdaq', 'QQQ', 'S'),
 ('2018-12-24', 'Nochebuena bajista de 2018 → comprar el Nasdaq', 'QQQ', 'S'),
 ('2019-08-05', 'China deja caer el yuan por la guerra comercial → comprar el Nasdaq', 'QQQ', 'S'),
 ('2020-03-09', 'Guerra de precios del petróleo → comprar petroleras', 'XOM CVX COP', 'S'),
 ('2022-06-13', 'La inflación sorprende al alza → comprar el Nasdaq', 'QQQ', 'S'),
 ('2024-08-05', 'Se deshace la apuesta con el yen → comprar el Nasdaq', 'QQQ', 'S'),
 ('2016-06-24', 'Brexit → comprar bancos británicos castigados', 'LYG BCS', 'S'),
 ('2013-03-25', 'Rescate de Chipre con quita a depositantes → bancos europeos', 'SAN DB ING', 'S'),
 ('2014-06-12', 'El Estado Islámico avanza en Irak → petroleras', 'XOM CVX', 'S'),
 ('2017-08-09', 'Corea del Norte amenaza a Guam → defensa', 'LMT NOC GD', 'S'),
 ('2019-06-20', 'Irán derriba un dron de EE. UU. → defensa', 'LMT NOC GD', 'S'),
 ('2010-07-15', 'Se aprueba la reforma financiera Dodd-Frank → bancos', 'JPM BAC GS', 'D'),
 ('2019-07-31', 'Primera bajada de tipos desde 2008 → constructoras', 'DHI LEN PHM', 'D'),
 ('2021-11-15', 'Se firma la ley de infraestructuras → cementeras y maquinaria', 'VMC MLM CAT', 'D'),
 ('2022-09-21', 'Tercera subida de 75 pb → bancos', 'JPM BAC WFC', 'D'),
 ('2024-03-20', 'La Fed mantiene tres bajadas para 2024 → pequeñas', 'IWM', 'D'),
 ('2017-04-03', 'Amazon y otras tiendas en línea crecen; cierre masivo de tiendas → comprar grandes almacenes', 'M KSS DDS', 'V'),
 ('2019-10-01', 'Charles Schwab elimina las comisiones → comprar los brókeres castigados', 'SCHW', 'V'),
 ('2020-11-17', 'Amazon abre farmacia → comprar la farmacia castigada CVS', 'CVS', 'V'),
 ('2022-11-30', 'ChatGPT → comprar al buscador castigado', 'GOOGL', 'V'),
 ('2012-10-18', 'Google se hunde por un adelanto de resultados → comprar Google', 'GOOGL', 'C'),
 ('2014-01-28', 'Apple decepciona en ventas de iPhone → comprar Apple', 'AAPL', 'C'),
 ('2016-01-20', 'Netflix y el temor a la competencia → comprar Netflix', 'NFLX', 'C'),
 ('2018-10-11', 'GE se desploma por sus cuentas → comprar GE', 'GE', 'C'),
 ('2019-02-22', 'Kraft Heinz recorta valor de marcas y se hunde un 27 % → comprar Kraft Heinz', 'KHC', 'C'),
 ('2022-02-24', 'Rusia invade Ucrania → comprar las aerolíneas castigadas', 'DAL UAL AAL', 'S'),
 ('2023-01-26', 'Intel se hunde por su previsión → comprar Intel', 'INTC', 'C'),
 ('2023-04-26', 'Meta gasta en IA y sube un 14 % → otras de publicidad', 'GOOGL PINS SNAP', 'B'),
 ('2024-01-31', 'Microsoft y Alphabet decepcionan con la IA → comprar Alphabet', 'GOOGL', 'C'),
 ('2019-02-22', 'Kraft Heinz se hunde → otras de alimentación', 'GIS CPB MDLZ', 'R'),
 ('2014-01-28', 'Apple decepciona → Samsung y proveedores (Qualcomm)', 'QCOM', 'R'),
 ('2016-01-20', 'Netflix se hunde → Disney', 'DIS', 'R'),
 ('2020-01-30', 'Primer brote de coronavirus fuera de China → mascarillas y guantes', 'MMM', 'A'),
 ('2016-03-22', 'Atentados de Bruselas → ciberseguridad y vigilancia', 'PANW FTNT', 'A'),
 ('2019-03-15', 'El Gobierno demanda a Facebook por privacidad → comprar Facebook', 'META', 'L'),
 ('2019-06-03', 'Se abren investigaciones antimonopolio a las grandes tecnológicas → comprar Alphabet y Facebook', 'GOOGL META', 'L'),
 ('2018-04-30', 'T-Mobile y Sprint anuncian su fusión → Verizon y AT&T', 'VZ T', 'M'),
 ('2022-03-07', 'Google compra Mandiant → otras de ciberseguridad', 'CRWD PANW ZS', 'M'),
 ('2019-08-07', 'Uber pierde mucho más de lo previsto → comprar a la castigada Lyft', 'LYFT', 'K'),
 ('2022-04-26', 'Alphabet decepciona en publicidad → comprar otra publicidad digital', 'META SNAP PINS', 'K'),
 ('2023-05-17', 'Target avisa de robos y menos ventas → comprar Walmart', 'WMT', 'K'),
 ('2016-04-14', 'Se firma el acuerdo de la OPEP en Doha y fracasa → comprar petroleras', 'XOM CVX COP', 'S'),
 ('2020-06-11', 'Segunda ola de contagios en EE. UU. → comprar las aerolíneas castigadas', 'DAL UAL AAL', 'S'),
 ('2021-03-23', 'Un portacontenedores bloquea el canal de Suez → navieras', 'ZIM MATX', 'S'),
 ('2024-01-08', 'La SEC aprueba los fondos de bitcoin al contado → mineras y bolsa de cripto', 'COIN MARA RIOT', 'D'),
 ('2016-11-09', 'Trump gana (2016) → constructoras de cárceles privadas', 'GEO CXW', 'E'),
 ('2020-11-04', 'Victoria de Biden y Congreso dividido (2020) → Nasdaq', 'QQQ', 'E'),
 ('2022-11-09', 'Elecciones de mitad de mandato (2022) → pequeñas', 'IWM', 'E'),
 ('2014-11-05', 'Los republicanos ganan el Senado (2014) → sanidad y energía', 'XLV XLE', 'E'),
 ('2023-03-31', 'La OPEP+ anuncia un recorte sorpresa → petroleras', 'XOM CVX COP', 'D'),
 ('2020-03-16', 'La Fed baja los tipos a cero un domingo → comprar bancos', 'JPM BAC WFC', 'D'),
 # ── más variedad ──
 ('2013-12-19', 'Robo de tarjetas en Target → ciberseguridad', 'PANW FTNT CHKP', 'A'),
 ('2014-11-24', 'Ataque informático a Sony → ciberseguridad', 'PANW FTNT CHKP', 'A'),
 ('2017-05-15', 'Virus WannaCry → ciberseguridad', 'PANW FTNT CHKP', 'A'),
 ('2020-12-14', 'Ataque a través de SolarWinds → ciberseguridad', 'CRWD PANW ZS FTNT', 'A'),
 ('2021-05-10', 'Ciberataque al oleoducto Colonial → ciberseguridad', 'CRWD PANW FTNT', 'A'),
 ('2023-01-11', 'Apagón del sistema de avisos de la FAA (2023) → contratistas del control aéreo', 'LDOS LHX RTX', 'A'),
 ('2015-03-25', 'Kraft y Heinz se fusionan → otras de alimentación', 'GIS CPB MDLZ', 'M'),
 ('2016-10-24', 'AT&T compra Time Warner → otros medios', 'DIS', 'M'),
 ('2019-01-03', 'Bristol compra Celgene → otra biotecnología', 'GILD BIIB XBI', 'M'),
 ('2020-11-25', 'Salesforce negocia comprar Slack → otro software de trabajo', 'ZM DOCU BOX', 'M'),
 ('2022-01-18', 'Microsoft compra Activision → otros videojuegos', 'EA TTWO', 'M'),
 ('2023-03-13', 'Pfizer compra Seagen → otra biotecnología', 'XBI', 'M'),
 ('2023-10-06', 'Exxon negocia comprar Pioneer → otras petroleras de esquisto', 'FANG DVN EOG', 'M'),
 ('2023-10-23', 'Chevron compra Hess → otras petroleras independientes', 'OXY DVN', 'M'),
 ('2023-11-30', 'AbbVie compra ImmunoGen → otra biotecnología', 'XBI', 'M'),
 ('2024-02-12', 'Diamondback compra Endeavor → otras petroleras de esquisto', 'CTRA DVN', 'M'),
 ('2024-08-14', 'Mars compra Kellanova → otras de alimentación', 'GIS CPB SJM HSY', 'M'),
 ('2018-10-23', 'Caterpillar avisa de que ha tocado techo → comprar otra maquinaria', 'DE CMI', 'K'),
 ('2022-05-24', 'Snap avisa y se hunde un 43 % → comprar otra publicidad digital', 'META PINS GOOGL', 'K'),
 ('2022-07-26', 'Walmart recorta previsiones → comprar otras cadenas', 'TGT COST AMZN', 'K'),
 ('2022-07-29', 'Intel se hunde tras resultados → comprar otros chips', 'AMD NVDA', 'K'),
 ('2022-08-09', 'Micron avisa de menos demanda → comprar otros chips', 'NVDA AMD', 'K'),
 ('2022-09-16', 'FedEx avisa y se hunde un 21 % → comprar otro transporte', 'UPS XPO', 'K'),
 ('2023-08-31', 'Dollar General se hunde → comprar Dollar Tree', 'DLTR', 'K'),
 ('2020-10-20', 'El Gobierno demanda a Google → comprar Alphabet', 'GOOGL', 'L'),
 ('2020-12-09', 'El regulador demanda a Facebook → comprar Facebook', 'META', 'L'),
 ('2022-12-08', 'El regulador intenta frenar la compra de Activision → comprar Microsoft', 'MSFT', 'L'),
 ('2023-09-26', 'El regulador demanda a Amazon → comprar Amazon', 'AMZN', 'L'),
 ('2024-03-21', 'El Gobierno demanda a Apple → comprar Apple', 'AAPL', 'L'),
 ('2018-01-30', 'Amazon, Berkshire y JPMorgan entran en sanidad → comprar las aseguradoras castigadas', 'UNH CVS CI', 'V'),
 ('2020-11-17', 'Amazon abre farmacia → comprar las farmacias castigadas', 'CVS WBA GDRX', 'V'),
 ('2023-10-05', 'Los fármacos de la obesidad reducen la compra de comida → comprar alimentación castigada', 'GIS KHC PEP MDLZ', 'V'),
 ('2014-10-01', 'Primer caso de ébola en EE. UU. → comprar las aerolíneas castigadas', 'DAL UAL AAL', 'S'),
 ('2023-09-15', 'Huelga del automóvil en EE. UU. → comprar GM y Ford', 'GM F', 'S'),
 ('2010-01-27', 'Toyota para sus ventas por los aceleradores → comprar Toyota', 'TM', 'C'),
 ('2021-05-05', 'Peloton retira sus cintas → comprar Peloton', 'PTON', 'C'),
 ('2010-01-27', 'Toyota para sus ventas → Ford', 'F', 'R'),
 ('2016-10-11', 'Samsung retira el Galaxy Note 7 → Apple', 'AAPL', 'R'),
 ('2022-11-08', 'Quiebra FTX → Coinbase', 'COIN', 'R'),
 # O · fusiones u OPAs que se rompen (fecha = primer día de reacción)
 ('2018-03-13', 'Trump bloquea la compra de Qualcomm por Broadcom → comprar Qualcomm', 'QCOM', 'O'),
 ('2018-07-26', 'Qualcomm renuncia a comprar NXP → comprar NXP', 'NXPI', 'O'),
 ('2021-07-26', 'Aon y Willis Towers Watson rompen su fusión → comprar Willis', 'WTW', 'O'),
 ('2023-08-16', 'Intel renuncia a comprar Tower → comprar Tower', 'TSEM', 'O'),
 ('2024-10-25', 'El juez bloquea la compra de Capri por Tapestry → comprar Capri', 'CPRI', 'O'),
 ('2024-12-10', 'Los jueces bloquean la fusión de Kroger y Albertsons → comprar Albertsons', 'ACI', 'O'),
 ('2025-10-17', 'Fracasa la OPA de BBVA sobre el Sabadell → comprar el Sabadell', 'SAB.MC', 'O'),
 # T · aranceles → comprar a los castigados
 ('2018-03-22', 'Trump anuncia aranceles a China (2018) → comprar a los exportadores castigados', 'BA CAT DE', 'T'),
 ('2018-05-31', 'EE. UU. pone aranceles al acero europeo → comprar ArcelorMittal', 'MT.AS', 'T'),
 ('2019-05-06', 'Trump sube los aranceles a China (mayo de 2019) → comprar a los castigados', 'AAPL CAT DE', 'T'),
 ('2019-08-01', 'Trump anuncia aranceles al resto de importaciones chinas → comprar a los castigados', 'AAPL NKE BBY', 'T'),
 ('2025-04-03', 'Aranceles del «Día de la Liberación» → comprar a los importadores castigados', 'AAPL NKE BBY RH DECK', 'T'),
 ('2025-04-03', 'Aranceles del «Día de la Liberación» → comprar los coches alemanes castigados', 'VOW3.DE MBG.DE BMW.DE', 'T'),
 ('2002-03-05', 'Bush pone aranceles al acero (2002) → acereras de EE. UU.', 'NUE STLD', 'S'),
 # U · multas y normas de Bruselas contra un gigante
 ('2004-03-24', 'Bruselas multa a Microsoft con 497 millones → comprar Microsoft', 'MSFT', 'U'),
 ('2016-08-30', 'Bruselas obliga a Apple a pagar 13.000 millones en impuestos → comprar Apple', 'AAPL', 'U'),
 ('2017-06-27', 'Bruselas multa a Google con 2.420 millones (Shopping) → comprar Alphabet', 'GOOGL', 'U'),
 ('2018-07-18', 'Bruselas multa a Google con 4.340 millones (Android) → comprar Alphabet', 'GOOGL', 'U'),
 ('2019-03-20', 'Bruselas multa a Google con 1.490 millones (AdSense) → comprar Alphabet', 'GOOGL', 'U'),
 ('2023-05-22', 'Multa récord de 1.200 millones a Meta por los datos de los europeos → comprar Meta', 'META', 'U'),
 ('2024-03-04', 'Bruselas multa a Apple con 1.800 millones (música) → comprar Apple', 'AAPL', 'U'),
 ('2025-04-23', 'Primeras multas de la ley de mercados digitales a Apple y Meta → comprarlas', 'AAPL META', 'U'),
 ('2025-09-05', 'Bruselas multa a Google con 2.950 millones (publicidad) → comprar Alphabet', 'GOOGL', 'U'),
 # Y · grandes recompras (casi siempre llegan junto a los resultados: el primer día mezcla las dos cosas)
 ('2018-05-02', 'Apple anuncia una recompra de 100.000 millones → comprar Apple', 'AAPL', 'Y'),
 ('2018-07-18', 'Berkshire cambia su política para recomprar acciones → comprar Berkshire', 'BRK-B', 'Y'),
 ('2021-04-28', 'Alphabet anuncia una recompra de 50.000 millones → comprar Alphabet', 'GOOGL', 'Y'),
 ('2023-02-02', 'Meta amplía su recompra en 40.000 millones → comprar Meta', 'META', 'Y'),
 ('2023-05-05', 'Apple anuncia una recompra de 90.000 millones → comprar Apple', 'AAPL', 'Y'),
 ('2024-04-26', 'Alphabet anuncia una recompra de 70.000 millones y su primer dividendo → comprar Alphabet', 'GOOGL', 'Y'),
 ('2024-05-03', 'Apple anuncia una recompra de 110.000 millones → comprar Apple', 'AAPL', 'Y'),
 ('2024-08-29', 'Nvidia anuncia una recompra de 50.000 millones → comprar Nvidia', 'NVDA', 'Y'),
 # Europa y España (se miden frente a la media de las grandes europeas)
 ('2012-06-11', 'Rescate a la banca española → comprar los bancos españoles', 'SAN.MC BBVA.MC SAB.MC BKT.MC', 'D'),
 ('2012-07-26', 'Draghi: «haré lo que haga falta» → bancos del sur de Europa', 'SAN.MC BBVA.MC ISP.MI UCG.MI', 'D'),
 ('2015-01-22', 'El BCE lanza la compra masiva de deuda → bancos europeos', 'SAN.MC BBVA.MC BNP.PA ISP.MI', 'D'),
 ('2016-06-24', 'Brexit → comprar los bancos españoles castigados', 'SAN.MC BBVA.MC SAB.MC', 'S'),
 ('2020-11-09', 'Vacuna de Pfizer (2020) → turismo español', 'IAG.MC AMS.MC MEL.MC AENA.MC', 'E'),
 ('2022-07-12', 'Impuesto a la banca española → comprar los bancos castigados', 'SAN.MC BBVA.MC CABK.MC SAB.MC BKT.MC', 'N'),
 ('2022-07-12', 'Impuesto a las eléctricas y petroleras españolas → comprar el castigo', 'IBE.MC ELE.MC REP.MC NTGY.MC', 'N'),
 ('2022-07-21', 'Primera subida de tipos del BCE en once años → bancos españoles', 'SAN.MC BBVA.MC CABK.MC SAB.MC BKT.MC', 'D'),
 ('2023-03-20', 'UBS compra Credit Suisse → comprar los bancos europeos castigados', 'SAN.MC BBVA.MC BNP.PA INGA.AS DBK.DE', 'S'),
 ('2024-05-09', 'OPA hostil de BBVA sobre el Sabadell → las siguientes candidatas', 'UNI.MC BKT.MC', 'M'),
 ('2010-05-10', 'Europa crea el fondo de rescate de 750.000 millones → bancos del sur', 'SAN.MC BBVA.MC ISP.MI UCG.MI', 'D'),
 ('2020-07-21', 'Acuerdo del fondo europeo de recuperación → bancos del sur', 'SAN.MC BBVA.MC ISP.MI UCG.MI', 'D'),
 ('2025-03-03', 'Cumbre de Londres: Europa se rearma → defensa europea', 'RHM.DE LDO.MI HO.PA', 'D'),
 ('2022-02-24', 'Rusia invade Ucrania → defensa europea', 'RHM.DE LDO.MI HO.PA', 'E'),
 ('2022-02-24', 'Rusia invade Ucrania → petroleras europeas', 'TTE.PA ENI.MI REP.MC', 'E'),
 ('2015-07-06', 'Grecia vota «no» en el referéndum → comprar los bancos del sur castigados', 'SAN.MC BBVA.MC ISP.MI UCG.MI', 'S'),
 ('2018-05-29', 'Crisis política en Italia → comprar los bancos italianos castigados', 'ISP.MI UCG.MI', 'S'),
 ('2024-06-10', 'Macron convoca elecciones anticipadas → comprar los bancos franceses castigados', 'BNP.PA GLE.PA ACA.PA', 'S'),
 ('2021-11-26', 'Variante ómicron → comprar las aerolíneas europeas castigadas', 'IAG.MC LHA.DE AF.PA', 'S'),
 ('2023-08-08', 'Italia anuncia un impuesto a la banca → comprar los bancos italianos castigados', 'ISP.MI UCG.MI', 'N'),
 ('2024-01-09', 'Informe de Gotham contra Grifols → comprar el castigo', 'GRF.MC', 'X'),
 ('2023-08-17', 'Adyen se hunde un 39 % tras resultados → comprar Adyen', 'ADYEN.AS', 'C'),
 ('2018-08-13', 'Primera condena por el glifosato → comprar Bayer', 'BAYN.DE', 'C'),
 ('2011-03-14', 'Merkel frena las nucleares tras Fukushima → comprar RWE y E.ON castigadas', 'RWE.DE EOAN.DE', 'N'),
 ('2012-09-06', 'El BCE anuncia la compra ilimitada de deuda → bancos del sur', 'SAN.MC BBVA.MC ISP.MI UCG.MI', 'D'),
 ('2014-06-05', 'El BCE pone los tipos en negativo → bancos europeos', 'SAN.MC BBVA.MC BNP.PA ISP.MI', 'D'),
 ('2015-09-21', 'Escándalo del diésel → comprar Volkswagen', 'VOW3.DE', 'C'),
 ('2015-09-21', 'Escándalo del diésel → BMW y Mercedes', 'BMW.DE MBG.DE', 'R'),
 ('2016-09-16', 'EE. UU. pide 14.000 millones a Deutsche Bank → comprar Deutsche Bank', 'DBK.DE', 'C'),
 ('2017-04-24', 'Macron gana la primera vuelta → bancos franceses', 'BNP.PA GLE.PA ACA.PA', 'E'),
 ('2017-10-02', 'Referéndum en Cataluña → comprar CaixaBank y Sabadell castigados', 'CABK.MC SAB.MC', 'S'),
 ('2021-09-14', 'El Gobierno recorta los beneficios de las eléctricas → comprar el castigo', 'IBE.MC ELE.MC NTGY.MC', 'N'),
 ('2023-03-08', 'Países Bajos limitan las ventas de ASML a China → comprar ASML', 'ASML.AS', 'N'),
 ('2024-10-15', 'ASML se hunde por sus previsiones → comprar ASML', 'ASML.AS', 'C'),
]
done = []
for d, name, b, tp in CASES:
    d = pd.Timestamp(d)
    if d not in pos: d = idx[idx.searchsorted(d)]
    i = pos[d]; b = [x for x in b.split() if x in col and np.isfinite(Av[i, col[x]])]
    if not b: print('  sin datos:', name); continue
    v = lambda x: None if not np.isfinite(x) else round(100 * x, 2)
    if all(is_eu(x) for x in b) and EUc:                                   # europeas: frente a la media de las grandes europeas, que cotizan en su mismo horario
        done.append({'n': 1, 'd0': v(day0_eu(i, b)), 'e5': v(after_eu(i, b, 5)), 'e20': v(after_eu(i, b, 20)), 'e60': v(after_eu(i, b, 60)), 'valores': b, 'fecha': str(d.date()), 'tipo': tp, 'nombre': name, 'nota': 'frente a la media de las grandes europeas', 'region': 'Europa'})
        continue
    done.append({'n': 1, 'd0': v(day0(i, b)), 'e5': v(after(i, b, 5)), 'e20': v(after(i, b, 20)), 'e60': v(after(i, b, 60)), 'valores': b, 'fecha': str(d.date()), 'tipo': tp, 'nombre': name})
G = 'Afinar · casos con nombre según cómo fue el primer día'
US_ = lambda c_: c_.get('region') != 'Europa'                           # los resúmenes de EE. UU. no mezclan casos medidos frente a otro índice
def bucket(nombre, logica, sel):
    x = [c_ for c_ in done if US_(c_) and sel(c_)]
    m = lambda k: (lambda v: round(float(np.mean(v)), 2) if v else None)([c_[k] for c_ in x if c_[k] is not None])
    hh = lambda k: (lambda v: int(round(100 * np.mean([y > 0 for y in v]))) if v else None)([c_[k] for c_ in x if c_[k] is not None])
    add(G, nombre, logica, {'n': len(x), 'd0': m('d0'), 'e5': m('e5'), 'e20': m('e20'), 'e60': m('e60'), 'hit20': hh('e20'), 'hit60': hh('e60')})
ben = lambda c_: c_['tipo'] in 'EDSAMB' and 'comprar' not in c_['nombre'].lower()
vix_at = lambda c_: vixp[pos[pd.Timestamp(c_['fecha'])]] if pd.Timestamp(c_['fecha']) in pos else np.nan
bucket('Hechos consumados con el miedo alto ese día (VIX ≥ 25)', 'Un cambio duradero que llega en plena tormenta.', lambda c_: c_['tipo'] == 'E' and vix_at(c_) >= 25)
bucket('Hechos consumados con el miedo bajo ese día (VIX < 25)', 'Un cambio duradero que llega en calma.', lambda c_: c_['tipo'] == 'E' and vix_at(c_) < 25)
bucket('Hechos consumados cuyo primer día fue pequeño (< 3 %)', 'El mercado aún no ha visto todo el alcance.', lambda c_: c_['tipo'] == 'E' and c_['d0'] is not None and c_['d0'] < 3)
bucket('Hechos consumados cuyo primer día fue grande (≥ 3 %)', 'El mercado ya reaccionó con fuerza.', lambda c_: c_['tipo'] == 'E' and c_['d0'] is not None and c_['d0'] >= 3)
bucket('Sustos · comprar al castigado con el miedo alto (VIX ≥ 25)', 'Castigo en plena tormenta.', lambda c_: c_['tipo'] == 'S' and 'comprar' in c_['nombre'].lower() and vix_at(c_) >= 25)
bucket('Sustos · comprar al castigado con el miedo bajo (VIX < 25)', 'Castigo en calma.', lambda c_: c_['tipo'] == 'S' and 'comprar' in c_['nombre'].lower() and vix_at(c_) < 25)
bucket('Gigante caído famoso con el miedo alto (VIX ≥ 25)', 'Escándalo o mal resultado en plena tormenta.', lambda c_: c_['tipo'] == 'C' and vix_at(c_) >= 25)
bucket('Beneficiarios que el primer día casi no se mueven (menos de un 2 %)', 'El mercado no ha visto la conexión.', lambda c_: ben(c_) and c_['d0'] is not None and c_['d0'] < 2)
bucket('Beneficiarios que el primer día suben entre un 2 % y un 6 %', 'Reacción normal.', lambda c_: ben(c_) and c_['d0'] is not None and 2 <= c_['d0'] < 6)
bucket('Beneficiarios que el primer día suben más de un 6 %', 'Reacción fuerte: ¿queda algo?', lambda c_: ben(c_) and c_['d0'] is not None and c_['d0'] >= 6)
bucket('Sustos · comprar al beneficiario', 'Defensa tras un atentado, reformas tras un huracán.', lambda c_: c_['tipo'] == 'S' and 'comprar' not in c_['nombre'].lower())
bucket('Sustos · comprar al castigado', 'Aerolíneas tras un brote, bancos tras una quiebra.', lambda c_: c_['tipo'] == 'S' and 'comprar' in c_['nombre'].lower())
bucket('Decisiones anunciadas en las que el primer día ya fue en contra', 'El mercado vendió la noticia desde el primer minuto.', lambda c_: c_['tipo'] == 'D' and c_['d0'] is not None and c_['d0'] < 0)
bucket('Decisiones anunciadas en las que el primer día fue a favor', 'El mercado la celebró.', lambda c_: c_['tipo'] == 'D' and c_['d0'] is not None and c_['d0'] >= 0)
bucket('Gigante caído por un escándalo, accidente o demanda', 'El negocio no ha cambiado, la reputación sí.', lambda c_: c_['tipo'] == 'C' and bool(re.search(r'SEC|coli|falsas|pasajero|datos|Analytica|Talco|accidente|puerta|Deepwater|ballena|Informe|demanda|Apagón|retira|para sus ventas', c_['nombre'])))
bucket('Gigante caído por sus propios resultados o previsiones', 'El negocio va peor de lo que se creía.', lambda c_: c_['tipo'] == 'C' and not re.search(r'SEC|coli|falsas|pasajero|datos|Analytica|Talco|accidente|puerta|Deepwater|ballena|Informe|demanda|Apagón|retira|para sus ventas', c_['nombre']))
G = 'Casos con nombre · resumen por tipo'
for tp, lab in TIPOS.items():
    cs = [c for c in done if c['tipo'] == tp and US_(c)]
    m = lambda k: (lambda x: round(float(np.mean(x)), 2) if x else None)([c[k] for c in cs if c[k] is not None])
    h = lambda k: (lambda x: int(round(100 * np.mean([y > 0 for y in x]))) if x else None)([c[k] for c in cs if c[k] is not None])
    add(G, lab, f'{len(cs)} casos elegidos a mano', {'n': len(cs), 'd0': m('d0'), 'e5': m('e5'), 'e20': m('e20'), 'e60': m('e60'), 'hit20': h('e20'), 'hit60': h('e60'), 'tipo': tp})
G = 'Europa · casos con nombre por tipo (frente a la media de las grandes europeas)'
for tp, lab in TIPOS.items():
    cs = [c for c in done if c['tipo'] == tp and not US_(c)]
    if not cs: continue
    m = lambda k: (lambda x: round(float(np.mean(x)), 2) if x else None)([c[k] for c in cs if c[k] is not None])
    h = lambda k: (lambda x: int(round(100 * np.mean([y > 0 for y in x]))) if x else None)([c[k] for c in cs if c[k] is not None])
    add(G, lab, f'{len(cs)} casos europeos elegidos a mano', {'n': len(cs), 'd0': m('d0'), 'e5': m('e5'), 'e20': m('e20'), 'e60': m('e60'), 'hit20': h('e20'), 'hit60': h('e60')})
G = 'Casos con nombre'
for c in sorted(done, key=lambda c: ('EDSNVAMKLCRBJXPIFHWOTUY'.index(c['tipo']), c['fecha'])):
    add(G, c['nombre'], TIPOS[c['tipo']].split(' (')[0], {k: v for k, v in c.items() if k != 'nombre'})
# ── salidas: cuánto cae una idea buena por el camino y si salir antes (o con un stop) mejora o empeora ──
G = 'Salidas · cuánto cae por el camino y cuándo salir'
def camino(i, syms):
    """Rendimiento ABSOLUTO de la cesta (a partes iguales) desde la apertura siguiente al día i hasta el cierre de cada una de las 60 sesiones."""
    if i + 61 >= len(idx): return None
    c_ = [col[s_] for s_ in syms if s_ in col]
    if not c_: return None
    w = pd.DataFrame(Av[i + 1:i + 61, c_] / Ov[i + 1, c_] - 1).ffill()          # un festivo de esa bolsa (el calendario es el de Nueva York) repite el último cierre
    with np.errstate(all='ignore'):
        p_ = np.nanmean(w.values, axis=1)
    return p_ if np.isfinite(p_).all() else None
def salida(nombre, logica, evs_):
    P_ = [p_ for p_ in (camino(i, b_) for i, b_ in evs_) if p_ is not None]
    if len(P_) < 5: return
    M = np.array(P_); mae = M.min(axis=1); f5, f20, f60 = M[:, 4], M[:, 19], M[:, 59]
    def stop(lv):                                                     # stop al cierre: se sale al cierre del primer día que la cesta está por debajo del nivel
        out = []
        for p_ in M:
            k = np.flatnonzero(p_ <= lv)
            out.append(p_[k[0]] if len(k) else p_[-1])
        return float(np.mean(out))
    hondo = mae <= -0.10
    r_ = {'n': len(M), 'd0': None, 'e5': round(100 * float(f5.mean()), 2), 'e20': round(100 * float(f20.mean()), 2), 'e60': round(100 * float(f60.mean()), 2),
          'hit20': int(round(100 * float((f20 > 0).mean()))), 'hit60': int(round(100 * float((f60 > 0).mean()))),
          'mae_med': round(100 * float(np.median(mae)), 1), 'mae_p80': round(100 * float(np.percentile(mae, 20)), 1),
          'stop10': round(100 * stop(-0.10), 2), 'stop15': round(100 * stop(-0.15), 2), 'mejor60': int(round(100 * float((f60 > f20).mean()))),
          'rec10': int(round(100 * float((f60[hondo] > 0).mean()))) if hondo.any() else None, 'n_hondo': int(hondo.sum())}
    fm = lambda x: f"{x:+.1f}".replace('.', ',') + ' %'
    r_['nota'] = (f"en precio, no frente al índice. Por el camino cae de mediana un {fm(r_['mae_med'])} y, en el peor 20 % de los casos, más de un {fm(r_['mae_p80'])}. "
                  f"A 60 sesiones gana de media {fm(r_['e60'])}; con un stop al −10 % (al cierre) daría {fm(r_['stop10'])} y con uno al −15 %, {fm(r_['stop15'])}. "
                  + (f"De los {r_['n_hondo']} casos que llegaron a caer un 10 %, el {r_['rec10']} % acabó en positivo. " if r_['n_hondo'] >= 5 else '')
                  + f"Aguantar de la sesión 20 a la 60 mejoró el resultado en el {r_['mejor60']} % de los casos.")
    add(G, nombre, logica, r_)
salida('Un tramo cualquiera del S&P 500 (para comparar)', 'Comprar el SPY un día cualquiera.', [(i, ['SPY']) for i in range(int(np.argmax(since)) + 1, len(idx) - 62, 10)])
pm28 = thin(idx[pan2 & (vixp >= 28)])
salida('Pánico con miedo alto → Nasdaq-100', 'La idea de pánico tal y como se da.', [(i, ['QQQ']) for i in pm28])
if 'SMH' in col: salida('Pánico con miedo alto → semiconductores', 'La variante que más rebota.', [(i, ['SMH']) for i in pm28])
salida('Una grande cae ≥ 10 % con el miedo alto → la propia caída', 'Comprar al castigado en plena tormenta.', [(e[0], [e[1]]) for e in evs['caida'] if vxs(e) >= 28])
salida('Una grande cae ≥ 10 % con el miedo bajo → la propia caída', 'Comprar al castigado en calma (la que no funciona).', [(e[0], [e[1]]) for e in evs['caida'] if vxs(e) < 18])
salida('El líder sube ≥ 10 % y castigan a sus competidoras con miedo → las competidoras', 'Castigo en plena tormenta.', [(e[0], e[2]) for e in evs['salto'] if e[4] <= -0.01 and vxs(e) >= 25])
cs_ = lambda tp: [(pos[pd.Timestamp(c_['fecha'])], c_['valores']) for c_ in done if c_['tipo'] == tp and US_(c_) and pd.Timestamp(c_['fecha']) in pos]
salida('Sector castigado por una norma → comprar el castigo (casos con nombre)', 'Como la idea del veto de chips.', cs_('N'))
salida('Hechos consumados → el beneficiario (casos con nombre)', 'La idea de cambio duradero.', cs_('E'))
if EUc and '^STOXX50E' in col and '^IBEX' in col:
    salida('Europa · Euro Stoxx 50 −2 % con el miedo alto → Euro Stoxx 50', 'El pánico en Europa.', [(i, ['^STOXX50E']) for i in thin(idx[since & (rE <= -0.02) & (vixp >= 28)])])
    salida('Europa · Ibex 35 −2 % con el miedo alto → Ibex 35', 'El pánico en España.', [(i, ['^IBEX']) for i in thin(idx[since & (rI <= -0.02) & (vixp >= 28)])])

# ── horizontes: las mismas ideas medidas a 5, 10, 20, 40, 60, 90 y 120 sesiones, para saber cuándo se cobra cada una (no cambia ningún otro ensayo) ──
G = 'Horizontes · cuándo se cobra cada idea'
HZ = (5, 10, 20, 40, 60, 90, 120)
_ab = {}
def abase(sym, h):
    """Lo que hace el valor en un tramo cualquiera de h sesiones (para medir un índice comprado tras un hecho)."""
    if (sym, h) not in _ab:
        j = col[sym]; ii = np.arange(int(np.argmax(since)) + 1, len(idx) - h - 2, 3)
        with np.errstate(all='ignore'):
            _ab[(sym, h)] = float(np.nanmean(Av[ii + h, j] / Ov[ii + 1, j] - 1))
    return _ab[(sym, h)]
def horiz(nombre, logica, evs_, medida, nota):
    """evs_: lista de eventos; medida(e, h) → resultado del evento a h sesiones (ya restado lo normal). Solo cuentan los que tienen las 120 sesiones."""
    E_ = [e for e in evs_ if e[0] + 1 + max(HZ) < len(idx)]
    M = np.array([[medida(e, h) for h in HZ] for e in E_], dtype=float) if E_ else np.zeros((0, len(HZ)))
    M = M[np.isfinite(M).all(axis=1)] if len(M) else M
    if len(M) < 5: return
    mu = M.mean(axis=0); sd = M.std(axis=0, ddof=1); t = mu / sd * np.sqrt(len(M))
    k = int(np.argmax(mu)); kt = int(np.argmax(t)); ritmo = mu / np.array(HZ)
    fm = lambda x: f"{100 * x:+.1f}".replace('.', ',') + ' %'
    r_ = {'n': int(len(M)), 'd0': None, 'e5': round(100 * mu[0], 2), 'e20': round(100 * mu[2], 2), 'e60': round(100 * mu[4], 2), 't5': round(float(t[0]), 1), 't20': round(float(t[2]), 1), 't60': round(float(t[4]), 1),
          'curva': {str(h): round(100 * float(v), 2) for h, v in zip(HZ, mu)}, 'tcurva': {str(h): round(float(v), 1) for h, v in zip(HZ, t)}, 'mejor_h': HZ[k], 'mejor_t_h': HZ[kt]}
    r_['nota'] = (nota + '. ' + ' · '.join(f'{h} s: {fm(v)} (t {tt:.1f})'.replace('.', ',') for h, v, tt in zip(HZ, mu, t))
                  + f'. Máximo a las {HZ[k]} sesiones; donde más se distingue del azar, a las {HZ[kt]}. Ritmo por sesión: {fm(ritmo[0])} en la primera semana, {fm(ritmo[2])} hasta la 20, {fm(ritmo[4])} hasta la 60')
    add(G, nombre, logica, r_)
_ub2 = {}
def ubase2(h):
    """Como ubase(), pero válida hasta 120 sesiones: lo que gana de más sobre el SPY una grande cualquiera en h sesiones (mismas fechas para todos los horizontes)."""
    if h not in _ub2:
        ii = np.arange(int(np.argmax(since)) + 61, len(idx) - max(HZ) - 2, 5); v = []
        for i in ii:
            c_ = np.flatnonzero(rk[i - 1] <= 300)
            with np.errstate(all='ignore'):
                v.append(np.nanmean(Av[i + h, c_] / Ov[i + 1, c_] - 1) - (Av[i + h, col['SPY']] / Ov[i + 1, col['SPY']] - 1))
        _ub2[h] = float(np.nanmean(v))
    return _ub2[h]
idx_m = lambda sym: (lambda e, h: (Av[e[0] + h, col[sym]] / Ov[e[0] + 1, col[sym]] - 1) - abase(sym, h))
st_self = lambda e, h: after(e[0], [e[1]], h) - ubase2(h)
st_peer = lambda e, h: after(e[0], e[2], h) - ubase2(h)
nm = lambda e, h: after(e[0], e[1], h)
pm28_ = [(i,) for i in thin(idx[pan2 & (vixp >= 28)])]
horiz('Pánico con miedo alto → Nasdaq-100', 'La idea de pánico.', pm28_, idx_m('QQQ'), 'Rendimiento del QQQ menos lo que hace en un tramo cualquiera')
if 'SMH' in col: horiz('Pánico con miedo alto → semiconductores', 'La variante que más rebota.', pm28_, idx_m('SMH'), 'Rendimiento del SMH menos lo que hace en un tramo cualquiera')
horiz('Una grande cae ≥ 10 % con el miedo alto → comprar la caída', 'El castigo en plena tormenta.', [e for e in evs['caida'] if vxs(e) >= 28], st_self, 'Exceso sobre el SPY menos el de una grande cualquiera')
horiz('Una grande cae ≥ 10 % con el miedo bajo → comprar la caída', 'El mismo castigo en calma (para comparar).', [e for e in evs['caida'] if vxs(e) < 18], st_self, 'Exceso sobre el SPY menos el de una grande cualquiera')
horiz('El líder sube ≥ 10 % y castigan a sus competidoras con miedo → las competidoras', 'Castigo en plena tormenta.', [e for e in evs['salto'] if e[4] <= -0.01 and vxs(e) >= 25], st_peer, 'Exceso sobre el SPY menos el de una grande cualquiera')
horiz('Una grande cae ≥ 10 % y sus competidoras suben ≥ 2 % ese día → las competidoras', 'Trasvase de negocio.', [e for e in evs['caida'] if e[4] >= 0.02], st_peer, 'Exceso sobre el SPY menos el de una grande cualquiera')
csh = lambda tp: [(pos[pd.Timestamp(c_['fecha'])], c_['valores']) for c_ in done if c_['tipo'] == tp and US_(c_) and pd.Timestamp(c_['fecha']) in pos]
horiz('Hechos consumados → el beneficiario (casos con nombre)', 'Cambio duradero.', csh('E'), nm, 'Exceso sobre el SPY')
horiz('Sector castigado por una norma → comprar el castigo (casos con nombre)', 'Como la idea del veto de chips.', csh('N'), nm, 'Exceso sobre el SPY')
horiz('Anuncio de split → comprar (casos con nombre)', 'Empuje de pequeños inversores.', csh('P'), nm, 'Exceso sobre el SPY')
horiz('Anuncio de entrada en el S&P 500 → comprar (casos con nombre)', 'Empuje de los fondos indexados.', csh('I'), nm, 'Exceso sobre el SPY')
if EUc and '^STOXX50E' in col and '^IBEX' in col:
    horiz('Europa · Euro Stoxx 50 −2 % con el miedo alto → Euro Stoxx 50', 'El pánico en Europa.', [(i,) for i in thin(idx[since & (rE <= -0.02) & (vixp >= 28)])], idx_m('^STOXX50E'), 'Rendimiento del índice menos lo que hace en un tramo cualquiera')
    horiz('Europa · Ibex 35 −2 % con el miedo alto → Ibex 35', 'El pánico en España.', [(i,) for i in thin(idx[since & (rI <= -0.02) & (vixp >= 28)])], idx_m('^IBEX'), 'Rendimiento del índice menos lo que hace en un tramo cualquiera')
    horiz('Europa · una grande cae ≥ 10 % con el miedo alto → comprar la caída', 'El castigo en plena tormenta, en Europa.', [e for e in eu_ev['caida'] if vixp[e[0]] >= 28], lambda e, h: after_eu(e[0], [e[1]], h, e[2]), 'Exceso sobre la media de su grupo europeo')

for o in OUT:
    o['nombre'] = '↳ ' + o['nombre'].strip() if o['nombre'].startswith('   ') else o['nombre'].strip()
pb = os.path.join(ARGS.out, 'ideas_playbook.json')
P = json.load(open(pb, encoding='utf-8')) if os.path.exists(pb) else {'lecciones': [], 'metodo': '', 'cautelas': '', 'resumen': ''}
P['estudios'] = OUT
P['generado'] = time.strftime('%Y-%m-%d')
json.dump(P, open(pb, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
