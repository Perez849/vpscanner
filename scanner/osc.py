#!/usr/bin/env python3
"""
osc.py — BATERÍA DE OSCILADORES E INDICADORES COMPLEJOS para buscar información nueva sobre los rebotes (comando `osc`; NO interviene en producción).

El modelo de producción usa unas 28 variables sencillas (RSI(2), IBS, distancias a medias, ATR, VIX…) y su AUC fuera de muestra es ≈ 0,51–0,52: casi todo el acierto
viene de la tasa base. Aquí se calculan ~50 osciladores más elaborados, todos causales (la barra i solo usa datos hasta i):
estocástico, StochRSI, CCI, MFI, Chaikin (CMF y oscilador), OBV, Ultimate Oscillator, Elder-ray, TSI, KST, Vortex, Aroon, Fisher, RSI de Laguerre, Connors RSI,
eficiencia de Kaufman, razón de varianzas, dimensión fractal, razones de volatilidad (ATR, Parkinson, semivolatilidad), Bollinger/Keltner/squeeze, ADX/DMI, VWAP,
rangos percentiles, divergencia alcista RSI/precio, RSI «semanal», asimetría/curtosis, PPO/MACD, TRIX, Coppock, Schaff, Ichimoku, Supertrend, Donchian, calendario.

Protocolo (hay ~50 candidatos, así que la mejor de las pruebas siempre parecerá buena por azar):
  1. Cribado univariante: pendiente de la rentabilidad sobre el rango (dentro de cada año) con error estándar por clústeres de día, consistencia por años, y el umbral
     del MÁXIMO |t| de 50 variables obtenido permutando los resultados dentro de cada día (el «mejor de 50» bajo el azar).
  2. Modelo ampliado: el sistema entero reentrenado con (a) todos los osciladores y (b) los mejores elegidos SOLO con datos hasta 2018; se compara el AUC fuera de
     muestra y la política top-3/día con el modelo actual.
"""
from __future__ import annotations
import math
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.signal import lfilter

import feats

NAMES: List[str] = []



def _quiet(fn, *a, **kw):
    """Ejecuta fn silenciando lo que imprime (build_plan escribe un informe largo por cada versión)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)

def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([np.full(n - 1, np.nan), a])


def _roll(x: np.ndarray, n: int, fn) -> np.ndarray:
    if len(x) < n:
        return np.full(len(x), np.nan)
    return _pad(fn(swv(x, n), axis=1), n)


def _sh(x: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k, np.nan), x[:-k]]) if k > 0 else x


def _lf(b, a, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    xx = pd.Series(x).ffill().bfill().to_numpy() if ok.any() else np.zeros_like(x)
    y = lfilter(b, a, xx)
    return np.where(ok | np.isfinite(y), y, np.nan)


def _stoch(x: np.ndarray, hi: np.ndarray, lo: np.ndarray, n: int) -> np.ndarray:
    hh, ll = _roll(hi, n, np.max), _roll(lo, n, np.min)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(hh > ll, (x - ll) / (hh - ll), np.nan)


def _prank(x: np.ndarray, n: int) -> np.ndarray:
    """Percentil de la barra actual entre las últimas n (incluida)."""
    if len(x) < n:
        return np.full(len(x), np.nan)
    w = swv(x, n)
    cur = w[:, -1][:, None]
    return _pad(((w < cur).sum(1) + 0.5 * ((w == cur).sum(1) - 1)) / (n - 1), n)


def battery(F: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Osciladores en cada barra de una serie (causal). F viene de feats.build()."""
    t, o, h, l, c, v = F['t'], F['o'], F['h'], F['l'], F['c'], F['v'].astype(np.float64)
    n = len(c)
    atr = F['atr']
    cp, hp, lp = _sh(c, 1), _sh(h, 1), _sh(l, 1)
    out: Dict[str, np.ndarray] = {}
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        tp = (h + l + c) / 3.0
        rng = h - l
        # ── estocásticos y rangos
        out['stoch14'] = _stoch(c, h, l, 14)
        out['stoch5'] = _stoch(c, h, l, 5)
        out['stochrsi'] = _stoch(F['rsi14'], F['rsi14'], F['rsi14'], 14)
        out['donch55'] = _stoch(c, h, l, 55)
        out['donch20'] = _stoch(c, h, l, 20)
        # ── CCI, MFI, Chaikin, OBV
        sm = _sma(tp, 20)
        mad = _roll(tp, 20, lambda w, axis: np.abs(w - w.mean(axis=axis, keepdims=True)).mean(axis=axis))
        out['cci20'] = (tp - sm) / (0.015 * mad)
        rmf = tp * v
        up = np.where(tp > _sh(tp, 1), rmf, 0.0)
        dn = np.where(tp < _sh(tp, 1), rmf, 0.0)
        pos14, neg14 = pd.Series(up).rolling(14, min_periods=14).sum().to_numpy(), pd.Series(dn).rolling(14, min_periods=14).sum().to_numpy()
        out['mfi14'] = 100.0 - 100.0 / (1.0 + pos14 / np.where(neg14 > 0, neg14, np.nan))
        mfm = np.where(rng > 0, ((c - l) - (h - c)) / rng, 0.0)
        mfv = mfm * v
        out['cmf20'] = pd.Series(mfv).rolling(20, min_periods=20).sum().to_numpy() / pd.Series(v).rolling(20, min_periods=20).sum().to_numpy()
        adl = np.cumsum(mfv)
        v10 = pd.Series(v).rolling(10, min_periods=10).sum().to_numpy()
        out['adl_slope10'] = (adl - _sh(adl, 10)) / v10
        out['chaikin_osc'] = (_ema(adl, 3) - _ema(adl, 10)) / pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()
        obv = np.cumsum(np.where(c > cp, v, np.where(c < cp, -v, 0.0)))
        so = pd.Series(obv)
        out['obv_z20'] = ((so - so.rolling(20, min_periods=20).mean()) / so.rolling(20, min_periods=20).std(ddof=0)).to_numpy()
        # ── Ultimate Oscillator, Elder-ray, TSI, KST
        bp = c - np.minimum(l, cp)
        trr = np.maximum(h, cp) - np.minimum(l, cp)
        avg = lambda k: pd.Series(bp).rolling(k, min_periods=k).sum().to_numpy() / pd.Series(trr).rolling(k, min_periods=k).sum().to_numpy()
        out['uo'] = 100.0 * (4 * avg(7) + 2 * avg(14) + avg(28)) / 7.0
        e13 = _ema(c, 13)
        out['bull_power'] = (h - e13) / atr
        out['bear_power'] = (l - e13) / atr
        mom = c - cp
        out['tsi'] = 100.0 * _ema(_ema(mom, 25), 13) / _ema(_ema(np.abs(mom), 25), 13)
        roc = lambda k: (c / _sh(c, k) - 1.0) * 100.0
        kst = _sma(roc(10), 10) + 2 * _sma(roc(15), 10) + 3 * _sma(roc(20), 10) + 4 * _sma(roc(30), 15)
        out['kst_hist'] = kst - _sma(kst, 9)
        # ── Vortex, Aroon
        vmp, vmm = np.abs(h - lp), np.abs(l - hp)
        trs = pd.Series(np.maximum.reduce([h - l, np.abs(h - cp), np.abs(l - cp)])).rolling(14, min_periods=14).sum().to_numpy()
        out['vortex'] = (pd.Series(vmp).rolling(14, min_periods=14).sum().to_numpy() - pd.Series(vmm).rolling(14, min_periods=14).sum().to_numpy()) / trs
        if n >= 25:
            wh, wl = swv(h, 25), swv(l, 25)
            au = (24 - (24 - np.argmax(wh, axis=1))) / 24.0 * 100.0     # barras desde el máximo → 100 si es la última
            ad = (24 - (24 - np.argmin(wl, axis=1))) / 24.0 * 100.0
            out['aroon25'] = _pad(au - ad, 25)
        else:
            out['aroon25'] = np.full(n, np.nan)
        # ── Ehlers: Fisher, Laguerre; Connors RSI
        hl2 = (h + l) / 2.0
        st10 = _stoch(hl2, h, l, 10)
        x = np.clip(_lf([0.66], [1, -0.67], 2.0 * (st10 - 0.5)), -0.999, 0.999)
        out['fisher'] = _lf([0.5], [1, -0.5], np.log((1 + x) / (1 - x)))
        g = 0.5
        L0 = _lf([1 - g], [1, -g], c)
        L1 = _lf([-g, 1], [1, -g], L0)
        L2 = _lf([-g, 1], [1, -g], L1)
        L3 = _lf([-g, 1], [1, -g], L2)
        d01, d12, d23 = L0 - L1, L1 - L2, L2 - L3
        cu = np.maximum(d01, 0) + np.maximum(d12, 0) + np.maximum(d23, 0)
        cd = np.maximum(-d01, 0) + np.maximum(-d12, 0) + np.maximum(-d23, 0)
        out['laguerre_rsi'] = np.where(cu + cd > 0, cu / (cu + cd), np.nan)
        streak = F['up_streak'] - F['dn_streak']
        out['connors'] = (F['rsi3'] + feats._rsi(streak.astype(np.float64), 2) + 100.0 * _prank(F['ret1'], 100)) / 3.0
        # ── ruido/tendencia: eficiencia de Kaufman, razón de varianzas, dimensión fractal
        for k in (10, 30):
            out[f'er{k}'] = np.abs(c - _sh(c, k)) / pd.Series(np.abs(mom)).rolling(k, min_periods=k).sum().to_numpy()
        lr = np.log(c / cp)
        v1 = pd.Series(lr).rolling(60, min_periods=60).var().to_numpy()
        r5 = np.log(c / _sh(c, 5))
        v5 = pd.Series(r5).rolling(60, min_periods=60).var().to_numpy()
        out['vr5'] = v5 / (5.0 * v1)
        if n >= 30:
            w = swv(c, 30)
            lo_, hi_ = w.min(1, keepdims=True), w.max(1, keepdims=True)
            z = np.where(hi_ > lo_, (w - lo_) / (hi_ - lo_), 0.0)
            ln_ = np.sqrt(np.diff(z, axis=1) ** 2 + (1.0 / 29.0) ** 2).sum(1)
            out['fractal30'] = _pad(1.0 + (np.log(ln_) + math.log(2.0)) / math.log(2.0 * 29.0), 30)
            # asimetría/curtosis y fracción de subidas
        else:
            out['fractal30'] = np.full(n, np.nan)
        out['skew20'] = pd.Series(lr).rolling(20, min_periods=20).skew().to_numpy()
        out['kurt20'] = pd.Series(lr).rolling(20, min_periods=20).kurt().to_numpy()
        out['up_frac10'] = pd.Series((c > cp).astype(float)).rolling(10, min_periods=10).mean().to_numpy()
        # ── volatilidad relativa
        trf = np.maximum.reduce([h - l, np.abs(h - cp), np.abs(l - cp)])
        out['atr5_50'] = feats._wilder(trf, 5) / feats._wilder(trf, 50)
        pk = np.log(h / l) ** 2 / (4 * math.log(2))
        out['pk10_60'] = np.sqrt(pd.Series(pk).rolling(10, min_periods=10).mean().to_numpy() / pd.Series(pk).rolling(60, min_periods=60).mean().to_numpy())
        r20 = pd.Series(lr)
        out['dsv20'] = np.sqrt((np.minimum(r20, 0) ** 2).rolling(20, min_periods=20).mean().to_numpy() / (r20 ** 2).rolling(20, min_periods=20).mean().to_numpy())
        # ── Bollinger / Keltner / squeeze
        sd20 = pd.Series(c).rolling(20, min_periods=20).std(ddof=0).to_numpy()
        s20 = _sma(c, 20)
        bbw = 4 * sd20 / s20
        out['pctb'] = (c - (s20 - 2 * sd20)) / (4 * sd20)
        out['bbw_pr120'] = _prank(bbw, 120)
        kcw = 3.0 * _ema(trf, 20) / s20
        out['squeeze'] = (bbw < kcw).astype(np.float64)
        out['kc_pos'] = (c - _ema(c, 20)) / (1.5 * _ema(trf, 20))
        # ── ADX / DMI
        upm, dnm = h - hp, lp - l
        pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
        mdm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
        a14 = feats._wilder(trf, 14)
        pdi, mdi = 100 * feats._wilder(pdm, 14) / a14, 100 * feats._wilder(mdm, 14) / a14
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
        out['adx14'] = feats._wilder(np.nan_to_num(dx), 14)
        out['dmi_diff'] = pdi - mdi
        # ── VWAP, percentiles de rango/volumen/retorno
        out['vwap20_dist'] = (c - pd.Series(tp * v).rolling(20, min_periods=20).sum().to_numpy() / pd.Series(v).rolling(20, min_periods=20).sum().to_numpy()) / atr
        out['rng_pr100'] = _prank(rng, 100)
        out['vol_pr100'] = _prank(v, 100)
        out['ret_pr100'] = _prank(F['ret1'], 100)
        # ── divergencia alcista RSI/precio: mínimo reciente de precio más bajo que el previo, pero RSI(14) más alto
        mn5, mn_prev = pd.Series(l).rolling(5, min_periods=5).min().to_numpy(), _sh(pd.Series(l).rolling(10, min_periods=10).min().to_numpy(), 5)
        r5m, r_prev = pd.Series(F['rsi14']).rolling(5, min_periods=5).min().to_numpy(), _sh(pd.Series(F['rsi14']).rolling(10, min_periods=10).min().to_numpy(), 5)
        out['bull_div'] = np.where(np.isfinite(mn_prev) & np.isfinite(r_prev), ((mn5 < mn_prev) & (r5m > r_prev)).astype(np.float64), np.nan)
        # ── RSI «semanal»: 14 cierres espaciados de 5 en 5 barras
        if n > 75:
            idx = np.arange(70, n)
            lagged = np.stack([c[idx - 5 * k] for k in range(15)], 1)
            d = lagged[:, :-1] - lagged[:, 1:]
            su, sd_ = np.maximum(d, 0).sum(1), np.maximum(-d, 0).sum(1)
            w_rsi = np.full(n, np.nan)
            w_rsi[idx] = 100.0 * su / (su + sd_)
            out['wrsi14'] = w_rsi
        else:
            out['wrsi14'] = np.full(n, np.nan)
        # ── PPO / MACD, TRIX, Coppock, Schaff
        ppo = (_ema(c, 12) - _ema(c, 26)) / _ema(c, 26) * 100.0
        out['ppo_hist'] = ppo - _ema(ppo, 9)
        out['ppo_slope3'] = ppo - _sh(ppo, 3)
        e3 = _ema(_ema(_ema(c, 15), 15), 15)
        out['trix15'] = (e3 / _sh(e3, 1) - 1.0) * 10000.0
        rr = roc(14) + roc(11)
        w10 = np.arange(1, 11, dtype=np.float64)
        out['coppock'] = _roll(np.nan_to_num(rr), 10, lambda w, axis: (w * w10).sum(axis=axis) / w10.sum())
        macd = _ema(c, 23) - _ema(c, 50)
        f1 = _stoch(macd, macd, macd, 10)
        pf = _lf([0.5], [1, -0.5], f1)
        f2 = _stoch(pf, pf, pf, 10)
        out['schaff'] = _lf([0.5], [1, -0.5], f2)
        # ── Ichimoku
        tenkan = (_roll(h, 9, np.max) + _roll(l, 9, np.min)) / 2.0
        kijun = (_roll(h, 26, np.max) + _roll(l, 26, np.min)) / 2.0
        sa = _sh((tenkan + kijun) / 2.0, 26)
        sb = _sh((_roll(h, 52, np.max) + _roll(l, 52, np.min)) / 2.0, 26)
        out['ichi_cloud'] = (c - (sa + sb) / 2.0) / atr
        out['ichi_tk'] = (tenkan - kijun) / atr
        # ── Supertrend (10, 3): distancia en ATR al nivel y dirección
        a10 = feats._wilder(trf, 10)
        mid = hl2
        ub, lb = mid + 3.0 * a10, mid - 3.0 * a10
        fub, flb = ub.copy(), lb.copy()
        dirn = np.ones(n)
        for i in range(1, n):
            if not (np.isfinite(ub[i]) and np.isfinite(fub[i - 1])):
                fub[i], flb[i] = ub[i], lb[i]
                continue
            fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
            flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
            if dirn[i - 1] > 0:
                dirn[i] = -1.0 if c[i] < flb[i] else 1.0
            else:
                dirn[i] = 1.0 if c[i] > fub[i] else -1.0
        stl = np.where(dirn > 0, flb, fub)
        out['supertrend_dist'] = (c - stl) / atr
        out['supertrend_dir'] = np.where(np.isfinite(atr), dirn, np.nan)
        # ── calendario
        d64 = t.astype('datetime64[s]').astype('datetime64[D]')
        out['dow'] = ((d64.astype(np.int64) + 4) % 7).astype(np.float64)               # 0 = lunes
        mend = (d64.astype('datetime64[M]') + 1).astype('datetime64[D]')
        out['dtm'] = (mend - d64).astype(np.int64).astype(np.float64)                   # días naturales hasta fin de mes
    # compuesto: media de percentiles de agotamiento (cuanto más alto, más sobrevendido)
    return out


def composite(M: np.ndarray, names: List[str]) -> np.ndarray:
    """Puntuación de «agotamiento vendedor» = media de rangos (de 0 a 1) de osciladores de sobreventa, orientada para que más alto = más sobrevendido."""
    sign = {'stoch14': -1, 'stoch5': -1, 'cci20': -1, 'mfi14': -1, 'uo': -1, 'tsi': -1, 'laguerre_rsi': -1, 'fisher': -1, 'connors': -1, 'stochrsi': -1, 'pctb': -1, 'cmf20': -1}
    cols = [names.index(k) for k in sign]
    R = np.full((len(M), len(cols)), np.nan)
    for q, j in enumerate(cols):
        x = M[:, j]
        ok = np.isfinite(x)
        if ok.sum() > 10:
            r = np.full(len(x), np.nan)
            r[ok] = pd.Series(x[ok]).rank(pct=True).to_numpy()
            R[:, q] = sign[names[j]] * r
    return np.nanmean(R, axis=1) + 0.0 if np.isfinite(R).any() else np.full(len(M), np.nan)


def event_matrix(data: Dict, EV: Dict, rows: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Osciladores en la barra de la señal de cada candidato (`rows` indexa la tabla de eventos)."""
    sym, bi = EV['sym'][rows], EV['bi'][rows]
    names_all = EV['sym_names']
    M: np.ndarray | None = None
    cols: List[str] = []
    order = np.argsort(sym, kind='stable')
    sym_o = sym[order]
    cuts = np.flatnonzero(np.diff(sym_o)) + 1
    for grp in np.split(order, cuts):
        if not len(grp):
            continue
        F = feats.build(data[names_all[sym[grp[0]]]])
        B = battery(F)
        if M is None:
            cols = list(B)
            M = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
        i = bi[grp].astype(np.int64)
        for q, k in enumerate(cols):
            M[grp, q] = B[k][i]
    return (M if M is not None else np.zeros((len(rows), 0), np.float32)), cols


# ═════════════════════════════════════════════════════════════════════════
#  Estudio
# ═════════════════════════════════════════════════════════════════════════
def _slope(zc: np.ndarray, y: np.ndarray, inv: np.ndarray, mask: np.ndarray):
    """Pendiente de la rentabilidad sobre el rango (de -0,5 a 0,5) con error estándar por clústeres de día. zc ya vale 0 fuera de la máscara."""
    cnt = mask.sum()
    ym = float((y * mask).sum() / cnt)
    yc = (y - ym) * mask
    sxx = float((zc * zc).sum())
    beta = float((zc * yc).sum() / sxx)
    g = np.bincount(inv, weights=zc * (yc - beta * zc))
    se = math.sqrt(float((g * g).sum())) / sxx
    return beta, beta / (se + 1e-12)


def _rank_within_year(x: np.ndarray, year: np.ndarray, mask: np.ndarray) -> np.ndarray:
    zc = np.zeros(len(x))
    df = pd.DataFrame({'x': x, 'y': year})[mask]
    r = df.groupby('y')['x'].rank(pct=True, method='average').to_numpy() - 0.5
    zc[np.flatnonzero(mask)] = r
    # centrado dentro de la máscara
    zc[mask] -= zc[mask].mean()
    return zc


def run(uni: Dict, data: Dict, args, out_dir: str):
    import model as MD
    import pit
    import research as RS
    import setups as SU
    import simulate as SM
    import stops as ST
    uni2, data2, row_ok, info = pit.prepare(uni, data, args)
    EV = RS.build_events(data2, uni2, SU.SETUPS, SM.default_variants())
    rows, FL, names = RS.union_events(EV, +1)
    k = row_ok(EV, rows)
    rows, FL = rows[k], FL[k]
    last_day = int(EV['day'].max())
    day, sym, gid = EV['day'][rows], EV['sym'][rows], EV['grp'][rows]
    year = RS.year_of(day)
    print(f'\n######## OSCILADORES COMPLEJOS · {len(rows):,} candidatos · datos hasta {np.datetime64("1970-01-01") + np.timedelta64(last_day, "D")}', flush=True)

    exp: Dict = {}
    base = RS.build_plan(EV, rows, FL, names, ST.plan_of(4.0), last_day, spy=data2['SPY'], export=exp, verbose=False)
    allowed = exp['allowed']
    M, cols = event_matrix(data2, EV, rows)
    comp = composite(M, cols)
    cols = cols + ['composite']
    M = np.concatenate([M, comp[:, None].astype(np.float32)], 1)
    print(f'  {len(cols)} osciladores calculados (causales) · cobertura media {np.isfinite(M).mean() * 100:.0f}%', flush=True)

    vi = EV['variants'].index(ST.vname(4.0))
    pnl = EV['pnl'][rows, vi]
    atrp = EV['X'][rows][:, RS.FEATURES.index('atrp')]
    elig = np.isfinite(pnl) & (day <= last_day) & np.isfinite(atrp) & (atrp >= RS.MIN_ATR_PCT) & (atrp < RS.MAX_ATR_PCT) & np.isin(gid, allowed)
    print(f'  fondo del cribado: {int(elig.sum()):,} candidatos elegibles (ATR {RS.MIN_ATR_PCT}–{RS.MAX_ATR_PCT} %, grupos {[RS.GROUPS[g] for g in allowed]}) · media {pnl[elig].mean():+.2f}% · acierto {np.mean(pnl[elig] > 0) * 100:.1f}%', flush=True)
    d_e, y_e, yr_e = day[elig], pnl[elig], year[elig]
    _, inv = np.unique(d_e, return_inverse=True)
    win = (y_e > 0).astype(np.float64)

    Z, MK, res = [], [], {}
    for j, nm in enumerate(cols):
        x = M[elig, j].astype(np.float64)
        mk = np.isfinite(x)
        if mk.sum() < 2000 or np.nanstd(x[mk]) == 0:
            Z.append(None); MK.append(None)
            continue
        zc = _rank_within_year(x, yr_e, mk)
        Z.append(zc); MK.append(mk.astype(np.float64))
        beta, t = _slope(zc, y_e, inv, MK[-1])
        _, tw = _slope(zc, win, inv, MK[-1])
        pos = tot = 0
        for Y in sorted(set(yr_e)):
            my = (yr_e == Y) & mk
            if my.sum() >= 400:
                zy = zc * my
                zy[my] -= zy[my].mean()
                by = float((zy * (y_e - y_e[my].mean())).sum() / (zy * zy).sum())
                tot += 1
                pos += by > 0
        # extremo: quintil superior frente al inferior
        q = np.quantile(zc[mk], [0.2, 0.8])
        lo, hi = mk & (zc <= q[0]), mk & (zc >= q[1])
        res[nm] = {'beta': beta, 't': t, 'tw': tw, 'pos': int(pos), 'tot': int(tot), 'q_hi': float(y_e[hi].mean()), 'q_lo': float(y_e[lo].mean()),
                   'w_hi': float(win[hi].mean() * 100), 'w_lo': float(win[lo].mean() * 100)}

    # umbral del «mejor de N»: permutar los resultados dentro de cada día
    reps = 100
    rng = np.random.default_rng(11)
    o_day = np.argsort(d_e, kind='stable')
    mx = np.empty(reps)
    live = [j for j, zc in enumerate(Z) if zc is not None]
    for r in range(reps):
        o = np.lexsort((rng.random(len(d_e)), d_e))
        yp = np.empty(len(y_e)); yp[o_day] = y_e[o]
        mx[r] = max(abs(_slope(Z[j], yp, inv, MK[j])[1]) for j in live)
    thr95, thr99 = float(np.quantile(mx, 0.95)), float(np.quantile(mx, 0.99))
    print(f'\n  A. CRIBADO UNIVARIANTE (pendiente de la rentabilidad sobre el rango dentro de cada año; error estándar por clústeres de día)')
    print(f'     bajo el azar, el MEJOR |t| de {len(live)} osciladores es ≥ {thr95:.1f} el 5 % de las veces y ≥ {thr99:.1f} el 1 % ({reps} permutaciones dentro de cada día)')
    print('     ' + 'oscilador'.ljust(18) + 'pend.(pp)'.rjust(10) + 't'.rjust(7) + 't(acierto)'.rjust(11) + 'años +'.rjust(9) + 'Q5−Q1 media'.rjust(13) + 'Q5−Q1 acierto'.rjust(15))
    order = sorted(res, key=lambda k_: -abs(res[k_]['t']))
    for nm in order[:25]:
        r_ = res[nm]
        star = '★' if abs(r_['t']) >= thr95 else ' '
        print('     ' + (nm + star).ljust(18) + f"{r_['beta']:+9.2f}{r_['t']:+7.1f}{r_['tw']:+10.1f}{r_['pos']:6d}/{r_['tot']:<2d}{r_['q_hi'] - r_['q_lo']:+11.2f} pp{r_['w_hi'] - r_['w_lo']:+11.1f} pp", flush=True)
    sig = [nm for nm in order if abs(res[nm]['t']) >= thr95]
    print(f'     superan el umbral del mejor de {len(live)}: {len(sig)} ({", ".join(sig) if sig else "ninguno"}) · esperados por azar ≈ {0.05 * len(live):.1f} si se miran sueltos')

    # ── B. modelo ampliado
    print('\n  B. SISTEMA ENTERO REENTRENADO CON OSCILADORES (modelo, política y cartera recalculados; AUC fuera de muestra)')
    print('     ' + 'versión'.ljust(46) + 'AUC'.rjust(7) + 'n'.rjust(6) + 'acierto'.rjust(9) + 'media'.rjust(8) + 'PF'.rjust(6) + 'alfa vs S&P'.rjust(14) + '   cartera 10×10 %: anual · caída · Sharpe')
    import feats
    base_mf = list(MD.MODEL_FEATURES)
    base_fe = list(feats.FEATURES)
    newX = np.full((EV['X'].shape[0], M.shape[1]), np.nan, dtype=np.float32)
    newX[rows] = M
    EV2 = dict(EV)
    EV2['X'] = np.hstack([EV['X'], newX])
    # selección SOLO con datos hasta 2018 (dos o tres años): los mejores por |t| de entonces
    pre = yr_e <= 2018
    print(f'     selección con datos hasta 2018: {int(pre.sum()):,} candidatos elegibles', flush=True)
    pre_sel = []
    if pre.sum() > 800:
        _, inv_p = np.unique(d_e[pre], return_inverse=True)
        ts = {}
        for j in live:
            mk = MK[j][pre]
            if mk.sum() < 500:
                continue
            zp = Z[j][pre] * mk
            zp[mk > 0] -= zp[mk > 0].mean()
            ts[cols[j]] = abs(_slope(zp, y_e[pre], inv_p, mk)[1])
        pre_sel = [k_ for k_, _ in sorted(ts.items(), key=lambda kv: -kv[1])[:8]]
    full_sel = [k_ for k_ in order if k_ != 'composite'][:4]
    arms = [('modelo actual (28 variables)', []), ('+ todos los osciladores', [c for c in cols if c != 'composite']),
            ('+ compuesto de agotamiento (1 variable)', ['composite']), (f'+ 8 mejores con datos ≤2018: {",".join(pre_sel[:3])}…', pre_sel),
            (f'+ 4 mejores de la muestra completa (con ventaja): {",".join(full_sel[:2])}…', full_sel),
            ('+ los que superan el umbral del mejor de N', sig)]
    try:
        feats.FEATURES.extend(cols)
        for lab, extra in arms:
            if lab.startswith('+ los que superan') and not extra:
                print('     ' + lab.ljust(46) + '  (ninguno supera el umbral)')
                continue
            MD.MODEL_FEATURES = base_mf + list(extra)
            try:
                pl = _quiet(RS.build_plan, EV2, rows, FL, names, ST.plan_of(4.0), last_day, spy=data2['SPY'])
                o, vs, p10, ca = pl['stats']['oos'], pl['stats'].get('vsSpy', {}), pl['stats']['portfolio']['10x10'], pl['stats']['calibration']
                print('     ' + lab.ljust(46) + f"{ca['aucAll']:7.3f}{o['n']:6.0f}{o['wr']:8.1f}%{o['mean']:+7.2f}%{o['pf']:6.2f}{vs.get('alpha', float('nan')):+8.2f}% (t={vs.get('alphaT', float('nan')):.1f})   "
                      f"{p10['cagr']:+6.1f}% · {p10['dd']:4.1f}% · {p10['sharpe']:.2f}", flush=True)
            except Exception as exc:
                print('     ' + lab.ljust(46) + f'  sin plan ({type(exc).__name__}: {exc})', flush=True)
    finally:
        MD.MODEL_FEATURES = base_mf
        feats.FEATURES[:] = base_fe
    sys.stdout.flush()
