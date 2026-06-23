"""
signals.py — Clasificación de señales (classifySignal) + EMA.

Portado LITERALMENTE desde stocks.html (v2026.05.31-r49). Replica:
  · calcEMA            (HTML 2437-2448)
  · classifySignal     (HTML 1420-1675), incluyendo:
      - trendBias (EMA200)
      - indicadores de clasificación (RSI, accel, velas verdes)
      - ATR(14)
      - Range Oscillator (Zeiierman) con ATR de horizonte largo
      - ramas SHORT / LONG / RANGE_LONG / RANGE_SHORT
      - etiquetas (longClass/shortClass/rangeClass)
      - stop por MAE (number u objeto {edge, range})
      - rr, flags de diagnóstico, isNewCross

Los textos de 'scenario' (emojis) se conservan idénticos para fidelidad, aunque
el servidor podría omitirlos; mantenerlos asegura paridad 1:1 con el HTML.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union
import math
from jsmath import to_fixed

MIN_RISK_PCT = 1.5   # HTML 1417
MAX_RISK_PCT = 8.0   # HTML 1418


# ── calcEMA (HTML 2437-2448) ──────────────────────────────────────────────
def calc_ema(values: List[float], period: int) -> List[float]:
    res = [float('nan')] * len(values)
    if len(values) < period:
        return res
    k = 2 / (period + 1)
    s = 0.0
    for i in range(period):
        s += values[i]
    res[period - 1] = s / period
    for i in range(period, len(values)):
        res[i] = values[i] * k + res[i - 1] * (1 - k)
    return res


def _fmt1(x: float) -> str:
    """Replica JS .toFixed(1) para los textos de escenario."""
    return f"{x:.1f}"


# ── classifySignal (HTML 1420-1675) ───────────────────────────────────────
def classify_signal(close: float, vp: Dict[str, Any], df: List[Dict[str, Any]],
                     mae_stop_pct: Optional[Union[float, Dict[str, float]]] = None) -> Dict[str, Any]:
    poc = vp['pocPrice']; vah = vp['vahPrice']; val = vp['valPrice']
    ph = vp['priceHigh']; pl = vp['priceLow']; step = vp['priceStep']

    if df and len(df) < 15:
        return {'signal': 'NEUTRAL', 'label': 'Perfil insuficiente', 'stop': None,
                'target': None, 'target2': None, 'invalidation': None, 'rr': None,
                'scenario': 'Perfil en desarrollo con menos de 15 barras. VP estadísticamente infiable — esperar más datos.'}

    buf = step * 0.5
    va_w = vah - val

    # trendBias (EMA200)
    trend_bias = 0
    if df and len(df) >= 50:
        closes = [b['close'] for b in df]
        e200 = calc_ema(closes, min(200, len(closes)))
        ema200 = e200[-1]
        if not math.isnan(ema200):
            trend_bias = 1 if close > ema200 else -1

    in_va = val <= close <= vah

    # Indicadores de clasificación (RSI / accel / velas verdes)
    cls_rsi: Optional[float] = None
    cls_accel: Optional[float] = None
    cls_green = 0
    if df and len(df) >= 15:
        L = len(df)
        g = 0.0; l = 0.0
        for k in range(L - 14, L):
            if k < 1:
                continue
            ch = df[k]['close'] - df[k - 1]['close']
            if ch >= 0:
                g += ch
            else:
                l -= ch
        rs = (g / 14) / (l / 14) if l > 0 else 100
        cls_rsi = 100 - 100 / (1 + rs)
        c0 = df[L - 1]['close']
        c1 = df[L - 2]['close'] if L >= 2 else None
        c2 = df[L - 3]['close'] if L >= 3 else None
        if c1 is not None and c2 is not None:
            cls_accel = (c0 - c1) - (c1 - c2)
        k = L - 1
        while k >= max(1, L - 6):
            if df[k]['close'] > df[k - 1]['close']:
                cls_green += 1
            else:
                break
            k -= 1
    rsi_neutro = cls_rsi is not None and 40 <= cls_rsi <= 60
    frenando = cls_accel is not None and cls_accel > 0
    dos_verdes = cls_green >= 2

    # ATR(14)
    atr: Optional[float] = None
    if df and len(df) >= 15:
        period = 14
        sum_tr = 0.0; count = 0
        for i in range(len(df) - period, len(df)):
            if i < 1:
                continue
            tr = max(df[i]['high'] - df[i]['low'],
                     abs(df[i]['high'] - df[i - 1]['close']),
                     abs(df[i]['low'] - df[i - 1]['close']))
            sum_tr += tr; count += 1
        if count > 0:
            atr = sum_tr / count

    # Range Oscillator (Zeiierman)
    ro_break_up = False; ro_break_dn = False; ro_osc: Optional[float] = None
    if df and len(df) >= 50:
        ro_len = min(50, len(df) - 1)
        ro_atr_len = min(200, len(df) - 1)
        sum_tr_long = 0.0; cnt_tr = 0
        for k in range(len(df) - ro_atr_len, len(df)):
            if k < 1:
                continue
            tr = max(df[k]['high'] - df[k]['low'],
                     abs(df[k]['high'] - df[k - 1]['close']),
                     abs(df[k]['low'] - df[k - 1]['close']))
            sum_tr_long += tr; cnt_tr += 1
        atr_long = (sum_tr_long / cnt_tr) if cnt_tr > 0 else (atr or 0)
        range_atr = atr_long * 2.0
        sum_wc = 0.0; sum_w = 0.0
        for i in range(ro_len):
            c0 = df[len(df) - 1 - i]['close']
            c1 = df[len(df) - 2 - i]['close']
            if c1 is None or c1 == 0:
                continue
            w = abs(c0 - c1) / c1
            sum_wc += c0 * w; sum_w += w
        ro_ma = (sum_wc / sum_w) if sum_w != 0 else None
        if ro_ma is not None and range_atr > 0:
            ro_break_up = close > ro_ma + range_atr
            ro_break_dn = close < ro_ma - range_atr
            ro_osc = 100 * (close - ro_ma) / range_atr

    sig: Dict[str, Any] = {
        'signal': 'NEUTRAL', 'label': 'En PoC — esperar', 'stop': None, 'target': None,
        'target2': None, 'invalidation': None, 'rr': None, 'trendBias': trend_bias,
        'scenario': 'Precio en el PoC. Esperar ruptura del Value Area con volumen.'
    }

    def stop_from_mae(entry: float, is_long: bool, is_range: bool) -> Optional[float]:
        if mae_stop_pct is None:
            return None
        if isinstance(mae_stop_pct, (int, float)):
            pct = mae_stop_pct
        else:
            pct = mae_stop_pct.get('range') if is_range else mae_stop_pct.get('edge')
        if pct is None:
            return None
        d = entry * (pct / 100)
        return entry - d if is_long else entry + d

    if close > vah + buf and not ro_break_up:
        raw_stop = vah + va_w * 0.25
        min_stop = close * (1 + MIN_RISK_PCT / 100)
        max_stop = close * (1 + MAX_RISK_PCT / 100)
        stop = stop_from_mae(close, False, False)
        if stop is None:
            stop = min(max(raw_stop, min_stop), max_stop)
        dist_above_vah = (close - vah) / close * 100
        a_favor_baja = trend_bias <= 0
        short_seguro = dist_above_vah < 3 and (rsi_neutro or a_favor_baja)
        sig.update({'signal': 'SHORT', 'label': 'Short → PoC',
                    'stop': stop, 'target': poc, 'target2': val, 'invalidation': ph + step * 2,
                    'shortClass': 'seguro' if short_seguro else None,
                    'scenario': f"Precio sobre VAH ({_fmt1((close-vah)/vah*100)}% sobre el VA)."
                                + (' 🛡️ SHORT seguro: cerca del borde con giro/contexto bajista (mayor probabilidad).' if short_seguro else '')
                                + ' Sesgo bajista hacia el PoC.'
                                + (' ⚠️ Contra tendencia alcista (EMA200).' if trend_bias > 0 else '')})
    elif close < val - buf and not ro_break_dn:
        raw_stop = val - va_w * 0.25
        min_stop = close * (1 - MIN_RISK_PCT / 100)
        max_stop = close * (1 - MAX_RISK_PCT / 100)
        stop = stop_from_mae(close, True, False)
        if stop is None:
            stop = max(min(raw_stop, min_stop), max_stop)
        dist_below_val = (val - close) / close * 100
        dist_to_poc = (poc - close) / close * 100 if close else 0
        a_favor_tend = trend_bias >= 0
        deep = dist_below_val > 8
        giro = dos_verdes or frenando or cls_green >= 1   # giro confirmado (acción de precio)
        # DOS NIVELES DE PELOTAZO (validados en el laboratorio sobre el histórico):
        #  💎💎 pelotazo_max: caído + ≥2 verdes + a favor de tendencia → 70% WR, +29%/op
        #  💎   pelotazo:     caído + PoC lejos (>10%) + giro            → 62% WR, +21%/op
        # El caído SIN giro (cuchillo cayendo) ya NO es pelotazo.
        is_pelotazo_max = deep and dos_verdes and a_favor_tend
        is_pelotazo = deep and dist_to_poc > 10 and giro
        is_seguro = dist_below_val < 3 and (rsi_neutro or frenando)
        if is_pelotazo_max:
            long_class = 'pelotazo_max'
        elif is_pelotazo:
            long_class = 'pelotazo'
        elif is_seguro:
            long_class = 'seguro'
        else:
            long_class = None
        _scn_lbl = {
            'pelotazo_max': ' 💎💎 LONG pelotazo máximo: muy caído + giro confirmado (2 verdes a favor). La receta más potente (alto WR y PnL).',
            'pelotazo': ' 💎 LONG pelotazo: muy caído + gran recorrido al PoC + giro. Alto PnL/op con buena probabilidad.',
            'seguro': ' 🛡️ LONG seguro: cerca del VAL con giro confirmado (mayor probabilidad de acierto).',
        }.get(long_class, '')
        sig.update({'signal': 'LONG', 'label': 'Long → PoC',
                    'stop': stop, 'target': poc, 'target2': vah, 'invalidation': pl - step * 2,
                    'deepValueLong': deep, 'longClass': long_class,
                    'scenario': f"Precio bajo VAL ({_fmt1((val-close)/val*100)}% bajo el VA)."
                                + _scn_lbl
                                + ' Sesgo alcista hacia el PoC.'
                                + (' ⚠️ Contra tendencia bajista (EMA200).' if trend_bias < 0 else '')})
    elif in_va and close < poc:
        raw_stop = val - va_w * 0.15
        min_stop = close * (1 - MIN_RISK_PCT / 100)
        max_stop = close * (1 - MAX_RISK_PCT / 100)
        stop = stop_from_mae(close, True, True)
        if stop is None:
            stop = max(min(raw_stop, min_stop), max_stop)
        rl_premium = (cls_rsi is not None and cls_rsi > 70) or (frenando and trend_bias >= 0)
        sig.update({'signal': 'RANGE_LONG', 'label': 'Range Long → PoC',
                    'stop': stop, 'target': poc, 'target2': vah, 'invalidation': pl - step,
                    'rangeClass': 'premium' if rl_premium else None,
                    'scenario': 'Dentro del VA bajo el PoC.'
                                + (' ⭐ Range premium: momentum/empuje que favorece llegar al PoC (WR muy alto).' if rl_premium else '')
                                + ' Sesgo alcista hacia el PoC.'})
    elif in_va and close > poc:
        raw_stop = vah + va_w * 0.15
        min_stop = close * (1 + MIN_RISK_PCT / 100)
        max_stop = close * (1 + MAX_RISK_PCT / 100)
        stop = stop_from_mae(close, False, True)
        if stop is None:
            stop = min(max(raw_stop, min_stop), max_stop)
        rs_bueno = (cls_rsi is not None and cls_rsi < 30) or (trend_bias <= 0 and frenando)
        sig.update({'signal': 'RANGE_SHORT', 'label': 'Range Short → PoC',
                    'stop': stop, 'target': poc, 'target2': val, 'invalidation': ph + step,
                    'rangeClass': 'rebote' if rs_bueno else None,
                    'scenario': 'Dentro del VA sobre el PoC.'
                                + (' 🛡️ Range con contexto favorable (sobreventa/tendencia) — el resto de RANGE_SHORT pierde de media.' if rs_bueno else '')
                                + ' Sesgo bajista hacia el PoC.'})

    if sig['stop'] and sig['target']:
        risk = abs(close - sig['stop'])
        reward = abs(sig['target'] - close)
        sig['rr'] = to_fixed(reward / risk, 2) if risk > 0 else None

    sig['diagRoBreakUp'] = ro_break_up
    sig['diagRoBreakDn'] = ro_break_dn
    sig['diagRoOsc'] = ro_osc
    sig['diagTrendBias'] = trend_bias
    is_long_dir = sig['signal'] in ('LONG', 'RANGE_LONG')
    is_short_dir = sig['signal'] in ('SHORT', 'RANGE_SHORT')
    sig['diagTrendAgainst'] = (is_long_dir and trend_bias < 0) or (is_short_dir and trend_bias > 0)
    sig['clsRsi'] = cls_rsi; sig['clsAccel'] = cls_accel; sig['clsGreen'] = cls_green

    sig['isNewCross'] = True
    if df and len(df) >= 2 and sig['signal'] != 'NEUTRAL':
        prev_close = df[-2]['close']

        def zone_of(p):
            if p > vah + buf:
                return 'above'
            if p < val - buf:
                return 'below'
            if p > poc:
                return 'inHigh'
            if p < poc:
                return 'inLow'
            return 'poc'
        sig['isNewCross'] = zone_of(close) != zone_of(prev_close)

    # ════════════════════════════════════════════════════════════════════
    # FILTRO ÉLITE — solo los nichos validados en el laboratorio (1d).
    # isElite=True → la señal "cuenta" (se muestra, entra en histórico y stats).
    # eliteClass → etiqueta de élite. Lo que no es élite, para el sistema NO existe.
    # ════════════════════════════════════════════════════════════════════
    st = sig['signal']
    elite = False
    elite_class = None
    if st != 'NEUTRAL' and df and len(df) >= 2:
        # Velas ROJAS consecutivas (para señales bajistas), espejo de cls_green
        red = 0
        for k in range(len(df) - 1, max(0, len(df) - 7), -1):
            if k < 1:
                break
            if df[k]['close'] < df[k - 1]['close']:
                red += 1
            else:
                break
        confirm = cls_green if st in ('LONG', 'RANGE_LONG') else red
        c2 = confirm >= 2
        va_width_pct = (vah - val) / close * 100 if close else 0
        dist_to_poc = abs(poc - close) / close * 100 if close else 0
        a_favor = (trend_bias >= 0) if st in ('LONG', 'RANGE_LONG') else (trend_bias <= 0)

        if st == 'LONG':
            dist_below_val = (val - close) / close * 100
            deep = dist_below_val > 8
            # borde con volumen alto (suelo fuerte) en el VAL
            edge_vol = 0
            vol_by_level = vp.get('volByLevel'); total_vol = vp.get('totalVol', 0)
            val_level = vp.get('valLevel', 0); n_rows = vp.get('nRows', 25)
            if vol_by_level and total_vol and total_vol > 0:
                lo = max(0, val_level - 1); hi = min(n_rows - 1, val_level + 1)
                edge_vol = sum(vol_by_level[lo:hi+1]) / total_vol * 100
            if deep and c2 and a_favor:
                elite, elite_class = True, 'pelotazo_max'      # 74% WR, +32%/op
            elif deep and c2:
                elite, elite_class = True, 'pelotazo_max'      # 74% WR, +30%/op
            elif (dist_to_poc > 10 and c2) or (deep and edge_vol > 10):
                elite, elite_class = True, 'pelotazo'          # 65% WR, +18-22%/op
        elif st == 'RANGE_LONG':
            if a_favor and c2:
                elite, elite_class = True, 'premium'           # 79% WR, +4.6%/op
            else:
                extended = (close < val) or dist_to_poc > 10
                if extended and c2:
                    elite, elite_class = True, 'pelotazo'      # 58% WR, +11.5%/op (PnL)
        elif st == 'RANGE_SHORT':
            # SOLO a favor de tendencia + 2 velas a favor (rojas). Resto = veneno.
            if a_favor and c2:
                elite, elite_class = True, 'rebote'            # 77% WR, +2.2%/op
        elif st == 'SHORT':
            dist_above_vah = (close - vah) / close * 100
            recovered = False
            if len(df) >= 2:
                recovered = (df[-2]['close'] > vah) and (close <= vah * 1.01)
            if va_width_pct < 10 or recovered:
                elite, elite_class = True, 'seguro'            # 57-67% WR, +2.5%/op

    sig['isElite'] = elite
    sig['eliteClass'] = elite_class

    return sig
