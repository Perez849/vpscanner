"""
vp_core.py — Núcleo de cálculo Volume Profile.

Portado LITERALMENTE desde stocks.html (v2026.05.31-r49) para que el scanner
servidor (GitHub Action) produzca EXACTAMENTE los mismos números que la app.
Cada función lleva referencia a la línea del HTML que replica.

Convención de datos: una vela (bar) es un dict con claves
  {'ts': int(ms), 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
y `df` es una lista de bars ordenada cronológicamente (ascendente).
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any


# ── pivotHigh / pivotLow (HTML líneas 1296-1318) ──────────────────────────
def pivot_high(highs: List[float], length: int) -> List[Optional[float]]:
    n = len(highs)
    res: List[Optional[float]] = [None] * n
    for i in range(length, n - length):
        is_pivot = True
        for j in range(i - length, i + length + 1):
            if j != i and highs[j] >= highs[i]:
                is_pivot = False
                break
        if is_pivot:
            res[i] = highs[i]
    return res


def pivot_low(lows: List[float], length: int) -> List[Optional[float]]:
    n = len(lows)
    res: List[Optional[float]] = [None] * n
    for i in range(length, n - length):
        is_pivot = True
        for j in range(i - length, i + length + 1):
            if j != i and lows[j] <= lows[i]:
                is_pivot = False
                break
        if is_pivot:
            res[i] = lows[i]
    return res


# ── getPivotSegments (HTML líneas 1320-1341) ──────────────────────────────
def get_pivot_segments(df: List[Dict[str, Any]], length: int) -> List[Dict[str, Any]]:
    highs = [b['high'] for b in df]
    lows = [b['low'] for b in df]
    ph = pivot_high(highs, length)
    pl = pivot_low(lows, length)
    pivots = []
    for i in range(len(df)):
        if ph[i] is not None:
            pivots.append({'i': i, 'type': 'H', 'price': ph[i]})
        if pl[i] is not None:
            pivots.append({'i': i, 'type': 'L', 'price': pl[i]})
    pivots.sort(key=lambda p: p['i'])
    segs = []
    for k in range(1, len(pivots)):
        p0, p1 = pivots[k - 1], pivots[k]
        if p1['i'] - p0['i'] < 3:
            continue
        segs.append({
            'startI': p0['i'], 'endI': p1['i'],
            'pivotType': p1['type'], 'pivotPrice': p1['price'],
            'prevPrice': p0['price'],
            'slice': df[p0['i']:p1['i'] + 1],
        })
    return segs


# ── calcVP (HTML líneas 1346-1404) ────────────────────────────────────────
def calc_vp(bars: List[Dict[str, Any]], n_rows: int = 25, va_pct: float = 0.68) -> Optional[Dict[str, Any]]:
    if not bars or len(bars) < 3:
        return None
    ph = max(b['high'] for b in bars)
    pl = min(b['low'] for b in bars)
    if ph <= pl:
        return None
    step = (ph - pl) / n_rows
    if step <= 0:
        return None

    vols = [0.0] * n_rows
    total_vol = 0.0

    for bar in bars:
        bv = bar['volume'] if bar['volume'] and bar['volume'] > 0 else 0.0
        bh, bl = bar['high'], bar['low']
        br = bh - bl
        total_vol += bv
        for lvl in range(n_rows):
            ll = pl + lvl * step
            # Pine: barPriceHigh >= priceLevel and barPriceLow < priceLevel + priceStep
            if bh >= ll and bl < ll + step:
                # Pine asigna fracción FIJA priceStep/(high-low) a cada nivel tocado
                # (NO recorta por el solapamiento real con el nivel). Réplica exacta.
                frac = 1.0 if br == 0 else step / br
                vols[lvl] += bv * frac

    # PoC
    poc = 0
    for i in range(1, n_rows):
        if vols[i] > vols[poc]:
            poc = i

    # Value Area — réplica del while-loop Pine.
    # CLAVE: el Pine usa target = array.sum(volumeStorageT)*isValueArea, o sea sobre
    # la SUMA DE NIVELES, no el volumen crudo. Con el reparto Pine (step/rango) ambas
    # difieren, así que hay que usar la suma de niveles.
    sum_levels = 0.0
    for v in vols:
        sum_levels += v
    va_target = sum_levels * va_pct
    va, la, lb = vols[poc], poc, poc
    while va < va_target:
        if lb == 0 and la == n_rows - 1:
            break
        va_up = vols[la + 1] if la < n_rows - 1 else 0.0
        va_dn = vols[lb - 1] if lb > 0 else 0.0
        if va_up == 0 and va_dn == 0:
            break
        if va_up >= va_dn:
            va += va_up
            la += 1
        else:
            va += va_dn
            lb -= 1

    zero_vol_bars = sum(1 for b in bars if not (b['volume'] and b['volume'] > 0))
    zero_vol_pct = zero_vol_bars / len(bars)
    vol_reliable = total_vol > 0 and zero_vol_pct < 0.20

    return {
        'pocPrice': pl + (poc + 0.5) * step,
        'vahPrice': pl + (la + 1.0) * step,
        'valPrice': pl + (lb + 0.0) * step,
        'pocLevel': poc, 'vahLevel': la, 'valLevel': lb,
        'priceHigh': ph, 'priceLow': pl, 'priceStep': step,
        'volByLevel': vols, 'totalVol': total_vol, 'nRows': n_rows,
        'volReliable': vol_reliable, 'zeroVolPct': int(zero_vol_pct * 100 + 0.5),
    }


# ── developing_slice — perfil "en desarrollo" como el Pine ─────────────────
def developing_slice(df: List[Dict[str, Any]], length: int) -> List[Dict[str, Any]]:
    """
    Devuelve las barras del perfil EN DESARROLLO, igual que el Pine de dgtrd
    para el último perfil que se ve en pantalla.

    En el Pine: profileLength = last_bar_index - x2 + pvtLength, donde x2 es el
    bar_index del último pivote confirmado. El perfil va desde ese pivote hasta
    la última barra. Un pivote en posición i se confirma 'length' barras después,
    así que el último pivote confirmado está en posición <= len(df)-1-length.
    """
    highs = [b['high'] for b in df]
    lows = [b['low'] for b in df]
    ph = pivot_high(highs, length)
    pl = pivot_low(lows, length)
    last_pivot_i = None
    # último índice con pivote confirmado (tiene 'length' barras a su derecha)
    for i in range(len(df) - 1 - length, -1, -1):
        if ph[i] is not None or pl[i] is not None:
            last_pivot_i = i
            break
    if last_pivot_i is None:
        return df[:]  # sin pivote: usa todo (caso degenerado)
    # Pine: profileLength = (last_bar_index - x2) + pvtLength ; usa highest/lowest
    # sobre profileLength+1 barras hasta el final. Eso equivale a empezar el tramo
    # en  len(df)-1 - profileLength  =  x2 - pvtLength.
    profile_len = ((len(df) - 1) - last_pivot_i) + length
    start = max(0, (len(df) - 1) - profile_len)
    return df[start:]