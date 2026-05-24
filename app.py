# VP Scanner · Volume Profile Pivot Anchored
# github.com — archivo único
# ─────────────────────────────────────────

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
import time

TOP_SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
    "DOGE/USDT","ADA/USDT","AVAX/USDT","TRX/USDT","TON/USDT",
    "LINK/USDT","DOT/USDT","LTC/USDT","BCH/USDT","NEAR/USDT",
    "UNI/USDT","ARB/USDT","OP/USDT","ATOM/USDT","INJ/USDT",
    "AAVE/USDT","STX/USDT","FIL/USDT","HBAR/USDT","VET/USDT",
    "MKR/USDT","IMX/USDT","GRT/USDT","XLM/USDT","ALGO/USDT",
    "EOS/USDT","FTM/USDT","SAND/USDT","MANA/USDT","AXS/USDT",
    "ICP/USDT","APT/USDT","EGLD/USDT","MATIC/USDT","SHIB/USDT",
]

TIMEFRAMES = ["1d", "4h"]

# Yahoo Finance ticker mapping: symbol → YF ticker
# YF uses BASE-USD for crypto (BTC-USD, ETH-USD, etc.)
_YF_MAP = {s.split("/")[0]: s.split("/")[0] + "-USD" for s in TOP_SYMBOLS}
# Exceptions where YF uses a different ticker
_YF_MAP.update({"BNB":"BNB-USD","SHIB":"SHIB-USD","MATIC":"MATIC-USD","FTM":"FTM-USD"})

# YF interval/period mapping
_YF_PARAMS = {
    "1d": {"interval": "1d", "period": "2y"},
    "4h": {"interval": "1h", "period": "60d"},   # YF doesn't have 4h; use 1h and resample
}


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame | None:
    """
    Fetch OHLCV via yfinance — works from any server, no geo restrictions.
    For 4H: downloads 1H data and resamples to 4H bars.
    """
    base = symbol.split("/")[0].upper()
    ticker = _YF_MAP.get(base, base + "-USD")
    params = _YF_PARAMS.get(timeframe, _YF_PARAMS["1d"])

    try:
        raw = yf.download(
            ticker,
            interval=params["interval"],
            period=params["period"],
            progress=False,
            auto_adjust=True,
        )
        if raw is None or len(raw) < 20:
            return None

        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index().astype(float)
        df = df[df["volume"] > 0].dropna()

        # Resample 1H → 4H for the 4h timeframe
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna()

        df = df.tail(limit)
        return df if len(df) >= 30 else None

    except Exception as e:
        print(f"[fetcher] Error {symbol} {timeframe}: {e}")
        return None


def pivot_high(high: np.ndarray, length: int) -> np.ndarray:
    """
    Returns array where non-nan values are pivot highs.
    Equivalent to ta.pivothigh(length, length) in Pine Script.
    A pivot high at index i means high[i] is the highest in the window
    [i-length, i+length] (inclusive).
    """
    n = len(high)
    result = np.full(n, np.nan)
    for i in range(length, n - length):
        window = high[i - length: i + length + 1]
        if high[i] == np.max(window):
            result[i] = high[i]
    return result


def pivot_low(low: np.ndarray, length: int) -> np.ndarray:
    """
    Returns array where non-nan values are pivot lows.
    """
    n = len(low)
    result = np.full(n, np.nan)
    for i in range(length, n - length):
        window = low[i - length: i + length + 1]
        if low[i] == np.min(window):
            result[i] = low[i]
    return result


def get_pivots(df: pd.DataFrame, length: int = 10) -> pd.DataFrame:
    """
    Adds pivot_high and pivot_low columns to df.
    Uses length=10 for daily (equivalent to Pine's 20 with left+right).
    """
    df = df.copy()
    df["pivot_high"] = pivot_high(df["high"].values, length)
    df["pivot_low"] = pivot_low(df["low"].values, length)
    return df


def get_pivot_segments(df: pd.DataFrame, length: int = 10) -> list[dict]:
    """
    Returns list of segments between consecutive pivots.
    Each segment: {
        'start_idx': int,
        'end_idx': int,
        'start_bar': int,   # absolute bar index in df
        'end_bar': int,
        'pivot_type': 'H' or 'L',
        'pivot_price': float,
        'df_slice': DataFrame
    }
    """
    df = get_pivots(df, length)
    pivots = []

    for i, row in df.iterrows():
        if not np.isnan(row["pivot_high"]):
            pivots.append({"idx": df.index.get_loc(i), "ts": i, "type": "H", "price": row["pivot_high"]})
        if not np.isnan(row["pivot_low"]):
            pivots.append({"idx": df.index.get_loc(i), "ts": i, "type": "L", "price": row["pivot_low"]})

    # Sort by bar index, deduplicate same-bar pivots keeping highest priority
    # If H and L fall on the same bar, keep both but they will create a 0-length
    # segment — the filter end - start < 3 handles it, but let's be explicit
    pivots = sorted(pivots, key=lambda x: x["idx"])

    if len(pivots) < 2:
        return []

    segments = []
    for i in range(1, len(pivots)):
        p_prev = pivots[i - 1]
        p_curr = pivots[i]
        start = p_prev["idx"]
        end = p_curr["idx"]
        if end - start < 3:
            continue
        segments.append({
            "start_idx": start,
            "end_idx": end,
            "start_ts": p_prev["ts"],
            "end_ts": p_curr["ts"],
            "pivot_type": p_curr["type"],
            "pivot_price": p_curr["price"],
            "prev_pivot_price": p_prev["price"],
            "prev_pivot_type": p_prev["type"],
            "df_slice": df.iloc[start: end + 1],
        })

    return segments



def calculate_volume_profile(
    df_slice: pd.DataFrame,
    n_rows: int = 25,
    value_area_pct: float = 0.68,
) -> dict | None:
    """
    Exact replication of Pine Script volume profile algorithm.

    Returns dict with:
        poc_price       : float
        vah_price       : float
        val_price       : float
        poc_level       : int
        vah_level       : int  (levelAbovePoc)
        val_level       : int  (levelBelowPoc)
        price_high      : float
        price_low       : float
        price_step      : float
        volume_by_level : np.ndarray  (length = n_rows)
        total_volume    : float
        n_bars          : int
    """
    if df_slice is None or len(df_slice) < 3:
        return None

    price_high = df_slice["high"].max()
    price_low = df_slice["low"].min()

    if price_high <= price_low:
        return None

    price_step = (price_high - price_low) / n_rows
    if price_step <= 0:
        return None

    volume_storage = np.zeros(n_rows)
    total_volume = 0.0

    # --- Pine Script inner loop replication ---
    for _, bar in df_slice.iterrows():
        bar_vol = bar["volume"] if bar["volume"] > 0 else 0
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_range = bar_high - bar_low

        total_volume += bar_vol

        for level in range(n_rows):
            level_low = price_low + level * price_step
            level_high = level_low + price_step

            # Pine condition: barHigh >= priceLevel and barLow < priceLevel + priceStep
            if bar_high >= level_low and bar_low < level_high:
                if bar_range == 0:
                    proportion = 1.0
                else:
                    # overlap between bar and level
                    overlap_low = max(bar_low, level_low)
                    overlap_high = min(bar_high, level_high)
                    proportion = max(0, overlap_high - overlap_low) / bar_range
                volume_storage[level] += bar_vol * proportion

    # --- PoC: level with max volume ---
    poc_level = int(np.argmax(volume_storage))

    # --- Value Area: Pine Script while-loop ---
    va_target = total_volume * value_area_pct
    value_area = volume_storage[poc_level]
    level_above_poc = poc_level
    level_below_poc = poc_level

    while value_area < va_target:
        if level_below_poc == 0 and level_above_poc == n_rows - 1:
            break

        vol_above = volume_storage[level_above_poc + 1] if level_above_poc < n_rows - 1 else 0.0
        vol_below = volume_storage[level_below_poc - 1] if level_below_poc > 0 else 0.0

        if vol_above == 0 and vol_below == 0:
            break

        if vol_above >= vol_below:
            value_area += vol_above
            level_above_poc += 1
        else:
            value_area += vol_below
            level_below_poc -= 1

    # --- Price levels (Pine: poc at +0.50, vah at +1.00, val at +0.00) ---
    poc_price = price_low + (poc_level + 0.50) * price_step
    vah_price = price_low + (level_above_poc + 1.00) * price_step
    val_price = price_low + (level_below_poc + 0.00) * price_step

    return {
        "poc_price": poc_price,
        "vah_price": vah_price,
        "val_price": val_price,
        "poc_level": poc_level,
        "vah_level": level_above_poc,
        "val_level": level_below_poc,
        "price_high": price_high,
        "price_low": price_low,
        "price_step": price_step,
        "volume_by_level": volume_storage,
        "total_volume": total_volume,
        "n_bars": len(df_slice),
        "n_rows": n_rows,
    }


def get_developing_profile(df: pd.DataFrame, last_pivot_idx: int, n_rows: int = 25, value_area_pct: float = 0.68) -> dict | None:
    """
    Calculates the 'developing' (current, unfinished) volume profile
    from last pivot to present bar — replicates Pine's barstate.islast block.
    """
    if last_pivot_idx >= len(df) - 1:
        return None
    df_slice = df.iloc[last_pivot_idx:]
    return calculate_volume_profile(df_slice, n_rows, value_area_pct)



# ─── Signal Classification ─────────────────────────────────────────────────────

def classify_signal(close: float, vp: dict) -> dict:
    """
    Determines trade setup based on price position relative to Value Area.

    Level hierarchy (LONG):   close > stop > invalidation
    Level hierarchy (SHORT):  close < stop < invalidation

    Stop      = technical level where the trade is in danger (near structure)
    Target    = logical destination given the VP setup
    Target 2  = extended destination if momentum continues
    Invalidation = level where the entire VP analysis is broken (beyond profile extremes)
                   Always further from price than stop. Never between price and stop.

    Returns signal dict with:
        signal        : 'LONG' | 'SHORT' | 'RANGE_LONG' | 'RANGE_SHORT' | 'NEUTRAL'
        signal_label  : human-readable string
        stop          : float
        target        : float
        target2       : float (secondary target)
        invalidation  : float
        rr            : float (risk/reward ratio)
        scenario      : str (description of the setup)
        in_value_area : bool
    """
    poc   = vp["poc_price"]
    vah   = vp["vah_price"]
    val   = vp["val_price"]
    ph    = vp["price_high"]   # top of entire profile
    pl    = vp["price_low"]    # bottom of entire profile
    step  = vp["price_step"]
    buffer = step * 0.5        # half-row buffer

    in_va    = val <= close <= vah
    va_width = vah - val
    pr_range = ph - pl         # full profile range

    result = {
        "signal": "NEUTRAL",
        "signal_label": "Sin señal clara",
        "stop": None,
        "target": None,
        "target2": None,
        "invalidation": None,
        "rr": None,
        "scenario": "",
        "in_value_area": in_va,
    }

    # ── Scenario A: Price ABOVE VA → SHORT toward PoC ─────────────────────────
    # Hierarchy: close < stop < invalidation
    #   stop         = just above VAH (market re-accepts VA = trade wrong)
    #   invalidation = above profile high (entire structure broken)
    if close > vah + buffer:
        stop         = vah + va_width * 0.20          # ~20% VA above VAH
        invalidation = ph + step * 2                  # above profile top, always > stop
        result.update({
            "signal": "SHORT",
            "signal_label": "SHORT → PoC",
            "stop": round(stop, 8),
            "target": round(poc, 8),
            "target2": round(val, 8),
            "invalidation": round(invalidation, 8),
            "scenario": (
                "Precio sobre VAH. Sesgo bajista hacia el PoC. "
                "Stop por encima del VAH — si el mercado reacepta el VA, la tesis se invalida. "
                "Invalidación por encima del máximo del perfil completo."
            ),
        })

    # ── Scenario B: Price BELOW VA → LONG toward PoC ──────────────────────────
    # Hierarchy: close > stop > invalidation
    #   stop         = just below VAL (market rejects support = trade wrong)
    #   invalidation = below profile low (entire structure broken)
    elif close < val - buffer:
        stop         = val - va_width * 0.20          # ~20% VA below VAL
        invalidation = pl - step * 2                  # below profile bottom, always < stop
        result.update({
            "signal": "LONG",
            "signal_label": "LONG → PoC",
            "stop": round(stop, 8),
            "target": round(poc, 8),
            "target2": round(vah, 8),
            "invalidation": round(invalidation, 8),
            "scenario": (
                "Precio bajo VAL. Sesgo alcista hacia el PoC. "
                "Stop por debajo del VAL — si el mercado rompe el soporte, la tesis se invalida. "
                "Invalidación por debajo del mínimo del perfil completo."
            ),
        })

    # ── Scenario C: Inside VA, BELOW PoC → RANGE LONG toward PoC / VAH ───────
    # Hierarchy: close > stop > invalidation
    #   stop         = below VAL (exits value area = no longer a range trade)
    #   invalidation = below profile low (structure fully broken)
    elif in_va and close < poc:
        stop         = val - buffer                   # below VAL boundary
        invalidation = pl - step                      # below profile low (always < stop since pl < val)
        result.update({
            "signal": "RANGE_LONG",
            "signal_label": "RANGE LONG → VAH",
            "stop": round(stop, 8),
            "target": round(poc, 8),
            "target2": round(vah, 8),
            "invalidation": round(invalidation, 8),
            "scenario": (
                "Dentro del VA por debajo del PoC. Rango definido. "
                "Stop bajo el VAL — si el precio abandona el Value Area, el rango se rompe. "
                "Invalidación en el mínimo del perfil."
            ),
        })

    # ── Scenario D: Inside VA, ABOVE PoC → RANGE SHORT toward PoC / VAL ──────
    # Hierarchy: close < stop < invalidation
    #   stop         = above VAH (exits value area = no longer a range trade)
    #   invalidation = above profile high (structure fully broken)
    elif in_va and close > poc:
        stop         = vah + buffer                   # above VAH boundary
        invalidation = ph + step                      # above profile high (always > stop since ph > vah)
        result.update({
            "signal": "RANGE_SHORT",
            "signal_label": "RANGE SHORT → VAL",
            "stop": round(stop, 8),
            "target": round(poc, 8),
            "target2": round(val, 8),
            "invalidation": round(invalidation, 8),
            "scenario": (
                "Dentro del VA por encima del PoC. Rango definido. "
                "Stop sobre el VAH — si el precio abandona el Value Area al alza, el rango se rompe. "
                "Invalidación en el máximo del perfil."
            ),
        })

    # ── Scenario E: At PoC ────────────────────────────────────────────────────
    else:
        result.update({
            "signal": "NEUTRAL",
            "signal_label": "En PoC — esperar",
            "scenario": (
                "Precio en el PoC. Zona de máximo equilibrio. "
                "Esperar ruptura confirmada del VA con volumen antes de posicionarse."
            ),
        })

    # ── Sanity check: enforce hierarchy ──────────────────────────────────────
    # For LONG/RANGE_LONG: price > stop > invalidation
    # For SHORT/RANGE_SHORT: price < stop < invalidation
    sig = result["signal"]
    s   = result["stop"]
    inv = result["invalidation"]
    if s is not None and inv is not None:
        if sig in ("LONG", "RANGE_LONG"):
            # stop must be below price, invalidation must be below stop
            if s >= close:
                result["stop"] = round(close * 0.99, 8)
                s = result["stop"]
            if inv >= s:
                result["invalidation"] = round(s * 0.99, 8)
        elif sig in ("SHORT", "RANGE_SHORT"):
            # stop must be above price, invalidation must be above stop
            if s <= close:
                result["stop"] = round(close * 1.01, 8)
                s = result["stop"]
            if inv <= s:
                result["invalidation"] = round(s * 1.01, 8)

    # ── Calculate R:R ─────────────────────────────────────────────────────────
    if result["stop"] and result["target"]:
        risk   = abs(close - result["stop"])
        reward = abs(result["target"] - close)
        result["rr"] = round(reward / risk, 2) if risk > 0 else None

    return result


# ─── Confluences ───────────────────────────────────────────────────────────────

def calculate_confluences(df: pd.DataFrame, vp: dict, close: float) -> dict:
    """
    Calculates confluence factors that strengthen or weaken the signal.
    Returns score 0-10 and list of active confluences.
    """
    confluences = []
    score = 0

    poc = vp["poc_price"]
    vah = vp["vah_price"]
    val = vp["val_price"]
    step = vp["price_step"]

    closes = df["close"].values
    volumes = df["volume"].values
    highs = df["high"].values
    lows = df["low"].values

    n = len(closes)
    if n < 50:
        return {"score": 0, "items": [], "max_score": 10}

    # Determine signal direction — covers both outer and inner VA positions
    # _long_bias: price below PoC (LONG from below VA, or RANGE_LONG inside VA below PoC)
    # _short_bias: price above PoC (SHORT from above VA, or RANGE_SHORT inside VA above PoC)
    _long_bias  = close <= poc   # price at or below PoC → bullish setups
    _short_bias = close >= poc   # price at or above PoC → bearish setups
    # Note: at exactly PoC both are True — that's NEUTRAL, confluences still calculated

    # ── 1. EMA21 position ─────────────────────────────────────────────────────
    ema21_arr = _ema(closes, 21)
    ema50_arr = _ema(closes, 50) if n >= 50 else np.full(n, np.nan)
    ema21 = float(ema21_arr[-1]) if not np.isnan(ema21_arr[-1]) else None
    ema50 = float(ema50_arr[-1]) if not np.isnan(ema50_arr[-1]) else None

    if ema21 is not None:
        if abs(ema21 - poc) / poc < 0.015:
            # EMA21 coinciding with PoC: strongest EMA confluence, always valid
            confluences.append({"label": "EMA21 ≈ PoC", "icon": "🎯", "strength": "alta"})
            score += 2
        elif close < ema21 and _long_bias:
            # Price below EMA21 while in long setup: confirms downward momentum may be exhausted
            confluences.append({"label": "Precio bajo EMA21 — momentum bajista agotado", "icon": "📉", "strength": "media"})
            score += 1
        elif close > ema21 and _short_bias:
            # Price above EMA21 while in short setup: confirms upward momentum may be exhausted
            confluences.append({"label": "Precio sobre EMA21 — momentum alcista agotado", "icon": "📈", "strength": "media"})
            score += 1

    # ── 2. EMA50 position ─────────────────────────────────────────────────────
    if ema50 is not None:
        if abs(ema50 - val) / val < 0.015 or abs(ema50 - vah) / vah < 0.015:
            confluences.append({"label": "EMA50 ≈ VAH/VAL", "icon": "🎯", "strength": "alta"})
            score += 2
        elif ema21 is not None and ema21 > ema50 and _long_bias:
            confluences.append({"label": "EMA21 > EMA50 (tendencia alcista)", "icon": "🟢", "strength": "media"})
            score += 1
        elif ema21 is not None and ema21 < ema50 and _short_bias:
            confluences.append({"label": "EMA21 < EMA50 (tendencia bajista)", "icon": "🔻", "strength": "media"})
            score += 1

    # ── 3. RSI ─────────────────────────────────────────────────────────────────
    rsi = _rsi(closes, 14)
    if rsi is not None:
        if rsi < 35 and _long_bias:
            confluences.append({"label": f"RSI sobrevendido ({rsi:.0f}) — agotamiento bajista", "icon": "🔥", "strength": "alta"})
            score += 2
        elif rsi > 65 and _short_bias:
            confluences.append({"label": f"RSI sobrecomprado ({rsi:.0f}) — agotamiento alcista", "icon": "❄️", "strength": "alta"})
            score += 2
        elif rsi < 35 and _short_bias and not _long_bias:
            confluences.append({"label": f"RSI sobrevendido ({rsi:.0f}) — contradice sesgo SHORT", "icon": "⚠️", "strength": "baja"})
        elif rsi > 65 and _long_bias and not _short_bias:
            confluences.append({"label": f"RSI sobrecomprado ({rsi:.0f}) — contradice sesgo LONG", "icon": "⚠️", "strength": "baja"})

    # ── 4. Volume on last bar + candle type classification ────────────────────
    vol_sma    = np.mean(volumes[-89:]) if n >= 89 else np.mean(volumes)
    last_vol   = volumes[-1]
    last_open  = df["open"].values[-1]
    last_close = df["close"].values[-1]
    last_bull  = last_close >= last_open

    vol_ratio = last_vol / vol_sma if vol_sma > 0 else 1.0
    high_vol  = vol_ratio > 1.618
    low_vol   = vol_ratio < 0.618

    if high_vol and last_bull and not _short_bias:
        # Dark green — institutional buying, best LONG confirmation
        confluences.append({"label": f"🟩 Vela verde oscuro — compra institucional (×{vol_ratio:.1f} SMA)", "icon": "💥", "strength": "alta"})
        score += 2
    elif high_vol and not last_bull and not _long_bias:
        # Dark red — institutional selling, best SHORT confirmation
        confluences.append({"label": f"🟥 Vela rojo oscuro — venta institucional (×{vol_ratio:.1f} SMA)", "icon": "💥", "strength": "alta"})
        score += 2
    elif high_vol and last_bull and _short_bias:
        # High vol bull but we're looking for short — warning
        confluences.append({"label": f"⚠️ Vela verde oscuro contradice sesgo SHORT (×{vol_ratio:.1f} SMA)", "icon": "⚠️", "strength": "baja"})
    elif high_vol and not last_bull and _long_bias:
        # High vol bear but we're looking for long — warning
        confluences.append({"label": f"⚠️ Vela rojo oscuro contradice sesgo LONG (×{vol_ratio:.1f} SMA)", "icon": "⚠️", "strength": "baja"})
    elif low_vol and last_bull:
        # Aqua — low conviction bullish, flag only
        confluences.append({"label": f"🔵 Vela aqua — movimiento alcista sin convicción (×{vol_ratio:.1f} SMA)", "icon": "🌫️", "strength": "baja"})
    elif low_vol and not last_bull:
        # Amber — low conviction bearish, can mean seller exhaustion in LONG setup
        if not _short_bias:
            confluences.append({"label": f"🟡 Vela ámbar — vendedores sin convicción, posible agotamiento (×{vol_ratio:.1f} SMA)", "icon": "🌫️", "strength": "media"})
        else:
            confluences.append({"label": f"🟡 Vela ámbar — movimiento bajista sin convicción (×{vol_ratio:.1f} SMA)", "icon": "🌫️", "strength": "baja"})
    else:
        # Normal volume — informational only
        direction = "alcista" if last_bull else "bajista"
        confluences.append({"label": f"Vela {direction} de volumen normal (×{vol_ratio:.1f} SMA)", "icon": "📊", "strength": "baja"})

    # ── 5. Price distance to PoC ───────────────────────────────────────────────
    dist_pct = abs(close - poc) / poc * 100
    if dist_pct < 1.5:
        confluences.append({"label": f"Precio muy cerca del PoC ({dist_pct:.1f}%) — imán activo", "icon": "🧲", "strength": "alta"})
        score += 2
    elif dist_pct > 8:
        # Far from PoC: informational, no score bonus but no penalty either
        confluences.append({"label": f"Precio lejos del PoC ({dist_pct:.1f}%) — recorrido amplio", "icon": "↔️", "strength": "baja"})

    # ── 6. VA width ────────────────────────────────────────────────────────────
    va_width_pct = (vah - val) / poc * 100
    if va_width_pct < 5:
        confluences.append({"label": f"VA estrecho ({va_width_pct:.1f}%) — consolidación previa a movimiento", "icon": "🗜️", "strength": "media"})
        score += 1

    # ── Determine candle type label for display ───────────────────────────────
    if high_vol and last_bull:
        candle_type = "verde_oscuro"
    elif high_vol and not last_bull:
        candle_type = "rojo_oscuro"
    elif low_vol and last_bull:
        candle_type = "aqua"
    elif low_vol and not last_bull:
        candle_type = "ambar"
    elif last_bull:
        candle_type = "verde_normal"
    else:
        candle_type = "rojo_normal"

    return {
        "score": min(score, 10),
        "max_score": 10,
        "items": confluences,
        "rsi": rsi,
        "ema21": ema21,
        "ema50": ema50,
        "vol_ratio": round(vol_ratio, 2),
        "candle_type": candle_type,
        "candle_bull": last_bull,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result
    k = 2 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _rsi(values: np.ndarray, period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    deltas = np.diff(values[-(period + 10):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)



PIVOT_LENGTH = 20  # matches Pine Script pvtLength=20 exactly


def scan_symbol(symbol: str, df_1d: pd.DataFrame | None, df_4h: pd.DataFrame | None,
                pivot_length: int = PIVOT_LENGTH) -> dict:
    """
    Full analysis for one symbol across both timeframes.
    Returns structured result dict.
    """
    result = {
        "symbol": symbol,
        "close": None,
        "1d": None,
        "4h": None,
        "error": None,
    }

    for tf, df in [("1d", df_1d), ("4h", df_4h)]:
        if df is None or len(df) < 60:
            continue

        close = float(df["close"].iloc[-1])
        result["close"] = close  # always update to latest available TF

        try:
            # 1. Get pivot segments using configured length
            segments = get_pivot_segments(df, length=pivot_length)
            if not segments:
                continue

            # 2. Developing profile (last pivot to now)
            last_seg = segments[-1]
            last_pivot_idx = last_seg["end_idx"]
            vp = get_developing_profile(df, last_pivot_idx, n_rows=25, value_area_pct=0.68)

            if not vp:
                # fallback: use last completed segment
                vp = calculate_volume_profile(last_seg["df_slice"], n_rows=25, value_area_pct=0.68)

            if not vp:
                continue

            # 3. Compute historical segment VPs for chart overlay
            hist_segments = []
            for seg in segments[-8:]:
                seg_vp = calculate_volume_profile(seg["df_slice"], n_rows=25, value_area_pct=0.68)
                hist_segments.append({**seg, "vp": seg_vp})

            # 4. Signal classification
            signal = classify_signal(close, vp)

            # 5. Confluences
            confluences = calculate_confluences(df, vp, close)

            # 6. Pivot % change for label
            last_pct = None
            if len(segments) >= 2:
                pp = segments[-2]["pivot_price"]
                cp = segments[-1]["pivot_price"]
                if pp and pp > 0:
                    last_pct = round((cp - pp) / pp * 100, 2)

            result[tf] = {
                "df": df,
                "vp": vp,
                "signal": signal,
                "confluences": confluences,
                "segments": hist_segments,
                "last_pct": last_pct,
                "close": close,
                "n_bars_in_profile": vp.get("n_bars", 0),
            }

        except Exception as e:
            print(f"[scanner] {symbol} {tf} error: {e}")
            import traceback; traceback.print_exc()
            continue

    return result


def run_full_scan(symbols: list[str] = TOP_SYMBOLS, progress_cb=None,
                  pivot_length: int = PIVOT_LENGTH) -> list[dict]:
    """
    Runs full scan for all symbols.
    progress_cb: optional callback(i, total, symbol) for progress bar.
    pivot_length: bars each side for pivot detection (default 20 = Pine Script pvtLength=20)
    """
    results = []
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if progress_cb:
            progress_cb(i, total, symbol)

        df_1d = fetch_ohlcv(symbol, "1d", limit=500)
        df_4h = fetch_ohlcv(symbol, "4h", limit=500)
        time.sleep(0.08)

        r = scan_symbol(symbol, df_1d, df_4h, pivot_length=pivot_length)
        results.append(r)

    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Flattens scan results into a DataFrame for the main table display.
    """
    rows = []
    for r in results:
        symbol = r["symbol"]
        close = r.get("close")

        for tf in ["1d", "4h"]:
            tf_data = r.get(tf)
            if not tf_data:
                continue

            vp = tf_data["vp"]
            sig = tf_data["signal"]
            conf = tf_data["confluences"]

            rows.append({
                "symbol": symbol,
                "tf": tf.upper(),
                "close": close,
                "poc": round(vp["poc_price"], 4),
                "vah": round(vp["vah_price"], 4),
                "val": round(vp["val_price"], 4),
                "dist_poc_pct": round(abs(close - vp["poc_price"]) / vp["poc_price"] * 100, 2) if close else None,
                "in_va": sig.get("in_value_area"),
                "signal": sig.get("signal"),
                "signal_label": sig.get("signal_label"),
                "stop": round(sig["stop"], 4) if sig.get("stop") else None,
                "target": round(sig["target"], 4) if sig.get("target") else None,
                "target2": round(sig["target2"], 4) if sig.get("target2") else None,
                "invalidation": round(sig["invalidation"], 4) if sig.get("invalidation") else None,
                "rr": sig.get("rr"),
                "score": conf.get("score", 0),
                "rsi": conf.get("rsi"),
                "vol_ratio": conf.get("vol_ratio"),
                "scenario": sig.get("scenario"),
            })

    return pd.DataFrame(rows)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Color palette ────────────────────────────────────────────────────────────
COL_POC       = "#FF3B3B"
COL_VAH       = "#2979FF"
COL_VAL       = "#2979FF"
COL_VA_FILL   = "rgba(41,121,255,0.08)"
COL_VP_INSIDE = "rgba(251,192,45,0.65)"   # inside VA bars
COL_VP_OUTSIDE= "rgba(67,70,81,0.65)"     # outside VA bars
COL_BG        = "#0D1117"
COL_GRID      = "#1C2128"
COL_EMA21     = "#FFD600"
COL_EMA50     = "#FF6D00"
COL_SIGNAL_LONG  = "#00E676"
COL_SIGNAL_SHORT = "#FF1744"

# ─── Volume-weighted candle colors (replicates Pine Script barcolor logic) ────
# Bull + high vol  → dark green   (institucional comprando)
# Bear + high vol  → dark red     (institucional vendiendo)
# Bull + low vol   → aqua/cyan    (movimiento sin convicción alcista)
# Bear + low vol   → amber/orange (movimiento sin convicción bajista)
# Bull + normal    → green
# Bear + normal    → red
VOL_BULL_HIGH   = "#006400"   # dark green
VOL_BEAR_HIGH   = "#910000"   # dark red
VOL_BULL_LOW    = "#7FFFD4"   # aqua
VOL_BEAR_LOW    = "#FF9800"   # amber
VOL_BULL_NORMAL = "#26A69A"   # standard green
VOL_BEAR_NORMAL = "#EF5350"   # standard red

VOL_SMA_PERIOD  = 89
VOL_HIGH_MULT   = 1.618       # above this × SMA = high volume
VOL_LOW_MULT    = 0.618       # below this × SMA = low volume


def _candle_colors(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Returns (body_colors, wick_colors) lists for each bar.
    Replicates Pine Script barcolor() logic exactly:
      - high vol bull  → dark green
      - high vol bear  → dark red
      - low vol bull   → aqua
      - low vol bear   → amber
      - normal bull    → green
      - normal bear    → red
    Bull = close >= open. Bear = close < open.
    Volume compared against rolling SMA(89).
    """
    vols   = df["volume"].values
    opens  = df["open"].values
    closes = df["close"].values
    n      = len(vols)

    # Rolling SMA of volume — use expanding window for first 89 bars
    vol_sma = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - VOL_SMA_PERIOD + 1)
        vol_sma[i] = np.mean(vols[start: i + 1])

    colors = []
    for i in range(n):
        bull = closes[i] >= opens[i]
        v    = vols[i]
        sma  = vol_sma[i]

        if np.isnan(sma) or sma == 0:
            colors.append(VOL_BULL_NORMAL if bull else VOL_BEAR_NORMAL)
            continue

        if v > sma * VOL_HIGH_MULT:
            colors.append(VOL_BULL_HIGH if bull else VOL_BEAR_HIGH)
        elif v < sma * VOL_LOW_MULT:
            colors.append(VOL_BULL_LOW if bull else VOL_BEAR_LOW)
        else:
            colors.append(VOL_BULL_NORMAL if bull else VOL_BEAR_NORMAL)

    return colors, colors  # body and wick same color


def build_chart(
    df: pd.DataFrame,
    vp: dict,
    signal: dict,
    confluences: dict,
    symbol: str,
    timeframe: str,
    segments: list | None = None,
) -> go.Figure:
    """
    Builds the full interactive chart.
    df: full OHLCV dataframe
    vp: volume profile dict (developing, last segment)
    signal: signal dict
    confluences: confluences dict
    segments: list of historical segment VPs to draw (optional)
    """

    # ── Limit to last 200 bars for display ────────────────────────────────────
    df_disp = df.tail(200).copy()
    bars = len(df_disp)

    # ── EMA overlays ──────────────────────────────────────────────────────────
    closes = df["close"].values
    ema21 = _ema(closes, 21)[-bars:]
    ema50 = _ema(closes, 50)[-bars:] if len(closes) >= 50 else np.full(bars, np.nan)

    # ── Layout: 3 rows (candles + VP, volume, confluences info) ───────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    x_vals = list(range(bars))
    x_labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in df_disp.index]

    # ── Volume-weighted candle colors ─────────────────────────────────────────
    body_colors, wick_colors = _candle_colors(df_disp)

    # ── Candlesticks — built as two overlapping bar traces ─────────────────────
    # Plotly go.Candlestick doesn't support per-bar colors, so we use:
    #   trace 1: thin bars for wicks (high→low range)
    #   trace 2: wider bars for bodies (open→close range)
    wick_low  = df_disp[["high","low"]].min(axis=1)
    wick_high = df_disp[["high","low"]].max(axis=1)
    body_low  = df_disp[["open","close"]].min(axis=1)
    body_high = df_disp[["open","close"]].max(axis=1)
    # Ensure body has at least a pixel of height for doji candles
    body_high = body_high.where(body_high > body_low, body_low + df_disp["close"] * 0.0001)

    # Wicks
    fig.add_trace(go.Bar(
        x=x_vals,
        y=wick_high - wick_low,
        base=wick_low,
        marker_color=wick_colors,
        marker_line_width=0,
        width=0.15,
        showlegend=False,
        hoverinfo="skip",
        name="_wicks",
    ), row=1, col=1)

    # Bodies
    fig.add_trace(go.Bar(
        x=x_vals,
        y=body_high - body_low,
        base=body_low,
        marker_color=body_colors,
        marker_line_width=0,
        width=0.6,
        showlegend=False,
        name="Price",
        customdata=np.stack([
            df_disp["open"], df_disp["high"],
            df_disp["low"],  df_disp["close"],
            df_disp["volume"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "O: %{customdata[0]:.4f}  H: %{customdata[1]:.4f}<br>"
            "L: %{customdata[2]:.4f}  C: %{customdata[3]:.4f}<br>"
            "Vol: %{customdata[4]:,.0f}<extra></extra>"
        ),
    ), row=1, col=1)

    # ── EMA21 ─────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_vals, y=ema21,
        line=dict(color=COL_EMA21, width=1.5),
        name="EMA21", showlegend=True,
        hovertemplate="EMA21: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # ── EMA50 ─────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_vals, y=ema50,
        line=dict(color=COL_EMA50, width=1.5),
        name="EMA50", showlegend=True,
        hovertemplate="EMA50: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # ── Volume Profile bars (horizontal) ─────────────────────────────────────
    if vp:
        _add_volume_profile(fig, vp, x_vals, bars)

    # ── PoC line ──────────────────────────────────────────────────────────────
    if vp:
        poc = vp["poc_price"]
        vah = vp["vah_price"]
        val = vp["val_price"]
        ph  = vp["price_high"]
        pl  = vp["price_low"]

        # PoC
        fig.add_hline(y=poc, line_color=COL_POC, line_width=1.5,
                      line_dash="solid", row=1, col=1,
                      annotation_text=f"  PoC {poc:.4f}",
                      annotation_font_color=COL_POC,
                      annotation_position="right")
        # VAH
        fig.add_hline(y=vah, line_color=COL_VAH, line_width=1,
                      line_dash="dash", row=1, col=1,
                      annotation_text=f"  VAH {vah:.4f}",
                      annotation_font_color=COL_VAH,
                      annotation_position="right")
        # VAL
        fig.add_hline(y=val, line_color=COL_VAL, line_width=1,
                      line_dash="dash", row=1, col=1,
                      annotation_text=f"  VAL {val:.4f}",
                      annotation_font_color=COL_VAL,
                      annotation_position="right")

        # VA background fill
        fig.add_hrect(y0=val, y1=vah,
                      fillcolor=COL_VA_FILL,
                      layer="below", line_width=0, row=1, col=1)

    # ── Signal levels (stop, target, invalidation) ────────────────────────────
    if signal and signal.get("stop"):
        _add_signal_levels(fig, signal, df_disp["close"].iloc[-1])

    # ── Historical segment PoC lines (lighter) ────────────────────────────────
    if segments:
        for seg in segments[-5:]:  # last 5 historical segments
            seg_vp = seg.get("vp")
            if not seg_vp:
                continue
            # Draw lighter historical PoC
            fig.add_hline(
                y=seg_vp["poc_price"],
                line_color="rgba(255,59,59,0.25)",
                line_width=1, line_dash="dot",
                row=1, col=1,
            )

    # ── Pivot labels ──────────────────────────────────────────────────────────
    if segments:
        _add_pivot_labels(fig, df_disp, segments, x_vals, df_disp.index)

    # ── Volume bars (bottom row) — same 4-color system as candles ─────────────
    fig.add_trace(go.Bar(
        x=x_vals, y=df_disp["volume"],
        marker_color=body_colors,   # reuse already-computed colors
        opacity=0.8,
        name="Volume",
        showlegend=False,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # ── Layout styling ─────────────────────────────────────────────────────────
    tf_label = "1D" if timeframe == "1d" else "4H"
    sig_label = signal.get("signal_label", "") if signal else ""

    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b>  [{tf_label}]  —  {sig_label}",
            font=dict(size=18, color="#E6EDF3", family="'Space Mono', monospace"),
            x=0.01,
        ),
        paper_bgcolor=COL_BG,
        plot_bgcolor=COL_BG,
        font=dict(color="#8B949E", family="'Space Mono', monospace"),
        barmode="overlay",   # wicks and bodies overlap correctly
        xaxis=dict(
            showgrid=True, gridcolor=COL_GRID,
            tickvals=x_vals[::max(1, bars // 10)],
            ticktext=x_labels[::max(1, bars // 10)],
            tickangle=-35,
            showspikes=True, spikecolor="#30363D",
        ),
        xaxis2=dict(showgrid=True, gridcolor=COL_GRID),
        yaxis=dict(showgrid=True, gridcolor=COL_GRID, side="right"),
        yaxis2=dict(showgrid=False, side="right"),
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor=COL_GRID,
            font=dict(size=11),
            x=0.01, y=0.99,
        ),
        annotations=[
            # Volume color legend — bottom left of chart
            dict(
                x=0.01, y=0.22, xref="paper", yref="paper",
                text=(
                    "<span style='color:#006400'>■</span> Alto vol alcista &nbsp;"
                    "<span style='color:#910000'>■</span> Alto vol bajista &nbsp;"
                    "<span style='color:#7FFFD4'>■</span> Bajo vol alcista &nbsp;"
                    "<span style='color:#FF9800'>■</span> Bajo vol bajista"
                ),
                showarrow=False,
                font=dict(size=10, color="#8B949E"),
                bgcolor="rgba(13,17,23,0.7)",
                bordercolor="#30363D",
                borderwidth=1,
            )
        ],
        hovermode="x unified",
        margin=dict(l=10, r=90, t=60, b=40),
        dragmode="pan",
        height=700,
    )

    fig.update_xaxes(rangeslider_visible=False)

    return fig


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _add_volume_profile(fig, vp, x_vals, bars):
    """
    Adds horizontal Volume Profile bars to the chart.
    Bars extend leftward from the right edge of the profile range.
    LVN (low volume nodes) are drawn as 1-pixel-wide bars so gaps are visible.
    """
    vols    = vp["volume_by_level"]
    n_rows  = vp["n_rows"]
    pl      = vp["price_low"]
    step    = vp["price_step"]
    poc_l   = vp["poc_level"]
    vah_l   = vp["vah_level"]
    val_l   = vp["val_level"]
    max_vol = np.max(vols) if np.max(vols) > 0 else 1

    profile_width_bars = max(8, int(bars * 0.12))  # 12% of visible bars
    x_right = x_vals[-1]
    x_anchor = x_right  # bars extend leftward from here

    for level in range(n_rows):
        level_bottom = pl + (level + 0.05) * step
        level_top    = pl + (level + 0.95) * step
        is_in_va     = val_l <= level <= vah_l
        is_poc       = level == poc_l

        vol_frac  = vols[level] / max_vol
        bar_width = max(0.3, vol_frac * profile_width_bars)  # min 0.3 so LVN always visible

        color = (
            "rgba(255,59,59,0.85)"   if is_poc   else
            "rgba(251,192,45,0.60)"  if is_in_va else
            "rgba(67,70,81,0.55)"
        )

        fig.add_shape(
            type="rect",
            x0=x_anchor - bar_width, y0=level_bottom,
            x1=x_anchor,             y1=level_top,
            fillcolor=color,
            line=dict(width=0),
            row=1, col=1,
        )


def _add_signal_levels(fig, signal, close):
    """
    Adds stop, target, and invalidation lines.
    T1/T2 color is direction-aware: green for long targets, red for short targets.
    """
    stop    = signal.get("stop")
    target  = signal.get("target")
    target2 = signal.get("target2")
    inv     = signal.get("invalidation")
    sig     = signal.get("signal", "")

    is_long = sig in ("LONG", "RANGE_LONG")
    t_color  = COL_SIGNAL_LONG  if is_long else COL_SIGNAL_SHORT   # green or red
    t2_color = "#69F0AE"        if is_long else "#FF6B6B"

    if stop is not None:
        fig.add_hline(y=stop, line_color="#FF6D00", line_width=1.5,
                      line_dash="longdash",
                      annotation_text=f"  STOP {stop:.5g}",
                      annotation_font_color="#FF6D00",
                      annotation_position="right", row=1, col=1)
    if target is not None:
        fig.add_hline(y=target, line_color=t_color, line_width=1.5,
                      line_dash="longdash",
                      annotation_text=f"  T1 {target:.5g}",
                      annotation_font_color=t_color,
                      annotation_position="right", row=1, col=1)
    if target2 is not None:
        fig.add_hline(y=target2, line_color=t2_color, line_width=1,
                      line_dash="dot",
                      annotation_text=f"  T2 {target2:.5g}",
                      annotation_font_color=t2_color,
                      annotation_position="right", row=1, col=1)
    if inv is not None:
        fig.add_hline(y=inv, line_color="#E040FB", line_width=1,
                      line_dash="dot",
                      annotation_text=f"  INV {inv:.5g}",
                      annotation_font_color="#E040FB",
                      annotation_position="right", row=1, col=1)


def _add_pivot_labels(fig, df_disp, segments, x_vals, index):
    """Adds pivot high/low labels with % change. Uses O(n) index map."""
    disp_start = df_disp.index[0]
    # Build O(1) lookup map once instead of O(n) list.index() in loop
    ts_to_pos = {ts: i for i, ts in enumerate(index)}

    for seg in segments:
        ts = seg.get("end_ts")
        if ts is None or ts < disp_start or ts not in ts_to_pos:
            continue
        bar_pos    = ts_to_pos[ts]
        ptype      = seg["pivot_type"]
        pprice     = seg["pivot_price"]
        prev_price = seg["prev_pivot_price"]

        pct_str = ""
        if prev_price and prev_price > 0:
            pct = (pprice - prev_price) / prev_price * 100
            pct_str = f"{'+' if pct > 0 else ''}{pct:.1f}%"

        if ptype == "H":
            fig.add_annotation(
                x=bar_pos, y=pprice,
                text=f"▼ {pprice:.4g}<br>{pct_str}",
                showarrow=False,
                font=dict(size=9, color="#FFB3B3"),
                bgcolor="rgba(212,165,165,0.25)",
                bordercolor="#D4A5A5", borderwidth=1,
                yshift=14, row=1, col=1,
            )
        else:
            fig.add_annotation(
                x=bar_pos, y=pprice,
                text=f"▲ {pprice:.4g}<br>{pct_str}",
                showarrow=False,
                font=dict(size=9, color="#B3C6FF"),
                bgcolor="rgba(143,170,220,0.25)",
                bordercolor="#8FAADC", borderwidth=1,
                yshift=-14, row=1, col=1,
            )


import requests
import os


def send_telegram(message: str, bot_token: str, chat_id: str) -> bool:
    """Send a message via Telegram Bot API."""
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[telegram] Error: {e}")
        return False


def format_signal_alert(symbol: str, timeframe: str, signal: dict, vp: dict, close: float) -> str:
    """Formats a rich Telegram alert message."""
    sig = signal["signal"]
    emoji_map = {
        "LONG": "🟢",
        "SHORT": "🔴",
        "RANGE_LONG": "🔵",
        "RANGE_SHORT": "🟠",
        "NEUTRAL": "⚪",
    }
    emoji = emoji_map.get(sig, "⚪")
    tf_label = "Diario" if timeframe == "1d" else "4 Horas"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"{emoji} <b>VP Scanner — {symbol}</b> [{tf_label}]",
        f"📅 {ts}",
        f"",
        f"<b>Señal:</b> {signal['signal_label']}",
        f"<b>Precio actual:</b> {close:.4f}",
        f"",
        f"📊 <b>Volume Profile</b>",
        f"  PoC : {vp['poc_price']:.4f}",
        f"  VAH : {vp['vah_price']:.4f}",
        f"  VAL : {vp['val_price']:.4f}",
    ]

    if signal["stop"]:
        lines += [
            f"",
            f"🎯 <b>Niveles de operativa</b>",
            f"  Stop       : {signal['stop']:.4f}",
            f"  Target 1   : {signal['target']:.4f}",
        ]
        if signal.get("target2"):
            lines.append(f"  Target 2   : {signal['target2']:.4f}")
        lines.append(f"  Invalidación: {signal['invalidation']:.4f}")
        if signal.get("rr"):
            lines.append(f"  R:R        : 1 : {signal['rr']}")

    lines += [
        f"",
        f"📋 <i>{signal['scenario']}</i>",
    ]

    return "\n".join(lines)


def dispatch_alerts(scan_results: list[dict], bot_token: str, chat_id: str,
                    min_score: int = 3, signals_filter: list[str] | None = None):
    """
    Send alerts for actionable signals above a minimum confluence score.
    scan_results: list of result dicts from scanner.
    signals_filter: e.g. ['LONG', 'SHORT'] to limit alert types.
    """
    sent = 0
    for r in scan_results:
        for tf in ["1d", "4h"]:
            tf_data = r.get(tf)
            if not tf_data:
                continue
            sig = tf_data.get("signal", {})
            conf = tf_data.get("confluences", {})
            vp = tf_data.get("vp")
            if not sig or not vp:
                continue
            if sig.get("signal") in ("NEUTRAL", None):
                continue
            if signals_filter and sig["signal"] not in signals_filter:
                continue
            if conf.get("score", 0) < min_score:
                continue

            msg = format_signal_alert(
                symbol=r["symbol"],
                timeframe=tf,
                signal=sig,
                vp=vp,
                close=r.get("close", 0),
            )
            ok = send_telegram(msg, bot_token, chat_id)
            if ok:
                sent += 1
    return sent

"""
app.py — VP Scanner · Volume Profile Dashboard
Streamlit main application.
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VP Scanner", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#F4F6FB!important;}
[data-testid="stSidebar"]{background:#FFFFFF!important;border-right:1px solid #E4E7F0!important;}
[data-testid="stSidebar"] *{font-family:'Inter',sans-serif!important;}
.stSlider label,.stCheckbox label{font-size:12px!important;color:#6B7280!important;}
[data-testid="stMetric"]{background:#FFFFFF!important;border:1px solid #E4E7F0!important;border-radius:10px!important;padding:12px 16px!important;}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif!important;font-size:28px!important;font-weight:700!important;color:#0D1117!important;}
[data-testid="stMetricLabel"]{font-size:10px!important;color:#9CA3AF!important;letter-spacing:.06em!important;}
.stButton>button{background:#3B82F6!important;color:#fff!important;border:none!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;font-size:12px!important;font-weight:600!important;padding:8px 20px!important;}
.stButton>button:hover{background:#2563EB!important;}
.stTabs [data-baseweb="tab-list"]{background:#F1F3F9!important;border-radius:8px!important;padding:3px!important;border:1px solid #E4E7F0!important;}
.stTabs [data-baseweb="tab"]{border-radius:6px!important;font-family:'Inter',sans-serif!important;font-size:11px!important;font-weight:600!important;color:#6B7280!important;}
.stTabs [aria-selected="true"]{background:#FFFFFF!important;color:#3B82F6!important;box-shadow:0 1px 3px rgba(0,0,0,.08)!important;}
hr{border-color:#E4E7F0!important;}
.stSpinner>div{border-top-color:#3B82F6!important;}
[data-testid="stAlert"]{border-radius:8px!important;font-family:'Inter',sans-serif!important;font-size:12px!important;}
.badge{display:inline-flex;align-items:center;gap:3px;padding:3px 9px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.04em;white-space:nowrap;}
.bl{background:#ECFDF5;color:#065F46;}
.bs{background:#FEF2F2;color:#991B1B;}
.brl{background:#EFF6FF;color:#1E40AF;}
.brs{background:#FFFBEB;color:#92400E;}
.bn{background:#F3F4F6;color:#374151;}
.conf-item{display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:7px;background:#F9FAFB;border:1px solid #E4E7F0;margin-bottom:4px;font-size:12px;border-left:3px solid transparent;}
.conf-item.alta{border-left-color:#10B981;}
.conf-item.media{border-left-color:#F59E0B;}
.conf-item.baja{border-left-color:#D1D5DB;}
.metric-card{background:#FFFFFF;border:1px solid #E4E7F0;border-radius:10px;padding:14px 16px;margin:2px 0;}
.tbl-sym{font-size:13px;font-weight:700;color:#0D1117;font-family:'Space Grotesk',sans-serif;cursor:pointer;transition:color .12s;}
.tbl-sym:hover{color:#3B82F6;}
.logo-text{font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#3B82F6;letter-spacing:-.02em;}
.logo-sub{font-size:10px;color:#9CA3AF;letter-spacing:.08em;margin-top:2px;}
/* Hide Streamlit chrome */
header[data-testid="stHeader"]{display:none!important;}
#MainMenu{display:none!important;}
footer{display:none!important;}
.block-container{padding-top:1.5rem!important;}
/* Hide the ugly app.py docstring title */
[data-testid="stAppViewBlockContainer"] > div:first-child p{display:none!important;}
/* Make symbol buttons invisible — just the text matters */
.stButton>button{background:transparent!important;border:none!important;color:#3B82F6!important;font-family:'Space Grotesk',sans-serif!important;font-size:12px!important;font-weight:700!important;padding:0!important;text-align:left!important;box-shadow:none!important;}
.stButton>button:hover{color:#2563EB!important;background:transparent!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "scan_df"      not in st.session_state: st.session_state.scan_df      = None
if "scan_results" not in st.session_state: st.session_state.scan_results = None
if "selected_sym" not in st.session_state: st.session_state.selected_sym = None
if "selected_tf"  not in st.session_state: st.session_state.selected_tf  = "1d"
if "last_scan"    not in st.session_state: st.session_state.last_scan     = None
if "auto_scanned" not in st.session_state: st.session_state.auto_scanned  = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">VP · Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">PIVOT ANCHORED · YAHOO FINANCE</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ Configuración**")
    pivot_length = st.slider("Pivot Length", 5, 50, 20)
    n_rows       = st.slider("Filas del perfil", 10, 50, 25)
    va_pct       = st.slider("Value Area %", 50, 90, 68) / 100
    min_score    = st.slider("Score mínimo", 0, 10, 3)
    st.markdown("---")
    st.markdown("**📡 Filtros**")
    show_long  = st.checkbox("LONG",        True)
    show_short = st.checkbox("SHORT",       True)
    show_rl    = st.checkbox("RANGE LONG",  True)
    show_rs    = st.checkbox("RANGE SHORT", True)
    show_neut  = st.checkbox("NEUTRAL",     False)
    st.markdown("---")
    st.markdown("**🔔 Telegram**")
    tg_token = st.text_input("Bot Token", type="password",
                              value=st.secrets.get("TG_BOT_TOKEN", os.getenv("TG_BOT_TOKEN", "")))
    tg_chat  = st.text_input("Chat ID",
                              value=st.secrets.get("TG_CHAT_ID",   os.getenv("TG_CHAT_ID",   "")))
    st.markdown("---")
    rescan_btn = st.button("🔄 Actualizar datos", use_container_width=True)
    if st.session_state.last_scan:
        st.caption(f"Última actualización: {st.session_state.last_scan}")

# ── Helper: run scan ───────────────────────────────────────────────────────────
def run_scan_now(show_progress=True):
    if show_progress:
        prog = st.progress(0, text="Cargando datos de Yahoo Finance...")
        def cb(i, total, sym):
            prog.progress(int(i/total*100), text=f"Analizando {sym}... ({i+1}/{total})")
    else:
        def cb(i, total, sym): pass

    results = run_full_scan(TOP_SYMBOLS, progress_cb=cb, pivot_length=pivot_length)

    if show_progress:
        prog.progress(100, text="✅ Completado")
        time.sleep(0.3)
        prog.empty()

    df_real = results_to_dataframe(results)
    if not df_real.empty and "signal" in df_real.columns:
        st.session_state.scan_results = results
        st.session_state.scan_df      = df_real
        st.session_state.auto_scanned = True
        st.session_state.last_scan    = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
        if tg_token and tg_chat:
            sigs = []
            if show_long:  sigs.append("LONG")
            if show_short: sigs.append("SHORT")
            if show_rl:    sigs.append("RANGE_LONG")
            if show_rs:    sigs.append("RANGE_SHORT")
            sent = dispatch_alerts(results, tg_token, tg_chat, min_score=min_score, signals_filter=sigs)
            if sent: st.toast(f"📡 {sent} alertas enviadas", icon="✅")

# ── Auto-scan on first load ────────────────────────────────────────────────────
if not st.session_state.auto_scanned:
    run_scan_now(show_progress=True)

if rescan_btn:
    run_scan_now(show_progress=True)
    st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
col_t, col_empty = st.columns([5, 1])
with col_t:
    st.markdown('<h2 style="font-family:Space Grotesk,sans-serif;font-weight:700;font-size:22px;color:#0D1117;margin:0">📊 Crypto Volume Profile Scanner</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:11px;color:#9CA3AF;margin-top:4px;letter-spacing:.04em">TOP {len(TOP_SYMBOLS)} CRIPTOS · YAHOO FINANCE · 1D + 4H · PIVOTS ANCLADOS · RÉPLICA INDICADOR DGT</p>', unsafe_allow_html=True)

# ── Guard ──────────────────────────────────────────────────────────────────────
if st.session_state.scan_df is None or st.session_state.scan_df.empty:
    st.info("Cargando datos... si tarda más de 3 minutos pulsa 🔄 Actualizar datos en el sidebar.")
    st.stop()

df = st.session_state.scan_df.copy()
if "signal" not in df.columns:
    st.warning("Sin datos. Pulsa 🔄 Actualizar datos.")
    st.stop()

# ── Filter ─────────────────────────────────────────────────────────────────────
allowed = []
if show_long:  allowed.append("LONG")
if show_short: allowed.append("SHORT")
if show_rl:    allowed.append("RANGE_LONG")
if show_rs:    allowed.append("RANGE_SHORT")
if show_neut:  allowed.append("NEUTRAL")
df = df[df["signal"].isin(allowed) & (df["score"] >= min_score)]

# ── Metrics ────────────────────────────────────────────────────────────────────
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Señales activas", len(df))
m2.metric("Long",  len(df[df["signal"]=="LONG"]))
m3.metric("Short", len(df[df["signal"]=="SHORT"]))
m4.metric("Range", len(df[df["signal"].isin(["RANGE_LONG","RANGE_SHORT"])]))
m5.metric("Score medio", f"{df['score'].mean():.1f}" if len(df) else "—")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Todas", "📅 Diario 1D", "⏱️ 4 Horas"])

bcls = {"LONG":"bl","SHORT":"bs","RANGE_LONG":"brl","RANGE_SHORT":"brs"}
blbl = {"LONG":"↑ Long → PoC","SHORT":"↓ Short → PoC","RANGE_LONG":"↔ Range Long","RANGE_SHORT":"↔ Range Short"}
border_col = {"LONG":"#10B981","SHORT":"#EF4444","RANGE_LONG":"#3B82F6","RANGE_SHORT":"#F59E0B"}

def render_table(dft, label):
    if dft.empty:
        st.info(f"Sin señales en {label}.")
        return
    sig_order = {"LONG":0,"SHORT":1,"RANGE_LONG":2,"RANGE_SHORT":3,"NEUTRAL":4}
    dft = dft.copy()
    dft["_o"] = dft["signal"].map(sig_order).fillna(9)
    dft = dft.sort_values(["score","_o","symbol"], ascending=[False,True,True])

    cols = st.columns([1.4,.5,1,1,1,1,1.6,1,1,.6,.6])
    headers = ["SÍMBOLO","TF","PRECIO","PoC","VAH","VAL","SEÑAL","STOP","TARGET","R:R","SCORE"]
    for c,h in zip(cols,headers):
        c.markdown(f'<div style="font-size:9px;font-weight:600;color:#9CA3AF;letter-spacing:.1em;padding-bottom:4px">{h}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin:2px 0 6px">', unsafe_allow_html=True)

    for _, row in dft.iterrows():
        bc  = border_col.get(row["signal"],"#D1D5DB")
        sc  = row["score"]
        scc = "#10B981" if sc>=7 else "#F59E0B" if sc>=4 else "#EF4444"
        cls = bcls.get(row["signal"],"bn")
        lbl = blbl.get(row["signal"], row["signal"])
        cv  = row["close"]
        cs  = f"{cv:.4f}" if cv else "—"
        cols = st.columns([1.4,.5,1,1,1,1,1.6,1,1,.6,.6])

        # Clickable symbol as styled text button
        sym_key = f"sym_{row['symbol']}_{row['tf']}_{label}"
        cols[0].markdown(
            f'<div class="tbl-sym" onclick="void(0)" style="font-size:13px;font-weight:700;color:#3B82F6;font-family:Space Grotesk,sans-serif;cursor:pointer">'
            f'{row["symbol"]}</div>',
            unsafe_allow_html=True)
        # Use invisible button to capture click
        if cols[0].button("", key=sym_key, help=f"Ver gráfico de {row['symbol']}"):
            st.session_state.selected_sym = row["symbol"]
            st.session_state.selected_tf  = row["tf"].lower()

        cols[1].markdown(f'<span style="font-size:9px;background:#F1F3F9;border:1px solid #E4E7F0;padding:2px 5px;border-radius:4px;color:#6B7280;font-weight:600">{row["tf"]}</span>', unsafe_allow_html=True)
        cols[2].markdown(f'<span style="font-size:12px;font-family:Space Grotesk,sans-serif">{cs}</span>', unsafe_allow_html=True)
        cols[3].markdown(f'<span style="font-size:12px;color:#EF4444;font-weight:600">{row["poc"]}</span>', unsafe_allow_html=True)
        cols[4].markdown(f'<span style="font-size:12px;color:#3B82F6">{row["vah"]}</span>', unsafe_allow_html=True)
        cols[5].markdown(f'<span style="font-size:12px;color:#3B82F6">{row["val"]}</span>', unsafe_allow_html=True)
        cols[6].markdown(f'<span class="badge {cls}">{lbl}</span>', unsafe_allow_html=True)
        cols[7].markdown(f'<span style="font-size:12px;color:#F59E0B;font-weight:500">{row["stop"] or "—"}</span>', unsafe_allow_html=True)
        cols[8].markdown(f'<span style="font-size:12px;color:#10B981;font-weight:500">{row["target"] or "—"}</span>', unsafe_allow_html=True)
        cols[9].markdown(f'<span style="font-size:12px;color:#8B5CF6;font-weight:500">{row["rr"] or "—"}</span>', unsafe_allow_html=True)
        cols[10].markdown(
            f'<div style="font-size:12px;font-weight:700;color:{scc}">{int(sc)}/10</div>'
            f'<div style="height:3px;border-radius:2px;background:{scc};width:{int(sc)*10}%;opacity:.7;margin-top:2px"></div>',
            unsafe_allow_html=True)

with tab1: render_table(df, "Todas")
with tab2: render_table(df[df["tf"]=="1D"], "1D")
with tab3: render_table(df[df["tf"]=="4H"], "4H")

# ── Chart detail ───────────────────────────────────────────────────────────────
if st.session_state.selected_sym:
    sym = st.session_state.selected_sym
    tf  = st.session_state.selected_tf
    st.markdown("---")
    st.markdown(f'<h3 style="font-family:Space Grotesk,sans-serif;font-weight:700;color:#0D1117">📈 {sym} · {tf.upper()}</h3>', unsafe_allow_html=True)
    if st.button("✕ Cerrar", key="close_chart"):
        st.session_state.selected_sym = None
        st.rerun()

    rm = next((r for r in (st.session_state.scan_results or []) if r["symbol"] == sym), None)
    if rm and tf in rm and rm[tf]:
        td   = rm[tf]
        vp   = td["vp"]
        sig  = td["signal"]
        conf = td["confluences"]
        segs = td["segments"]

        import plotly.graph_objects as go
        fig = build_chart(td["df"], vp, sig, sym, tf, segs)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True,"displayModeBar":True})

        col_l, col_r = st.columns(2)
        sig_name = sig["signal"]
        is_long  = sig_name in ("LONG","RANGE_LONG")
        cls = bcls.get(sig_name,"bn")
        lbl = blbl.get(sig_name, sig_name)
        tC  = "#10B981" if is_long else "#EF4444"
        t2C = "#059669" if is_long else "#DC2626"

        cmap = {
            "verde_oscuro": ("#15803d","Verde oscuro — compra institucional"),
            "rojo_oscuro":  ("#b91c1c","Rojo oscuro — venta institucional"),
            "aqua":         ("#22d3ee","Aqua — alcista sin convicción"),
            "ambar":        ("#f59e0b","Ámbar — bajista sin convicción"),
            "verde_normal": ("#0d9488","Verde — volumen normal"),
            "rojo_normal":  ("#ef4444","Rojo — volumen normal"),
        }
        ct = conf.get("candle_type","verde_normal")
        cc, cd = cmap.get(ct, ("#6B7280","—"))
        close_v = rm.get("close", 0) or 0

        with col_l:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:9px;font-weight:600;color:#9CA3AF;letter-spacing:.1em;margin-bottom:10px">NIVELES DE OPERATIVA</div>
              <div style="margin-bottom:10px;display:flex;align-items:center;gap:8px">
                <span class="badge {cls}" style="font-size:12px">{lbl}</span>
                <span style="font-size:11px;color:{cc}">{cd}</span>
              </div>
              <table style="width:100%;font-size:12px;border-collapse:collapse">
                <tr><td style="color:#6B7280;padding:4px 0;border-bottom:1px solid #F3F4F6">Precio</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;font-weight:600">{close_v:.4f}</td></tr>
                <tr><td style="color:#EF4444;padding:4px 0;border-bottom:1px solid #F3F4F6">PoC</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;color:#EF4444;font-weight:600">{vp['poc_price']:.4f}</td></tr>
                <tr><td style="color:#3B82F6;padding:4px 0;border-bottom:1px solid #F3F4F6">VAH</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;color:#3B82F6">{vp['vah_price']:.4f}</td></tr>
                <tr><td style="color:#3B82F6;padding:4px 0;border-bottom:1px solid #F3F4F6">VAL</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;color:#3B82F6">{vp['val_price']:.4f}</td></tr>
                <tr style="background:#FFFBEB"><td style="color:#F59E0B;padding:4px 6px;font-weight:600">Stop</td><td style="text-align:right;padding:4px 6px;color:#F59E0B;font-weight:600">{sig.get('stop','—'):.4f if sig.get('stop') else '—'}</td></tr>
                <tr><td style="color:{tC};padding:4px 0;border-bottom:1px solid #F3F4F6">Target 1</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;color:{tC};font-weight:600">{sig.get('target','—'):.4f if sig.get('target') else '—'}</td></tr>
                <tr><td style="color:{t2C};padding:4px 0;border-bottom:1px solid #F3F4F6">Target 2</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;color:{t2C}">{sig.get('target2','—'):.4f if sig.get('target2') else '—'}</td></tr>
                <tr><td style="color:#8B5CF6;padding:4px 0;border-bottom:1px solid #F3F4F6">Invalidación</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;color:#8B5CF6">{sig.get('invalidation','—'):.4f if sig.get('invalidation') else '—'}</td></tr>
                <tr><td style="color:#8B5CF6;padding:4px 0;border-bottom:1px solid #F3F4F6">R : R</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6;color:#8B5CF6">{"1 : "+str(sig.get('rr')) if sig.get('rr') else '—'}</td></tr>
                <tr><td style="color:#6B7280;padding:4px 0;border-bottom:1px solid #F3F4F6">RSI(14)</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6">{conf.get('rsi','—')}</td></tr>
                <tr><td style="color:#6B7280;padding:4px 0;border-bottom:1px solid #F3F4F6">EMA21</td><td style="text-align:right;padding:4px 0;border-bottom:1px solid #F3F4F6">{f"{conf['ema21']:.4f}" if conf.get('ema21') else '—'}</td></tr>
                <tr><td style="color:#6B7280;padding:4px 0">Vol/SMA89</td><td style="text-align:right;padding:4px 0">×{conf.get('vol_ratio','—')}</td></tr>
              </table>
              <div style="margin-top:10px;padding:10px;background:#F9FAFB;border-radius:8px;font-size:11px;color:#6B7280;line-height:1.7;border-left:3px solid #3B82F6">
                {sig.get('scenario','')}
              </div>
            </div>""", unsafe_allow_html=True)

            if tg_token and tg_chat:
                if st.button(f"📡 Alerta Telegram — {sym}", key=f"tg_{sym}_{tf}"):
                    ok = send_telegram(format_signal_alert(sym, tf, sig, vp, close_v), tg_token, tg_chat)
                    st.success("✅ Enviada") if ok else st.error("❌ Error")

        with col_r:
            items = conf.get("items", [])
            sc    = conf.get("score", 0)
            scc   = "#10B981" if sc>=7 else "#F59E0B" if sc>=4 else "#EF4444"
            slbl  = "Alta calidad" if sc>=8 else "Sólida" if sc>=6 else "Moderada" if sc>=4 else "Débil"
            ch = "".join([
                f'<div class="conf-item {c["strength"]}">'
                f'<span style="font-size:15px">{c["icon"]}</span>'
                f'<span style="font-size:12px">{c["label"]}</span></div>'
                for c in items
            ]) if items else '<div style="color:#9CA3AF;font-size:12px;padding:8px">Sin confluencias detectadas.</div>'
            extras = [x for x in [
                f"RSI: {conf['rsi']}" if conf.get('rsi') else None,
                f"EMA21: {conf['ema21']:.4f}" if conf.get('ema21') else None,
                f"EMA50: {conf['ema50']:.4f}" if conf.get('ema50') else None,
                f"Vol/SMA: ×{conf['vol_ratio']}" if conf.get('vol_ratio') else None,
            ] if x]
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:9px;font-weight:600;color:#9CA3AF;letter-spacing:.1em;margin-bottom:12px">PANEL DE CONFLUENCIAS</div>
              <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
                <div style="font-family:Space Grotesk,sans-serif;font-size:48px;font-weight:700;color:{scc};line-height:1">{sc}</div>
                <div style="flex:1">
                  <div style="font-size:10px;color:#9CA3AF">Score / 10</div>
                  <div style="width:100%;height:6px;border-radius:3px;background:linear-gradient(90deg,#EF4444,#F59E0B,#10B981);margin:7px 0;position:relative">
                    <div style="position:absolute;top:-4px;left:{sc*10}%;width:2px;height:14px;background:#0D1117;border-radius:2px"></div>
                  </div>
                  <div style="font-size:10px;color:#9CA3AF">{slbl}</div>
                </div>
              </div>
              {ch}
              <div style="margin-top:10px;padding:8px;background:#F9FAFB;border-radius:6px;font-size:11px;color:#9CA3AF">{"  ·  ".join(extras)}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"Sin datos para {sym} en {tf.upper()}. Actualiza el scan.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#9CA3AF;font-size:11px">VP Scanner · Volume Profile Pivot Anchored · Réplica del indicador © dgtrd · Datos vía Yahoo Finance</div>', unsafe_allow_html=True)
