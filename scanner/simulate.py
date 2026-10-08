"""
simulate.py — Simulación de operaciones (vectorizada) con reglas HONESTAS:

  · La señal se calcula con el cierre de la barra i; se entra en la APERTURA de i+1
    (lo único que se puede ejecutar de verdad tras ver el cierre).
  · Stop y objetivo se fijan desde el precio de entrada con múltiplos del ATR(14) de i.
  · Huecos: si la apertura ya está más allá del stop → se sale a la apertura (peor que el
    stop); si está más allá del objetivo → se sale a la apertura.
  · Si una misma vela toca stop Y objetivo → se asume el STOP (criterio pesimista; el
    sistema anterior desempataba por cercanía a la apertura, lo cual es optimista).
  · Salida por tiempo: cierre de la barra H.
  · Costes de ida y vuelta por grupo de activo, restados del resultado.

La MISMA función se usa en el backtest y en el seguimiento en vivo, así que ambos son
comparables 1:1.
"""
from __future__ import annotations
from typing import Dict, List, NamedTuple, Optional
import numpy as np

HMAX = 10

# Coste de ida y vuelta (%) por grupo. Conservador: incluye comisión + deslizamiento.
COST_RT = {
    'us_large': 0.15, 'us_mid': 0.25, 'us_small': 0.40, 'thematic': 0.40, 'etf': 0.10,
    'crypto': 0.30, 'fx': 0.05, 'futures': 0.08, 'eu': 0.25, 'index': 0.10,
}


class Variant(NamedTuple):
    kind: str            # 'atr' (objetivo+stop+tiempo) | 'sig' (cierre sobre SMA5) | 'rsi' (RSI2>70) | 'ph' (cierre > máx. previo)
    T: float             # objetivo en ATR (solo 'atr')
    S: float             # stop en ATR
    H: int               # barras máximas

    @property
    def name(self) -> str:
        return f'{self.kind}_T{self.T:g}_S{self.S:g}_H{self.H}' if self.kind == 'atr' else f'{self.kind}_S{self.S:g}_H{self.H}'


def default_variants() -> List[Variant]:
    v: List[Variant] = []
    for H in (5, 10):
        for T in (0.5, 1.0, 1.5, 2.5):
            for S in (1.5, 2.5, 4.0):
                v.append(Variant('atr', T, S, H))
    for H in (5, 10):
        for S in (2.5, 4.0):
            v.append(Variant('sig', 0.0, S, H))
    for kind in ('rsi', 'ph'):
        for H in (5, 10):
            v.append(Variant(kind, 0.0, 4.0, H))
    return v


class Result(NamedTuple):
    pnl: np.ndarray      # % neto de costes (NaN si no hay barras suficientes)
    kind: np.ndarray     # +1 objetivo, -1 stop, 0 tiempo/señal
    bars: np.ndarray     # barras mantenida
    entry: np.ndarray    # precio de entrada (apertura i+1)


def _forward(F: Dict[str, np.ndarray], idx: np.ndarray, sgn: int, hmax: int):
    """Matrices (n_ev, hmax) de O,H,L,C,SMA5 de las barras i+1..i+hmax; en espacio 'long'."""
    n = len(F['c'])
    cols = idx[:, None] + 1 + np.arange(hmax)[None, :]
    valid = cols < n
    cc = np.minimum(cols, n - 1)
    O, Hh, Ll, C, S5, R2 = F['o'][cc], F['h'][cc], F['l'][cc], F['c'][cc], F['sma5'][cc], F['rsi2'][cc]
    sig_h = F['h'][idx]
    if sgn < 0:
        O, Hh, Ll, C, S5, R2 = -O, -Ll, -Hh, -C, -S5, -R2
        sig_h = -F['l'][idx]
    PH = np.concatenate([sig_h[:, None], Hh[:, :-1]], 1)     # máximo de la barra anterior (en espacio 'largo')
    return O, Hh, Ll, C, S5, R2, PH, valid


def simulate(F: Dict[str, np.ndarray], idx: np.ndarray, sgn: int, variants: List[Variant],
             cost_pct: float | np.ndarray = 0.30) -> Dict[str, Result]:
    """idx = barras de señal (cierre). sgn=+1 largo, -1 corto. Devuelve {variant.name: Result}."""
    idx = np.asarray(idx, dtype=np.int64)
    out: Dict[str, Result] = {}
    if len(idx) == 0:
        return out
    hmax = max(v.H for v in variants)
    O, Hh, Ll, C, S5, R2, PH, valid = _forward(F, idx, sgn, hmax)
    rthr = 70.0 if sgn > 0 else -30.0
    atr = F['atr'][idx]
    E = O[:, 0]
    absE = np.abs(E)
    rows = np.arange(len(idx))
    cost = np.broadcast_to(np.asarray(cost_pct, dtype=np.float64), idx.shape)

    for v in variants:
        H = v.H
        ok_all = valid[:, H - 1] & np.isfinite(atr) & np.isfinite(E)
        stop = E - v.S * atr
        stop_hit = Ll[:, :H] <= stop[:, None]
        gap_stop = O[:, :H] <= stop[:, None]
        if v.kind == 'atr':
            tgt = E + v.T * atr
            tgt_hit = Hh[:, :H] >= tgt[:, None]
            gap_tgt = O[:, :H] >= tgt[:, None]
            ev = stop_hit | tgt_hit
            any_ev = ev.any(1)
            k = ev.argmax(1)
            Ok = O[rows, k]
            g_t = gap_tgt[rows, k]
            g_s = gap_stop[rows, k]
            s_h = stop_hit[rows, k]
            exit_ev = np.where(g_t | g_s, Ok, np.where(s_h, stop, tgt))
            exit_px = np.where(any_ev, exit_ev, C[:, H - 1])
            bars = np.where(any_ev, k + 1, H)
            kind = np.where(any_ev, np.where(exit_px >= tgt - 1e-12 * absE, 1, -1), 0)
        else:
            if v.kind == 'sig':
                cond = (C[:, :H] > S5[:, :H]) & np.isfinite(S5[:, :H])
            elif v.kind == 'rsi':
                cond = (R2[:, :H] > rthr) if sgn > 0 else (R2[:, :H] > rthr)
            else:
                cond = C[:, :H] > PH[:, :H]
            has_c = cond.any(1)
            kc = np.where(has_c, cond.argmax(1), 10 ** 6)
            has_s = stop_hit.any(1)
            ks = np.where(has_s, stop_hit.argmax(1), 10 ** 6)
            stop_first = has_s & (ks <= kc)
            Os = O[rows, np.minimum(ks, H - 1)]
            stop_px = np.where(gap_stop[rows, np.minimum(ks, H - 1)], Os, stop)
            sig_next = has_c & ~stop_first & (kc < H - 1)
            Onext = O[rows, np.minimum(kc + 1, H - 1)]
            exit_px = np.where(stop_first, stop_px, np.where(sig_next, Onext, C[:, H - 1]))
            bars = np.where(stop_first, ks + 1, np.where(sig_next, kc + 2, H))
            kind = np.where(stop_first, -1, 0)
        pnl = (exit_px - E) / absE * 100.0 - cost
        pnl = np.where(ok_all, pnl, np.nan)
        out[v.name] = Result(pnl, kind.astype(np.int8), bars.astype(np.int16), np.where(sgn < 0, -E, E))
    return out
