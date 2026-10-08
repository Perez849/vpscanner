"""
setups.py — Biblioteca de setups (patrones de entrada), todos causales y con parámetros
FIJOS tomados de la literatura de reversión a la media / tendencia (no optimizados a mano
sobre el histórico). Qué setups se usan de verdad lo decide la validación fuera de muestra
(research.py → model/validated.json), no este fichero.

Cada setup devuelve un array booleano: True en las barras (cierre) donde se dispara.
Largos: se compra en un retroceso dentro de una tendencia alcista (precio > SMA200).
Cortos: se vende un rebote dentro de una tendencia bajista (precio < SMA200).
El Volume Profile (idea original) entra como "descuento sobre el valor justo": precio por
debajo del VAL del perfil móvil de 60 barras (largos) o sobre el VAH (cortos).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np

MIN_DVOL = 2e6      # liquidez mínima: volumen diario medio en moneda (acciones/ETF)
MIN_PRICE = 3.0
NO_VOLUME_GROUPS = {'fx', 'futures', 'index'}


@dataclass(frozen=True)
class Setup:
    id: str
    dir: int                      # +1 largo / -1 corto
    family: str
    label: str
    fn: Callable[[Dict[str, np.ndarray]], np.ndarray]
    needs_vp: bool = False


def _lt(a, x):
    return np.isfinite(a) & (a < x)


def _gt(a, x):
    return np.isfinite(a) & (a > x)


def _ge(a, x):
    return np.isfinite(a) & (a >= x)


def _up(F):
    c, s = F['c'], F['sma200']
    return np.isfinite(s) & (c > s)


def _dn(F):
    c, s = F['c'], F['sma200']
    return np.isfinite(s) & (c < s)


def preselect_vp(F) -> np.ndarray:
    """Barras donde merece la pena calcular el perfil de volumen (ahorra cómputo)."""
    return (_up(F) & _lt(F['rsi2'], 30)) | (_dn(F) & _gt(F['rsi2'], 70))


def _vp_ok_long(F):
    return _up(F) & _lt(F['rsi2'], 30) & np.isfinite(F['vp_val']) & (F['c'] < F['vp_val'])


def _vp_ok_short(F):
    return _dn(F) & _gt(F['rsi2'], 70) & np.isfinite(F['vp_vah']) & (F['c'] > F['vp_vah'])


def _build() -> List[Setup]:
    S: List[Setup] = []
    # ── LARGOS: retroceso en tendencia alcista ───────────────────────────
    for th in (5, 10, 15):
        S.append(Setup(f'L_rsi2_{th}', +1, 'rsi2', f'RSI(2) < {th} sobre SMA200',
                       lambda F, th=th: _up(F) & _lt(F['rsi2'], th)))
    for th in (0.10, 0.20):
        S.append(Setup(f'L_ibs_{int(th*100)}', +1, 'ibs', f'Cierre en el {int(th*100)}% inferior de la vela, sobre SMA200',
                       lambda F, th=th: _up(F) & _lt(F['ibs'], th) & _lt(F['ret1'], 0)))
    for k in (3, 4, 5):
        S.append(Setup(f'L_streak{k}', +1, 'streak', f'{k}+ cierres bajistas seguidos sobre SMA200',
                       lambda F, k=k: _up(F) & _ge(F['dn_streak'], k)))
    for k in (5, 10, 20):
        S.append(Setup(f'L_low{k}', +1, 'lowN', f'Mínimo de cierre de {k} sesiones sobre SMA200',
                       lambda F, k=k: _up(F) & (F[f'lowest{k}'] > 0)))
    for z in (2.0, 2.5):
        S.append(Setup(f'L_bb{int(z*10)}', +1, 'bollinger', f'Cierre {z} desv. bajo la media de 20 sobre SMA200',
                       lambda F, z=z: _up(F) & _lt(F['bbz'], -z)))
    S.append(Setup('L_pbtrend', +1, 'pullback', 'Retroceso en tendencia fuerte (SMA50>SMA200)',
                   lambda F: _gt(F['sma50'] - F['sma200'], 0) & (F['c'] > F['sma50']) & (F['c'] < F['sma10'])
                   & _ge(F['dn_streak'], 2) & _lt(F['rsi14'], 50)))
    S.append(Setup('L_capit', +1, 'capitulacion', 'Capitulación: volumen x2 + caída fuerte + cierre bajo',
                   lambda F: _up(F) & _gt(F['vol_ratio'], 2.0) & (F['ret1'] < -1.5 * F['atrp']) & _lt(F['ibs'], 0.3)))
    S.append(Setup('L_dip12', +1, 'deepdip', 'Caída >12% desde máx. 20d con RSI(2)<15 sobre SMA200',
                   lambda F: _up(F) & _lt(F['dd20'], -12) & _lt(F['rsi2'], 15)))
    S.append(Setup('L_vpval', +1, 'vp', 'Bajo el VAL del perfil de volumen móvil (60d) + RSI(2)<30 sobre SMA200',
                   _vp_ok_long, needs_vp=True))
    S.append(Setup('L_rsi2_5_any', +1, 'control', 'CONTROL: RSI(2)<5 sin filtro de tendencia',
                   lambda F: _lt(F['rsi2'], 5)))
    # BENCHMARK de entrada aleatoria (misma tendencia, sin señal): un setup solo aporta si lo supera
    S.append(Setup('B_up', +1, 'baseline', 'BENCHMARK: cualquier barra sobre SMA200', lambda F: _up(F)))
    # ── CORTOS: rebote en tendencia bajista ───────────────────────────────
    for th in (95, 90):
        S.append(Setup(f'S_rsi2_{th}', -1, 'rsi2', f'RSI(2) > {th} bajo SMA200',
                       lambda F, th=th: _dn(F) & _gt(F['rsi2'], th)))
    for th in (0.90, 0.80):
        S.append(Setup(f'S_ibs_{int(th*100)}', -1, 'ibs', f'Cierre en el {int(round((1-th)*100))}% superior de la vela, bajo SMA200',
                       lambda F, th=th: _dn(F) & _gt(F['ibs'], th) & _gt(F['ret1'], 0)))
    for k in (3, 4):
        S.append(Setup(f'S_streak{k}', -1, 'streak', f'{k}+ cierres alcistas seguidos bajo SMA200',
                       lambda F, k=k: _dn(F) & _ge(F['up_streak'], k)))
    for k in (5, 10):
        S.append(Setup(f'S_high{k}', -1, 'highN', f'Máximo de cierre de {k} sesiones bajo SMA200',
                       lambda F, k=k: _dn(F) & (F[f'highest{k}'] > 0)))
    S.append(Setup('S_bb2', -1, 'bollinger', 'Cierre 2 desv. sobre la media de 20 bajo SMA200',
                   lambda F: _dn(F) & _gt(F['bbz'], 2.0)))
    S.append(Setup('S_vpvah', -1, 'vp', 'Sobre el VAH del perfil de volumen móvil (60d) + RSI(2)>70 bajo SMA200',
                   _vp_ok_short, needs_vp=True))
    S.append(Setup('S_rsi2_98_any', -1, 'control', 'CONTROL: RSI(2)>98 sin filtro de tendencia',
                   lambda F: _gt(F['rsi2'], 98)))
    S.append(Setup('B_dn', -1, 'baseline', 'BENCHMARK: cualquier barra bajo SMA200', lambda F: _dn(F)))
    return S


SETUPS: List[Setup] = _build()
BY_ID: Dict[str, Setup] = {s.id: s for s in SETUPS}


def tradable_mask(F: Dict[str, np.ndarray], group: str) -> np.ndarray:
    """Filtros de liquidez/precio para que la señal sea operable."""
    ok = np.isfinite(F['atr']) & (F['atr'] > 0) & (F['c'] >= (MIN_PRICE if group not in NO_VOLUME_GROUPS else 0))
    if group not in NO_VOLUME_GROUPS:
        ok &= np.isfinite(F['dvol20']) & (F['dvol20'] >= MIN_DVOL)
    return ok


def attach_vp(F: Dict[str, np.ndarray], W: int = 60) -> None:
    """Calcula el perfil móvil solo donde hace falta y lo deja en F['vp_*'] (NaN en el resto)."""
    from feats import vp_levels
    n = len(F['c'])
    for k in ('vp_poc', 'vp_vah', 'vp_val'):
        F[k] = np.full(n, np.nan)
    idx = np.flatnonzero(preselect_vp(F))
    if len(idx):
        poc, vah, val = vp_levels(F, idx, W=W)
        F['vp_poc'][idx], F['vp_vah'][idx], F['vp_val'][idx] = poc, vah, val


def detect(F: Dict[str, np.ndarray], group: str, setups: List[Setup] | None = None,
           cooldown: int = 10) -> Dict[str, np.ndarray]:
    """Devuelve {setup.id: índices de barras de señal} aplicando liquidez y enfriamiento."""
    setups = setups or SETUPS
    trad = tradable_mask(F, group)
    out: Dict[str, np.ndarray] = {}
    for s in setups:
        if s.needs_vp and 'vp_val' not in F:
            attach_vp(F)
        m = s.fn(F) & trad
        if cooldown > 1:
            # sin estado: solo cuenta la PRIMERA señal tras (cooldown-1) barras sin disparo
            cs = np.concatenate([[0], np.cumsum(m.astype(np.int64))])
            n = len(m)
            lo = np.maximum(np.arange(n) - (cooldown - 1), 0)
            prev = cs[np.arange(n)] - cs[lo]            # disparos en las cooldown-1 barras previas
            m = m & (prev == 0)
        idx = np.flatnonzero(m).astype(np.int64)
        out[s.id] = idx
    return out


UNION_COOLDOWN = 5


def long_pattern_ids() -> List[str]:
    """Patrones que generan candidatos (largos, sin benchmark ni controles). Orden = banderas del modelo."""
    return [s.id for s in SETUPS if s.dir > 0 and s.family not in ('baseline', 'control')]


def candidate_at_last(F: Dict[str, np.ndarray], group: str, pattern_ids: List[str]):
    """
    ¿Hay candidato en la última barra? Devuelve el vector de banderas (qué patrones saltaron) o None.
    Misma regla que la investigación: algún patrón dispara hoy y NINGUNO lo hizo en las 4 barras previas.
    """
    n = len(F['c'])
    last = n - 1
    ev = detect(F, group, [BY_ID[i] for i in pattern_ids])
    fire = np.zeros(n, dtype=bool)
    for idx in ev.values():
        fire[idx] = True
    if not fire[last] or fire[max(0, last - (UNION_COOLDOWN - 1)):last].any():
        return None
    return np.array([1.0 if (len(ev[i]) and ev[i][-1] == last) else 0.0 for i in pattern_ids], dtype=np.float32)
