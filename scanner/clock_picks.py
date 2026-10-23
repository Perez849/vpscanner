#!/usr/bin/env python3
"""
clock_picks.py — sectores que el «reloj de inversión» (repositorio Perez849/reloj-inversion) habría recomendado CADA MES, solo con datos previos.

El reloj publica la cartera de hoy y su backtest, pero no el historial de sectores elegidos. Este script reproduce, fuera de red, el bucle mensual de
su `rotation()` (mismo código de selección: `_sleeve_pick`, `_weights`, mismas constantes y la misma mezcla por probabilidad de fase) a partir de los
insumos que el propio reloj vuelca en `lab/` (rendimientos mensuales X, fases, probabilidades, factores) y guarda, para cada mes t, qué sectores de
renta variable (y de oro) se eligen EXPLÍCITAMENTE con información hasta el final del mes t-1.

Diferencia con el reloj original, a propósito y a favor de no hacer trampa: el original lee las probabilidades de fase del propio mes t; aquí se leen
las del mes t-1 (lo que se sabía al empezar el mes).

Uso:  python clock_picks.py <carpeta del repositorio reloj-inversion> [salida.json]
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

FROM_YEAR = 2005


def load_inputs(root: str):
    lab = os.path.join(root, 'lab')
    rd = lambda n: pd.read_csv(os.path.join(lab, n), index_col=0, parse_dates=True)
    X, probs, F = rd('X.csv'), rd('probs.csv'), rd('F.csv')
    ph = pd.read_csv(os.path.join(lab, 'phases.csv'), index_col=0, parse_dates=True).iloc[:, 0]
    cls_map = json.load(open(os.path.join(lab, 'cls_map.json'), encoding='utf-8'))
    return X, ph, probs, F, cls_map


def monthly_picks(root: str, lag_probs: int = 1, from_year: int = FROM_YEAR, min_train: int = 120):
    sys.path.insert(0, os.path.join(root, 'scripts'))
    import build_data as B                               # noqa: E402
    X, phases, probs, F, cls_map = load_inputs(root)
    PH = B.PHASES
    common = X.dropna(how='all').index.intersection(phases.dropna().index)
    X, ph = X.loc[common], phases.loc[common]
    HL = B.HALF_LIFE_M

    def _ew(frame, ref):
        months = np.array([(ref.year - d.year) * 12 + (ref.month - d.month) for d in frame.index], dtype=float)
        return pd.Series(0.5 ** (months / HL), index=frame.index)

    def _wmean(frame, w):
        ww = frame.notna().mul(w, axis=0)
        return frame.mul(w, axis=0).sum() / ww.sum().replace(0, np.nan)

    def factor_means(hist, hph, phase):
        b = B.factor_betas(hist, F) if F is not None else None
        m = hph[hph == phase].index.intersection(F.index)
        if not b or len(m) < 6:
            return None
        g, i = float(F.loc[m, 'growth'].mean()), float(F.loc[m, 'inflation'].mean())
        return pd.Series({k: v[0] + v[1] * g + v[2] * i for k, v in b.items()})

    def means(hist, hph, phase):
        ref = hist.index[-1]
        grand = _wmean(hist, _ew(hist, ref))
        mu_p, se_p, n_p = {}, {}, {}
        for p in PH:
            sub = hist[hph == p]
            if sub.empty:
                continue
            mu_p[p] = _wmean(sub, _ew(sub, ref))
            n = sub.notna().sum()
            n_p[p] = n
            se_p[p] = sub.std() / np.sqrt(n.clip(lower=1))
        if phase not in mu_p or len(mu_p) < 2:
            fm = factor_means(hist, hph, phase)
            if fm is not None and fm.notna().sum() >= 5:
                return fm.reindex(hist.columns) - grand
            return mu_p.get(phase, grand) - grand
        M = pd.DataFrame(mu_p).T
        SE2 = pd.DataFrame(se_p).T ** 2
        N = pd.DataFrame(n_p).T
        tau2 = (M.var(axis=0, ddof=1) - SE2.mean(axis=0)).clip(lower=0.0)
        d = tau2 + SE2.loc[phase]
        w = (tau2 / d).where(d > 0, 0.0).fillna(0.0)
        w = w.where(N.loc[phase] >= B.MIN_PHASE_OBS, 0.0)
        return w * (M.loc[phase] - grand)

    def ew_vol(hist):
        w = _ew(hist, hist.index[-1])
        m = _wmean(hist, w)
        dev2 = hist.sub(m, axis=1) ** 2
        ww = dev2.notna().mul(w, axis=0)
        return (dev2.mul(w, axis=0).sum() / ww.sum().replace(0, np.nan)) ** 0.5

    sleeves = B.SLEEVES
    over = {('Estanflación', 'Renta variable'): (3, 3)}
    out = {}
    for k in range(min_train, len(common)):
        t = common[k]
        if t.year < from_year:
            continue
        sig = ph.iloc[k - 1]
        hist, hph = X.iloc[:k], ph.iloc[:k]
        vol = ew_vol(hist)
        sub_sig = hist[hph == sig]
        raw_phase = (_wmean(sub_sig, _ew(sub_sig, hist.index[-1])) if not sub_sig.empty else _wmean(hist, _ew(hist, hist.index[-1])))
        tp = common[k - lag_probs] if lag_probs else t
        mu, pr = None, None
        if tp in probs.index:
            pr = probs.loc[tp]
            for phase in PH:
                w_p = float(pr.get(phase, 0.0))
                if not np.isfinite(w_p) or w_p <= 0.01:
                    continue
                m_p = means(hist, hph, phase)
                mu = m_p * w_p if mu is None else mu.add(m_p * w_p, fill_value=0.0)
        if mu is None:
            mu = means(hist, hph, sig)
        avail = list(X.loc[t].dropna().index)
        picks, scores = {}, {}
        for name, (classes, lo, hi, n_min, n_max) in sleeves.items():
            if (sig, name) in over:
                n_min, n_max = over[(sig, name)]
            picks[name], scores[name] = B._sleeve_pick(mu, vol, raw_phase, avail, classes, cls_map, n_min, n_max)
        if not any(picks.values()):
            continue
        budgets = B._sleeve_weights(scores, sleeves)
        w_all = {}
        for name, top in picks.items():
            if not top:
                continue
            inner = B._weights(top, vol, 'equal').to_dict()
            for c, wt in inner.items():
                w_all[c] = w_all.get(c, 0.0) + budgets[name] * float(wt)
        tot = sum(w_all.values())
        out[t.strftime('%Y-%m')] = {
            'phase': sig,
            'probs': ({p: round(float(pr.get(p, 0.0)), 3) for p in PH} if pr is not None else None),
            'equity': {c: round(v / tot, 4) for c, v in w_all.items() if cls_map.get(c) == 'Renta variable'},
            'gold': {c: round(v / tot, 4) for c, v in w_all.items() if cls_map.get(c) == 'Oro'},
        }
    return out


def main():
    root = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'model', 'clock_picks.json')
    try:
        commit = subprocess.check_output(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        commit = ''
    months = monthly_picks(root)
    json.dump({'source': 'Perez849/reloj-inversion', 'commit': commit, 'probsLag': 1, 'months': months}, open(dst, 'w'), ensure_ascii=False, indent=0)
    print(f'{len(months)} meses → {dst}')
    last = sorted(months)[-1]
    print(last, months[last])


if __name__ == '__main__':
    main()
