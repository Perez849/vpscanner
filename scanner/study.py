#!/usr/bin/env python3
"""
study.py — Estudio AMPLIADO de fiabilidad (comando `reasons` de research.py). NO interviene en producción.

Pregunta: ¿están bien estimadas las probabilidades y son fiables los motivos de compra? ¿Se puede detectar más fiabilidad?

  S1  Auditoría de calibración: AUC por año, Brier frente a la tasa base, calibración ANIDADA (solo con años anteriores)
      y probabilidad mostrada frente a acierto real por año.
  S2  Auditoría de los motivos de compra: rendimiento de cada uno de los 17 patrones y de la confluencia de patrones.
  S3  Nueva información (todo causal): tendencia a revertir de cada valor, régimen de reversión de todo el mercado,
      caída propia frente a su sector.
  S4  Modelos alternativos (gradient boosting poco profundo) y reglas de CONFLUENCIA entre modelos.

Protocolo: walk-forward (cada año se predice con modelos entrenados solo con los 3 años anteriores). Una idea solo vale
si mejora año a año y se repite con salidas vecinas; con tantas pruebas, una mejora aislada puede ser azar.
"""
from __future__ import annotations
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

from research import (FEATURES, GROUPS, SU, SM, day_of, year_of, stats, union_events, topn_mask, PLANS, rank_in_day)


# ═════════════════════════════════════════════════════════════════════════
#  Variables nuevas (causales)
# ═════════════════════════════════════════════════════════════════════════
def exit_day(day: np.ndarray, bars: np.ndarray) -> np.ndarray:
    """Día natural en que ya se conoce el resultado de la operación (cota conservadora: sesiones × 7/5 + 1)."""
    return day.astype(np.int64) + np.ceil(np.nan_to_num(bars.astype(float), nan=12.0) * 7 / 5).astype(np.int64) + 1


def feat_symbol_history(day, sym, pnl, bars, window_days: int = 1095, m: float = 5.0, prior_wr: float = 0.65):
    """
    «Tendencia a revertir» de cada valor: media (y % de aciertos) de SUS operaciones anteriores ya cerradas en los
    últimos 3 años, encogida hacia 0 (y hacia 65 %) con m operaciones ficticias. Solo usa resultados conocidos antes de la señal.
    """
    n = len(day)
    mean = np.full(n, np.nan); wr = np.full(n, np.nan); cnt = np.zeros(n)
    exd = exit_day(day, bars)
    order = np.lexsort((day, sym))
    sy = sym[order]
    cuts = np.flatnonzero(np.diff(sy)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, n]):
        idx = order[a:b]
        d, e, p = day[idx], exd[idx], pnl[idx]
        ok = np.isfinite(p)
        for k in range(1, len(idx)):
            sel = (e[:k] < d[k]) & (d[:k] >= d[k] - window_days) & ok[:k]
            c = int(sel.sum())
            if c:
                pk = p[:k][sel]
                mean[idx[k]] = pk.sum() / (c + m)
                wr[idx[k]] = ((pk > 0).sum() + m * prior_wr) / (c + m)
            cnt[idx[k]] = c
    return mean, wr, cnt


def feat_market_reversion(day, pnl, bars, window_days: int, min_n: int = 50):
    """Régimen de reversión de TODO el mercado: media de las operaciones (de todos los valores) cerradas en los últimos `window_days` días."""
    ex = exit_day(day, bars)
    ok = np.isfinite(pnl)
    d0 = int(day.min()) - window_days - 5
    size = int(max(ex.max(), day.max())) - d0 + 10
    cs = np.cumsum(np.bincount(ex[ok] - d0, weights=pnl[ok], minlength=size))
    cc = np.cumsum(np.bincount(ex[ok] - d0, minlength=size))
    hi = (day - 1 - d0).astype(np.int64)
    lo = (day - 1 - window_days - d0).astype(np.int64)
    s_ = cs[hi] - cs[lo]
    c_ = cc[hi] - cc[lo]
    return np.where(c_ >= min_n, s_ / np.maximum(c_, 1), np.nan)


class SectorAgg:
    """Acumula, por sector y día, la rentabilidad media a 1 y 5 sesiones y el % de valores en sobreventa (RSI2<10)."""
    def __init__(self, d0: int = 15000, d1: int = 24000):
        self.d0, self.n = d0, d1 - d0
        self.t: Dict[str, np.ndarray] = {}

    def add(self, F, sector: str):
        a = self.t.setdefault(sector, np.zeros((4, self.n)))
        day = (F['t'] // 86400).astype(np.int64) - self.d0
        ok = (day >= 0) & (day < self.n) & np.isfinite(F['ret5']) & np.isfinite(F['ret1']) & np.isfinite(F['rsi2'])
        d = day[ok]
        a[0, d] += 1
        a[1, d] += np.clip(F['ret5'][ok], -50, 50)
        a[2, d] += np.clip(F['ret1'][ok], -25, 25)
        a[3, d] += (F['rsi2'][ok] < 10)


def feat_sector(agg: SectorAgg, sector_of_event: np.ndarray, day: np.ndarray, own_ret5, own_ret1, own_os, min_n: int = 8):
    n = len(day)
    sec5 = np.full(n, np.nan); rel5 = np.full(n, np.nan); secos = np.full(n, np.nan); sec1 = np.full(n, np.nan)
    for s, a in agg.t.items():
        m = sector_of_event == s
        if not m.any():
            continue
        d = (day[m] - agg.d0).astype(np.int64)
        cnt = a[0, d] - 1
        good = cnt >= min_n - 1
        r5 = (a[1, d] - np.clip(own_ret5[m], -50, 50)) / np.maximum(cnt, 1)
        r1 = (a[2, d] - np.clip(own_ret1[m], -25, 25)) / np.maximum(cnt, 1)
        os_ = (a[3, d] - own_os[m]) / np.maximum(cnt, 1) * 100
        idx = np.flatnonzero(m)
        sec5[idx] = np.where(good, r5, np.nan); sec1[idx] = np.where(good, r1, np.nan)
        secos[idx] = np.where(good, os_, np.nan)
        rel5[idx] = np.where(good, own_ret5[m] - r5, np.nan)
    return sec5, rel5, secos, sec1


# ═════════════════════════════════════════════════════════════════════════
#  Estudio
# ═════════════════════════════════════════════════════════════════════════
def run(EV: Dict, out_dir: str, data: Dict | None = None, uni: Dict | None = None):
    import model as MD
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    here = os.path.dirname(os.path.abspath(__file__))
    reg = json.load(open(os.path.join(here, 'model', 'validated.json')))
    groups_of = {s['id']: s['groups'] for s in reg['strategies']}
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym, gid = EV['day'][rows], EV['sym'][rows], EV['grp'][rows]
    years = year_of(day)
    X = EV['X'][rows]
    yrs = [Y for Y in sorted(set(years)) if Y >= 2019]
    nF = len(FEATURES)
    print(f'\n######## REASONS · {len(rows):,} candidatos largos · walk-forward {yrs[0]}–{yrs[-1]}', flush=True)

    # ── columnas extra ───────────────────────────────────────────────────
    sector_of_event = None
    if uni is not None:
        gics = np.array([uni.get(EV['sym_names'][k], {}).get('gics') or '' for k in range(len(EV['sym_names']))])
        if os.environ.get('STUDY_FAKE_SECTORS'):
            gics = np.array([f'S{k % 8}' for k in range(len(EV['sym_names']))])
        cover = float(np.mean(gics[sym] != ''))
        print(f'  cobertura de sector (GICS) entre los candidatos: {cover * 100:.0f}%', flush=True)
        if cover > 0.3 and data is not None:
            sector_of_event = gics[sym]
    sec_cols = {}
    if sector_of_event is not None:
        import feats
        agg = SectorAgg()
        for k, sname in enumerate(EV['sym_names']):
            if gics[k] and sname in data:
                agg.add(feats.build(data[sname]), gics[k])
        r5, r1, o_ = X[:, FEATURES.index('ret5')], X[:, FEATURES.index('ret1')], (X[:, FEATURES.index('rsi2')] < 10).astype(float)
        s5, rel5, sos, s1 = feat_sector(agg, sector_of_event, day, r5, r1, o_)
        sec_cols = {'sec_ret5': s5, 'rel_sec5': rel5, 'sec_os': sos, 'sec_ret1': s1}
        print(f"  variables de sector con dato: {np.mean(np.isfinite(rel5)) * 100:.0f}% de los candidatos", flush=True)

    def build_ext(pnl, bars):
        h_mean, h_wr, h_n = feat_symbol_history(day, sym, pnl, bars)
        cols = {'sym_hist': h_mean, 'sym_wr': h_wr, 'mkt30': feat_market_reversion(day, pnl, bars, 30),
                'mkt90': feat_market_reversion(day, pnl, bars, 90), **sec_cols}
        return cols

    def fi(fs: List[str], ext_names: List[str] | None = None, ext_idx: Dict[str, int] | None = None):
        return [FEATURES.index(f) for f in fs] + ([ext_idx[e] for e in ext_names] if ext_names else [])

    def wf_lin(Xm, y, ok, fnames, window=3):
        idxf = fnames
        p = np.full(len(rows), np.nan); base = np.full(len(rows), np.nan)
        for Y in yrs:
            lo = day_of(f'{Y - window}-01-01')
            trm = (day < day_of(f'{Y}-01-01')) & (day >= lo) & ok
            tem = years == Y
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LinearLogitModel(idxf, names, GROUPS).fit(Xm[trm], FL[trm], gid[trm], y[trm])
            p[tem] = m1.predict_raw(Xm[tem], FL[tem], gid[tem]); base[tem] = float(y[trm].mean())
        return p, base

    def wf_gbm(Xm, y, ok, cols, window=3):
        Z = np.hstack([Xm[:, cols], FL, np.eye(len(GROUPS), dtype=np.float32)[gid]]).astype(np.float32)
        p = np.full(len(rows), np.nan); base = np.full(len(rows), np.nan)
        for Y in yrs:
            lo = day_of(f'{Y - window}-01-01')
            trm = (day < day_of(f'{Y}-01-01')) & (day >= lo) & ok
            tem = years == Y
            if trm.sum() < 3000 or not tem.any():
                continue
            g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=300,
                                               l2_regularization=10.0, early_stopping=False, random_state=0)
            g.fit(Z[trm], y[trm])
            p[tem] = g.predict_proba(Z[tem])[:, 1]; base[tem] = float(y[trm].mean())
        return p, base

    def pick(p, base, ok, allowed, N, extra_mask=None):
        cand = ok & np.isfinite(p) & (p >= base) & np.isin(gid, allowed)
        if extra_mask is not None:
            cand &= extra_mask
        return topn_mask(p, day, cand, N)

    def ymeans(sel, pnl):
        return {Y: float(np.nanmean(pnl[sel & (years == Y)])) for Y in yrs if (sel & (years == Y)).sum() >= 20}

    def line(tag, sel, pnl, base_ym=None, extra=''):
        st = stats(pnl[sel], day[sel], sym[sel])
        if st.get('n', 0) == 0:
            print(f'   {tag:40s} —'); return None
        ym = ymeans(sel, pnl)
        dev = stats(pnl[sel & (years <= 2023)]); rec = stats(pnl[sel & (years >= 2024)])
        cmp_ = ''
        if base_ym is not None:
            both = [Y for Y in ym if Y in base_ym]
            cmp_ = f" | mejor que base {sum(ym[Y] > base_ym[Y] for Y in both)}/{len(both)} años"
        print(f"   {tag:40s} n={st['n']:5d} WR={st['wr']:4.1f}% μ={st['mean']:+5.2f}% PF={st['pf']:4.2f} t={st['t_day']:3.1f} | ≤2023 {dev.get('mean', float('nan')):+5.2f} · 2024+ {rec.get('mean', float('nan')):+5.2f} · peor año {min(ym.values()):+5.2f}{cmp_}{extra}")
        return ym

    for plan in PLANS:
        vi = variants.index(plan['id'])
        N = int(plan.get('n', 5))
        pnl = EV['pnl'][rows, vi]; bars = EV['bars'][rows, vi]
        ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        allowed = [GROUPS.index(g) for g in groups_of[plan['id']]]
        elig_g = np.isin(gid, allowed)
        base_cols = fi(MD.MODEL_FEATURES)
        p0, b0 = wf_lin(X, y, ok, base_cols)
        sel0 = pick(p0, b0, ok, allowed, N)
        elig = ok & np.isfinite(p0) & elig_g
        print(f"\n================ {plan['id']} · {plan['label']} (N={N}) · grupos {groups_of[plan['id']]} ================")

        # ── S1: calibración ─────────────────────────────────────────────
        print('  [S1] AUDITORÍA DE CALIBRACIÓN (todos los candidatos de los grupos del plan, fuera de muestra)')
        auc_y = {Y: MD.auc(p0[elig & (years == Y)], y[elig & (years == Y)]) for Y in yrs if (elig & (years == Y)).sum() > 500}
        print(f"     AUC global {MD.auc(p0[elig], y[elig]):.3f} (0,5 = azar) · por año: " + ' '.join(f"{str(Y)[2:]}:{v:.3f}" for Y, v in auc_y.items()))
        pc = np.full(len(rows), np.nan)
        for Y in yrs:
            trn = elig & (years < Y)
            tem = elig & (years == Y)
            if trn.sum() >= 5000 and tem.any():
                pc[tem] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(p0[trn], y[trn]).predict(p0[tem])
        mm = elig & np.isfinite(pc)
        bb = np.mean((b0[mm] - y[mm]) ** 2); br = np.mean((p0[mm] - y[mm]) ** 2); bn = np.mean((pc[mm] - y[mm]) ** 2)
        print(f"     Brier: tasa base {bb:.4f} · probabilidad cruda {br:.4f} ({(1 - br / bb) * 100:+.1f}% vs base) · calibrada SOLO con años previos {bn:.4f} ({(1 - bn / bb) * 100:+.1f}% vs base)")
        qs = np.quantile(p0[mm], np.linspace(0, 1, 11))
        print('     fiabilidad por décima de probabilidad cruda (predicho → acierto real):')
        for a_, b_ in zip(qs[:-1], qs[1:]):
            m_ = mm & (p0 >= a_) & (p0 <= b_)
            e1, e2 = m_ & (years <= 2022), m_ & (years >= 2023)
            print(f"       p∈[{a_:.2f},{b_:.2f}] n={int(m_.sum()):6d} cruda {p0[m_].mean() * 100:4.1f}% · calibrada-previa {pc[m_].mean() * 100:4.1f}% → real {y[m_].mean() * 100:4.1f}%  (≤2022: {y[e1].mean() * 100:4.1f}% · 2023+: {y[e2].mean() * 100:4.1f}%)")
        iso_all = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(p0[elig], y[elig])
        pdsp = np.full(len(rows), np.nan)
        fin_ = np.isfinite(p0)
        pdsp[fin_] = iso_all.predict(p0[fin_])
        print('     SELECCIONADAS (política desplegada): probabilidad mostrada vs acierto real por año')
        err = []
        for Y in yrs:
            m_ = sel0 & (years == Y)
            if m_.sum() >= 30:
                lo_, hi_ = _wilson(int(y[m_].sum()), int(m_.sum()))
                e = pdsp[m_].mean() - y[m_].mean(); err.append(e)
                nested = f"{np.nanmean(pc[m_]) * 100:4.1f}%" if np.isfinite(pc[m_]).any() else '  — '
                print(f"       {Y}: n={int(m_.sum()):5d} cruda {p0[m_].mean() * 100:4.1f}% · mostrada {pdsp[m_].mean() * 100:4.1f}% · calibrada-previa {nested} → real {y[m_].mean() * 100:4.1f}% (IC95 {lo_ * 100:.0f}–{hi_ * 100:.0f})  error mostrada−real {e * 100:+.1f}")
        print(f"       error medio absoluto de la probabilidad mostrada: {np.mean(np.abs(err)) * 100:.1f} puntos · peor año {max(err, key=abs) * 100:+.1f}")

        # ── S2: motivos de compra (patrones) ─────────────────────────────
        print('  [S2] AUDITORÍA DE LOS MOTIVOS DE COMPRA (patrones). «Δ» = media del patrón − media de todos los candidatos del plan')
        mu_all = np.nanmean(pnl[elig])
        print(f"     todos los candidatos: n={int(elig.sum()):6d} WR={np.mean(y[elig]) * 100:4.1f}% μ={mu_all:+5.2f}%")
        for k, nm in enumerate(names):
            m_ = elig & (FL[:, k] > 0)
            if m_.sum() < 100:
                continue
            st = stats(pnl[m_], day[m_], sym[m_]); ym = ymeans(m_, pnl)
            ms = sel0 & (FL[:, k] > 0)
            print(f"     {SU.BY_ID[nm].label[:52]:52s} n={st['n']:6d} WR={st['wr']:4.1f}% μ={st['mean']:+5.2f}% (Δ {st['mean'] - mu_all:+5.2f}) años+ {sum(v > 0 for v in ym.values())}/{len(ym)} | elegidas por el modelo n={int(ms.sum()):5d} μ={np.nanmean(pnl[ms]) if ms.any() else float('nan'):+5.2f}%")
        nfl = FL.sum(1)
        print('     CONFLUENCIA (nº de patrones que saltan a la vez):')
        for lo_, hi_ in ((1, 1), (2, 2), (3, 3), (4, 99)):
            m_ = elig & (nfl >= lo_) & (nfl <= hi_)
            ms = sel0 & (nfl >= lo_) & (nfl <= hi_)
            if m_.sum() >= 100:
                st = stats(pnl[m_], day[m_], sym[m_])
                print(f"       {lo_}{'+' if hi_ == 99 else ('' if lo_ == hi_ else '-' + str(hi_))} patrón(es): n={st['n']:6d} WR={st['wr']:4.1f}% μ={st['mean']:+5.2f}% | elegidas n={int(ms.sum()):5d} μ={np.nanmean(pnl[ms]) if ms.any() else float('nan'):+5.2f}%")

        # ── S5: variables que separan de forma robusta (univariante) ────
        print('  [S5] VARIABLES QUE SEPARAN DE FORMA ROBUSTA (candidatos del plan; correlación de rangos con el resultado, calculada por año)')
        import pandas as pd
        cors = []
        for j, fname in enumerate(FEATURES):
            if fname.startswith('vp_'):
                continue
            xj = X[:, j]
            per = []
            for Y in yrs:
                m_ = elig & (years == Y) & np.isfinite(xj)
                if m_.sum() >= 300:
                    per.append(float(pd.Series(xj[m_]).rank().corr(pd.Series(pnl[m_]).rank())))
            if len(per) >= 5:
                mc = float(np.mean(per))
                cors.append((fname, mc, sum(np.sign(v) == np.sign(mc) for v in per), len(per)))
        cors.sort(key=lambda r: -abs(r[1]) * (r[2] / r[3]))
        for fname, mc, same, n_ in cors[:14]:
            xj = X[:, FEATURES.index(fname)]
            mk = elig & np.isfinite(xj)
            qe = np.quantile(xj[mk], [0.2, 0.4, 0.6, 0.8])
            bk = np.digitize(xj, qe)
            cells = ' '.join(f"Q{b + 1}:{np.nanmean(pnl[mk & (bk == b)]):+.2f}/{np.mean(y[mk & (bk == b)]) * 100:.0f}%" for b in range(5))
            print(f"     {fname:12s} corr {mc:+.3f} (mismo signo {same}/{n_} años) · cuantiles (μ/acierto): {cells}   [Q1={qe[0]:.2f} … Q5>{qe[3]:.2f}]")
        print('     (las variables con correlación ≈ 0 o signo cambiante no aportan información estable)')

        # ── S3: nueva información ───────────────────────────────────────
        print('  [S3] NUEVA INFORMACIÓN añadida al modelo lineal (AUC OOS entre paréntesis; base ' + f"{MD.auc(p0[elig], y[elig]):.3f})")
        base_ym = line('BASE (desplegado)', sel0, pnl)
        ext = build_ext(pnl, bars)
        ext_names = list(ext.keys())
        Xe = np.hstack([X, np.column_stack([ext[k] for k in ext_names]).astype(np.float32)])
        ext_idx = {k: nF + i for i, k in enumerate(ext_names)}
        fams = {'tendencia a revertir del valor': ['sym_hist', 'sym_wr'],
                'régimen de reversión del mercado': ['mkt30', 'mkt90'],
                'valor + mercado': ['sym_hist', 'sym_wr', 'mkt30', 'mkt90']}
        if sec_cols:
            fams['caída frente al sector'] = ['rel_sec5', 'sec_ret5', 'sec_os', 'sec_ret1']
            fams['todo'] = ['sym_hist', 'sym_wr', 'mkt30', 'mkt90', 'rel_sec5', 'sec_ret5', 'sec_os', 'sec_ret1']
        best = {}
        for fn, fl in fams.items():
            cols = base_cols + [ext_idx[e] for e in fl]
            pw, bw = wf_lin(Xe, y, ok, cols)
            ew = ok & np.isfinite(pw) & elig_g
            selw = pick(pw, bw, ok, allowed, N)
            line(f"+ {fn} ({MD.auc(pw[ew], y[ew]):.3f})", selw, pnl, base_ym)
            best[fn] = (pw, bw, cols)
        print('     réplica con salidas vecinas (μ de la política; entre paréntesis, años mejores que la base):')
        nb = {'rsi_S4_H10': ['rsi_S2.5_H10', 'rsi_S4_H5', 'rsi_S1.5_H10'],
              'atr_T1_S4_H10': ['atr_T0.5_S4_H10', 'atr_T1.5_S4_H10', 'atr_T1_S2.5_H10', 'atr_T1_S4_H5']}[plan['id']]
        for vn in nb:
            vj = variants.index(vn)
            pn, bn_ = EV['pnl'][rows, vj], EV['bars'][rows, vj]
            okj = np.isfinite(pn); yj = (pn > 0).astype(np.float32)
            exj = build_ext(pn, bn_)
            Xj = np.hstack([X, np.column_stack([exj[k] for k in ext_names]).astype(np.float32)])
            p_b, b_b = wf_lin(Xj, yj, okj, base_cols)
            s_b = pick(p_b, b_b, okj, allowed, N); ym_b = ymeans(s_b, pn)
            out = [f"base {np.nanmean(pn[s_b]):+.2f}%"]
            for fn, fl in fams.items():
                cols = base_cols + [ext_idx[e] for e in fl]
                pw, bw = wf_lin(Xj, yj, okj, cols)
                sw = pick(pw, bw, okj, allowed, N); ym = ymeans(sw, pn)
                out.append(f"{fn[:14]} {np.nanmean(pn[sw]):+.2f}% ({sum(ym[Y] > ym_b[Y] for Y in ym if Y in ym_b)}/{len(ym)})")
            print(f"       {vn:16s} " + ' | '.join(out), flush=True)

        # ── S4: modelos alternativos y confluencia ──────────────────────
        print('  [S4] MODELOS ALTERNATIVOS Y CONFLUENCIA')
        pg, bg = wf_gbm(X, y, ok, base_cols)
        eg = ok & np.isfinite(pg) & elig_g
        print(f"     AUC gradient boosting {MD.auc(pg[eg], y[eg]):.3f} · lineal {MD.auc(p0[elig], y[elig]):.3f}")
        line('boosting (variables base)', pick(pg, bg, ok, allowed, N), pnl, base_ym)
        mixed_cols = best['valor + mercado'][2]
        pg2, bg2 = wf_gbm(Xe, y, ok, mixed_cols)
        eg2 = ok & np.isfinite(pg2) & elig_g
        print(f"     AUC boosting con variables nuevas {MD.auc(pg2[eg2], y[eg2]):.3f}")
        line('boosting (+ valor y mercado)', pick(pg2, bg2, ok, allowed, N), pnl, base_ym)
        pl2, bl2, _ = best['valor + mercado']
        pbl, bbl = 0.5 * (p0 + pg), 0.5 * (b0 + bg)
        line('mezcla lineal+boosting', pick(pbl, bbl, ok, allowed, N), pnl, base_ym)
        pbl2, bbl2 = 0.5 * (pl2 + pg2), 0.5 * (bl2 + bg2)
        line('mezcla con variables nuevas', pick(pbl2, bbl2, ok, allowed, N), pnl, base_ym)
        agree = np.isfinite(pg) & (pg >= bg) & (p0 >= b0)
        line('CONFLUENCIA: ambos ≥ tasa base', pick(pbl, bbl, ok, allowed, N, extra_mask=agree), pnl, base_ym)
        for q in (0.6, 0.75, 0.9):
            thr_l = np.nanquantile(p0[elig & (years >= 2019)], q); thr_g = np.nanquantile(pg[eg], q)
            line(f'CONFLUENCIA: ambos en el {int(round((1 - q) * 100))}% superior', pick(pbl, bbl, ok, allowed, N, extra_mask=(p0 >= thr_l) & (pg >= thr_g)), pnl, base_ym)
        sys.stdout.flush()


def _wilson(w: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


# ═════════════════════════════════════════════════════════════════════════
#  Segunda tanda (comando `reasons2`): ¿se puede elegir mejor con reglas simples, retorno esperado o información de crédito/tipos?
# ═════════════════════════════════════════════════════════════════════════
def _asof(t_days: np.ndarray, v: np.ndarray, ev_days: np.ndarray) -> np.ndarray:
    j = np.searchsorted(t_days, ev_days, side='right') - 1
    return np.where(j >= 0, v[np.clip(j, 0, len(v) - 1)], np.nan)


def run2(EV: Dict, out_dir: str, data: Dict | None = None, uni: Dict | None = None):
    import pandas as pd
    import model as MD
    from research import portfolio_sim

    here = os.path.dirname(os.path.abspath(__file__))
    reg = json.load(open(os.path.join(here, 'model', 'validated.json')))
    groups_of = {s['id']: s['groups'] for s in reg['strategies']}
    variants = EV['variants']
    rows, FL, names = union_events(EV, +1)
    day, sym, gid = EV['day'][rows], EV['sym'][rows], EV['grp'][rows]
    years = year_of(day)
    X = EV['X'][rows]
    yrs = [Y for Y in sorted(set(years)) if Y >= 2019]
    nF = len(FEATURES)
    spy = data.get('SPY') if data else None
    cal_d = (spy['t'] // 86400).astype(np.int64) if spy is not None else None
    print(f'\n######## REASONS2 · {len(rows):,} candidatos largos · walk-forward {yrs[0]}–{yrs[-1]}', flush=True)

    # variables de crédito, tipos y dólar (ETF líquidos; cada una con su último dato conocido en la fecha de la señal)
    cr = {}
    if data is not None:
        for tk, nm in (('HYG', 'hyg'), ('LQD', 'lqd'), ('TLT', 'tlt'), ('UUP', 'uup')):
            if tk in data:
                b = data[tk]; td = (b['t'] // 86400).astype(np.int64); c = pd.Series(b['c'])
                if nm == 'hyg':
                    cr['hyg_ret5'] = _asof(td, (c / c.shift(5) - 1).to_numpy() * 100, day)
                    cr['hyg_dist50'] = _asof(td, (c / c.rolling(50, min_periods=30).mean() - 1).to_numpy() * 100, day)
                    cr['hyg_dd60'] = _asof(td, (c / c.rolling(60, min_periods=30).max() - 1).to_numpy() * 100, day)
                elif nm == 'tlt':
                    cr['tlt_ret20'] = _asof(td, (c / c.shift(20) - 1).to_numpy() * 100, day)
                elif nm == 'uup':
                    cr['uup_ret20'] = _asof(td, (c / c.shift(20) - 1).to_numpy() * 100, day)
                elif nm == 'lqd' and 'HYG' in data:
                    h = pd.Series(data['HYG']['c']).reindex(range(len(c))).to_numpy() if len(data['HYG']['c']) == len(c) else None
        if 'HYG' in data and 'LQD' in data:     # diferencial relativo crédito basura / grado de inversión (proxy de estrés de crédito)
            bh, bl = data['HYG'], data['LQD']
            th = (bh['t'] // 86400).astype(np.int64); tl = (bl['t'] // 86400).astype(np.int64)
            common = np.intersect1d(th, tl)
            ch = pd.Series(bh['c'][np.isin(th, common)]); cl = pd.Series(bl['c'][np.isin(tl, common)])
            ratio = (ch / cl).to_numpy()
            rz = (pd.Series(ratio) - pd.Series(ratio).rolling(120, min_periods=60).mean()) / pd.Series(ratio).rolling(120, min_periods=60).std()
            cr['credit_z'] = _asof(common, rz.to_numpy(), day)
    print(f'  variables de crédito/tipos/dólar disponibles: {list(cr)}', flush=True)
    cr_names = list(cr.keys())
    Xe = np.hstack([X, np.column_stack([cr[k] for k in cr_names]).astype(np.float32)]) if cr_names else X
    cr_idx = {k: nF + i for i, k in enumerate(cr_names)}

    def wf_lin(Xm, y, ok, cols, window=3):
        p = np.full(len(rows), np.nan); base = np.full(len(rows), np.nan)
        for Y in yrs:
            trm = (day < day_of(f'{Y}-01-01')) & (day >= day_of(f'{Y - window}-01-01')) & ok
            tem = years == Y
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.LinearLogitModel(cols, names, GROUPS).fit(Xm[trm], FL[trm], gid[trm], y[trm])
            p[tem] = m1.predict_raw(Xm[tem], FL[tem], gid[tem]); base[tem] = float(y[trm].mean())
        return p, base

    def wf_ridge(Xm, target, ok, cols, window=3):
        pr = np.full(len(rows), np.nan)
        for Y in yrs:
            trm = (day < day_of(f'{Y}-01-01')) & (day >= day_of(f'{Y - window}-01-01')) & ok
            tem = years == Y
            if trm.sum() < 3000 or not tem.any():
                continue
            m1 = MD.RidgeModel(cols, names, GROUPS).fit(Xm[trm], FL[trm], gid[trm], target[trm])
            pr[tem] = m1.predict_raw(Xm[tem], FL[tem], gid[tem])
        return pr

    def ymeans(sel, pnl):
        return {Y: float(np.nanmean(pnl[sel & (years == Y)])) for Y in yrs if (sel & (years == Y)).sum() >= 20}

    def pct(v):
        s = pd.Series(v).rank(pct=True).to_numpy()
        return np.where(np.isfinite(v), s, 0.5)

    for plan in PLANS:
        vi = variants.index(plan['id'])
        N = int(plan.get('n', 5))
        S_ = SM.default_variants()[vi].S
        pnl = EV['pnl'][rows, vi]; bars = EV['bars'][rows, vi]
        ok = np.isfinite(pnl)
        y = (pnl > 0).astype(np.float32)
        allowed = [GROUPS.index(g) for g in groups_of[plan['id']]]
        elig_g = np.isin(gid, allowed)
        base_cols = [FEATURES.index(f) for f in MD.MODEL_FEATURES]
        atrp = np.maximum(X[:, FEATURES.index('atrp')], 0.5)
        p0, b0 = wf_lin(X, y, ok, base_cols)
        in_ = ok & elig_g & (years >= 2019)
        sel0 = topn_mask(p0, day, in_ & np.isfinite(p0) & (p0 >= b0), N)
        print(f"\n================ {plan['id']} · {plan['label']} (N={N}) ================")

        def report(tag, sel, score, base_ym=None):
            st = stats(pnl[sel], day[sel], sym[sel])
            ym = ymeans(sel, pnl)
            rr = pnl[sel] / (S_ * atrp[sel])
            w = 1.0 / atrp[sel]
            muw = float(np.nansum(w * pnl[sel]) / np.nansum(w))
            cmp_ = ''
            if base_ym is not None:
                both = [Y for Y in ym if Y in base_ym]
                cmp_ = f" | mejor que desplegado {sum(ym[Y] > base_ym[Y] for Y in both)}/{len(both)}"
            port = ''
            if cal_d is not None:
                rk = rank_in_day(score, day, sel)
                ti = np.searchsorted(cal_d, day, side='left')
                o = portfolio_sim(cal_d, ti, bars, np.where(sel, pnl, np.nan), rk, 10, 0.10)
                port = f" | cartera 10×10%: {o['cagr']:+5.1f}% DD {o['dd']:4.1f}% Sharpe {o['sharpe']:.2f}"
            dev, rec = stats(pnl[sel & (years <= 2023)]), stats(pnl[sel & (years >= 2024)])
            print(f"   {tag:44s} n={st['n']:5d} WR={st['wr']:4.1f}% μ={st['mean']:+5.2f}% t={st['t_day']:3.1f} R̄={np.nanmean(rr):+.3f} μ(riesgo cte)={muw:+.2f} | ≤2023 {dev.get('mean', float('nan')):+5.2f} · 2024+ {rec.get('mean', float('nan')):+5.2f}{cmp_}{port}", flush=True)
            return ym

        print('  [R1] ¿Basta una REGLA SIMPLE y transparente? (se eligen las N mejores del día entre TODOS los candidatos del plan, sin modelo ni umbral)')
        base_ym = report('DESPLEGADO (modelo, N mejores con p ≥ tasa base)', sel0, p0)
        rnd = np.random.default_rng(1).random(len(rows))
        report('al azar (N por día)', topn_mask(rnd, day, in_, N), rnd, base_ym)
        d20, d52, r1, r3 = pct(-X[:, FEATURES.index('dd20')]), pct(-X[:, FEATURES.index('dd52')]), pct(-X[:, FEATURES.index('ret1')]), pct(-X[:, FEATURES.index('ret3')])
        pa, pv = pct(atrp), pct(X[:, FEATURES.index('vix')])
        rules = {'volatilidad (ATR%) + caída desde máx. 20d': pa + d20,
                 'volatilidad + caída 20d + caída 52 sem.': pa + d20 + d52,
                 'volatilidad + caída 3 sesiones': pa + r3,
                 'volatilidad + caída 20d + VIX': pa + d20 + pv,
                 'solo caída 20d': d20, 'solo volatilidad': pa}
        for rn, sc in rules.items():
            report(rn, topn_mask(sc, day, in_, N), sc, base_ym)
        print('  [R2] ¿Mejor ordenar por RETORNO ESPERADO que por probabilidad de ganar? (regresión ridge con ventana móvil de 3 años)')
        pr = wf_ridge(X, np.clip(pnl, -15, 15), ok, base_cols)
        report('ridge sobre el resultado %', topn_mask(pr, day, in_ & np.isfinite(pr) & (pr > 0), N), pr, base_ym)
        rmul = np.clip(pnl / (S_ * atrp), -1.5, 1.5)
        prr = wf_ridge(X, rmul, ok, base_cols)
        report('ridge sobre el resultado en R (por riesgo)', topn_mask(prr, day, in_ & np.isfinite(prr) & (prr > 0), N), prr, base_ym)
        mix = pct(np.nan_to_num(p0, nan=np.nanmedian(p0))) + pct(np.nan_to_num(pr, nan=0.0))
        report('mezcla probabilidad + retorno esperado', topn_mask(mix, day, in_ & np.isfinite(p0) & np.isfinite(pr) & (p0 >= b0) & (pr > 0), N), mix, base_ym)

        print('  [R3] RETORNO POR UNIDAD DE RIESGO según la volatilidad del valor (candidatos del plan, R = resultado / (stop en %)):')
        qe = np.quantile(atrp[in_], [0.2, 0.4, 0.6, 0.8])
        bk = np.digitize(atrp, qe)
        for b in range(5):
            m_ = in_ & (bk == b)
            print(f"     ATR% quintil {b + 1} (≈{np.nanmean(atrp[m_]):.1f}%): n={int(m_.sum()):6d} WR={np.mean(y[m_]) * 100:4.1f}% μ={np.nanmean(pnl[m_]):+5.2f}% R̄={np.nanmean(pnl[m_] / (S_ * atrp[m_])):+.3f}")

        if cr_names:
            print('  [R4] INFORMACIÓN DE CRÉDITO / TIPOS / DÓLAR añadida al modelo lineal')
            fams = {'crédito (HYG, diferencial)': [k for k in cr_names if k.startswith('hyg') or k == 'credit_z'],
                    'tipos y dólar (TLT, UUP)': [k for k in cr_names if k in ('tlt_ret20', 'uup_ret20')], 'todo el bloque': cr_names}
            for fn, fl in fams.items():
                if not fl:
                    continue
                cols = base_cols + [cr_idx[k] for k in fl]
                pw, bw = wf_lin(Xe, y, ok, cols)
                ew = ok & np.isfinite(pw) & elig_g
                selw = topn_mask(pw, day, in_ & np.isfinite(pw) & (pw >= bw), N)
                report(f"+ {fn} (AUC {MD.auc(pw[ew], y[ew]):.3f})", selw, pw, base_ym)

        print('  [R5] ¿Ayuda quedarse solo con los patrones que HAN FUNCIONADO hasta cada año? (elección anidada: cada año usa solo años previos)')
        nsel = np.zeros(len(rows), bool)
        for Y in yrs:
            if Y < 2021:
                continue
            tr_ = in_ & (years < Y)
            mu_all = np.nanmean(pnl[tr_])
            keep = [k for k in range(len(names)) if (tr_ & (FL[:, k] > 0)).sum() >= 200 and np.nanmean(pnl[tr_ & (FL[:, k] > 0)]) > mu_all]
            has = (FL[:, keep] > 0).any(1) if keep else np.zeros(len(rows), bool)
            nsel |= in_ & (years == Y) & has & np.isfinite(p0) & (p0 >= b0)
        selr = topn_mask(p0, day, nsel, N)
        sel_cmp = sel0 & (years >= 2021)
        base_ym2 = {Y: v for Y, v in base_ym.items() if Y >= 2021}
        report('desplegado (solo 2021+, comparable)', sel_cmp, p0, None)
        report('solo patrones que ya funcionaban', selr, p0, base_ym2)
        sys.stdout.flush()
