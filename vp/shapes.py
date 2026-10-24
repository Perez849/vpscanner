"""
Motor geométrico de figuras (causal): reconoce las figuras como lo haría un analista, no como «dos pivotes unidos por una recta».

  · Oscilaciones (swings) alternadas con zigzag de ATR: un pivote solo cuenta si la oscilación mide ≥ AMP·ATR. Se confirma K barras después de formarse.
  · Rectas ajustadas: entre todos los pares de pivotes se elige la recta con más TOQUES (pivotes a ≤ 0,35 ATR) que el precio respeta (ni los cierres la cruzan, ni las
    mechas la superan más de 0,6 ATR), que pasa por uno de los dos últimos pivotes y no queda lejos del precio actual.
  · Clasificación por pendientes y convergencia: triángulos, cuñas y rectángulo, con ápice y límites de duración y de altura de la base.
  · H-C-H y dobles/triples con criterios de forma, taza que abraza los mínimos y es redondeada, banderas con asta impulsiva, VCP, diamante, isla, directrices con ≥ 3 toques.
  · Cada figura recibe una CALIDAD 0–100 (toques, ajuste, duración, forma, ruptura, volumen); por debajo de QMIN no se considera figura.
  · Las figuras que aún NO han roto (compresiones, dobles/triples, H-C-H, banderas, taza, VCP) se devuelven aparte («en formación») con su disparador y su distancia en ATR.

Todo es causal: la decisión en la barra i solo usa datos hasta i (los pivotes se conocen K barras después).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from .ta import shift

K = 3                 # un pivote en j se confirma en j + K
AMP = 1.0             # amplitud mínima (ATR) de una oscilación
QMIN = 35             # calidad mínima para contar como figura
TOL = 0.35            # un pivote «toca» la recta si está a ≤ TOL·ATR
WICK = 0.6            # las mechas pueden superar la recta hasta WICK·ATR
CLOSE = 0.15          # margen (ATR) de los cierres sobre la recta
MAXLEN = 80           # duración máxima de una compresión (barras)
MAXBASE = 0.16        # altura máxima de la base de una compresión (fracción del precio)
MAXBASE_ATR = 9.0     # … y en ATR
NEAR = 3.0            # una figura «se forma» cuando el cierre está a ≤ NEAR ATR de su disparador
QFORM = 50            # calidad mínima de una figura en formación (sin ruptura ni volumen que puntuar)


# ── utilidades ────────────────────────────────────────────────────────────
def pivot_flags(h: np.ndarray, l: np.ndarray, k: int = K) -> Tuple[np.ndarray, np.ndarray]:
    n = len(h)
    ph, pl = np.zeros(n, bool), np.zeros(n, bool)
    if n > 2 * k + 1:
        ph[k:n - k] = np.argmax(swv(h, 2 * k + 1), axis=1) == k
        pl[k:n - k] = np.argmin(swv(l, 2 * k + 1), axis=1) == k
    return ph, pl


def _add(S: List[list], j: int, v: float, t: int, amp: float) -> bool:
    """Añade un pivote manteniendo la alternancia máximo/mínimo y la amplitud mínima. Devuelve True si cambió la lista."""
    if not S:
        S.append([j, v, t])
        return True
    la = S[-1]
    if la[2] == t:
        if (t > 0 and v >= la[1]) or (t < 0 and v <= la[1]):
            S[-1] = [j, v, t]
            return True
        return False
    if abs(v - la[1]) >= amp:
        S.append([j, v, t])
        return True
    return False


def at(L: Tuple[float, float, float], x: float) -> float:
    """Valor de la recta L = (x0, y0, pendiente) en x."""
    return L[1] + L[2] * (x - L[0])


def best_line(P: List[Tuple[int, float]], side: int, h, l, c, a: float, upto: int, min_touch: int = 2, max_back: int = 8, near: float = 6.0, recent: int = 40) -> Optional[Dict]:
    """Mejor recta por los pivotes P (máximos si side=+1: resistencia; mínimos si side=-1: soporte) que el precio respeta hasta la barra upto-1."""
    P = P[-max_back:]
    arr = h if side > 0 else l
    best = None
    last2 = {j for j, _ in P[-2:]}
    for x in range(len(P)):
        for y in range(x + 1, len(P)):
            (xa, va), (xb, vb) = P[x], P[y]
            if xb - xa < 6:
                continue
            s = (vb - va) / (xb - xa)
            xs = np.arange(xa, upto)
            ln = va + s * (xs - xa)
            if ((arr[xa:upto] - ln) * side).max() > WICK * a:
                continue
            if ((c[xa:upto] - ln) * side).max() > CLOSE * a:
                continue
            if abs(va + s * (upto - 1 - xa) - c[upto - 1]) > near * a:            # una recta lejana del precio actual no es la frontera de ninguna figura
                continue
            tch = [(j, v) for j, v in P if j >= xa and abs(v - (va + s * (j - xa))) <= TOL * a]
            if len(tch) < min_touch or upto - tch[-1][0] > recent or not any(j in last2 for j, _ in tch):      # la recta debe pasar por uno de los dos últimos pivotes
                continue
            dev = float(np.mean([abs(v - (va + s * (j - xa))) for j, v in tch])) / a
            span = tch[-1][0] - tch[0][0]
            sc = len(tch) * 10 - 4.0 * dev + 3.0 * (1.0 - min(upto - tch[-1][0], 40) / 40.0)
            if best is None or sc > best['sc']:
                best = {'L': (xa, va, s), 'tch': tch, 'sc': sc, 'dev': dev, 'span': span}
    return best


# ── compresiones: triángulos, cuñas, rectángulo ───────────────────────────
def compression(S: List[list], i: int, h, l, c, atr) -> Optional[Dict]:
    """Mejor estructura de compresión con los datos hasta la barra i-1 (la barra i puede ser la de ruptura)."""
    a = float(atr[i - 1]) if i > 0 else 0.0
    if not a > 0:
        return None
    SW = [s for s in S[-12:] if i - s[0] <= MAXLEN]
    H = [(j, v) for j, v, t in SW if t > 0]
    Lw = [(j, v) for j, v, t in SW if t < 0]
    if len(H) < 2 or len(Lw) < 2:
        return None
    U = best_line(H, 1, h, l, c, a, i)
    D = best_line(Lw, -1, h, l, c, a, i)
    if U is None or D is None:
        return None
    x0 = min(U['tch'][0][0], D['tch'][0][0])
    xl = max(U['tch'][-1][0], D['tch'][-1][0])
    xe = i - 1
    if xe - x0 < 12 or i - xl > 40:
        return None
    w0 = at(U['L'], x0) - at(D['L'], x0)
    w1 = at(U['L'], xe) - at(D['L'], xe)
    if w0 <= 0.5 * a or w1 <= 0.15 * a or w0 > min(MAXBASE * float(c[xe]), MAXBASE_ATR * a):      # una base enorme es un canal, no una figura
        return None
    xs = np.arange(x0, xe + 1)
    if (at(U['L'], xs) - at(D['L'], xs)).min() <= 0.15 * a:             # las rectas no se cruzan dentro de la figura
        return None
    du = U['L'][2] * (xe - x0) / a                                        # deriva de cada recta en ATR
    dl = D['L'][2] * (xe - x0) / a
    conv = w1 / w0
    su, sl = U['L'][2], D['L'][2]
    apex = x0 + w0 / (sl - su) if sl - su > 1e-12 else None
    FL, UP, DN = 1.3, 1.6, 1.6
    kind = None
    nt_ = len(U['tch']) + len(D['tch'])
    if conv <= 0.7 and apex is not None and apex > xe:
        if abs(du) <= FL and dl >= UP:
            kind = 'asc_triangle'
        elif abs(dl) <= FL and du <= -DN:
            kind = 'desc_triangle'
        elif du <= -DN and dl >= UP:
            kind = 'sym_tri'
        elif du >= FL and dl >= UP and dl > du and nt_ >= 5:
            kind = 'rising_wedge'
        elif du <= -DN and dl <= -FL and -du > -dl and nt_ >= 5:
            kind = 'falling_wedge'
    if kind is None and abs(du) <= FL and abs(dl) <= FL and w0 >= 2.5 * a and 0.65 <= conv <= 1.35 and nt_ >= 5:
        kind = 'rect'
    if kind is None and nt_ >= 6 and w0 >= 2.5 * a and 0.75 <= conv <= 1.3 and xe - x0 >= 20 and abs(du - dl) <= 0.35 * max(abs(du), abs(dl)):     # canal: rectas paralelas con pendiente
        kind = 'channel_up' if du >= FL and dl >= FL else 'channel_dn' if du <= -FL and dl <= -FL else None
    if kind is None:
        return None
    return {'kind': kind, 'U': U['L'], 'D': D['L'], 'tu': U['tch'], 'tl': D['tch'], 'x0': int(x0), 'xl': int(xl), 'apex': None if apex is None else float(apex),
            'dev': (U['dev'] + D['dev']) / 2.0, 'conv': float(conv), 'w0': float(w0), 'du': float(du), 'dl': float(dl), 'key': (int(U['tch'][0][0]), int(U['tch'][-1][0]), int(D['tch'][0][0]), int(D['tch'][-1][0]))}


def compression_quality(st: Dict, brk_margin: float, vol_ratio: float) -> int:
    nt = min(len(st['tu']) + len(st['tl']), 7)
    q = {2: 4, 3: 8, 4: 14, 5: 20, 6: 25, 7: 30}.get(nt, 4)
    q += 20.0 * max(0.0, 1.0 - st['dev'] / TOL)
    ln = st['xl'] - st['x0']
    q += 12 if 20 <= ln <= 80 else 6 if 12 <= ln <= 120 else 2
    q += 12 if st['conv'] <= 0.5 else 8 if st['conv'] <= 0.7 else 4
    q += 10 if brk_margin >= 0.5 else 6 if brk_margin >= 0.25 else 3
    q += 10 if vol_ratio >= 1.5 else 6 if vol_ratio >= 1.15 else 2
    return int(round(min(q, 100)))


def _vol_ratio(v: np.ndarray, i: int) -> float:
    base = v[max(i - 20, 0):i]
    m = float(np.mean(base)) if len(base) else 0.0
    return float(v[i] / m) if m > 0 else 1.0


# ── dobles, triples y hombro-cabeza-hombro (con la simetría de lo clásico) ──
def _view(d: int, h, l, c):
    """Orientación alcista: para d=+1 las series tal cual; para d=-1 se invierten (así un doble techo es un doble suelo)."""
    return (l, h, c) if d > 0 else (-h, -l, -c)


def _prior(hi: np.ndarray, j: int, base: float, back: int = 50) -> float:
    seg = hi[max(j - back, 0):j]
    return float(seg.max() - base) if len(seg) else 0.0


def _q_common(brk: float, vr: float) -> int:
    return (10 if brk >= 0.5 else 6 if brk >= 0.2 else 3) + (10 if vr >= 1.5 else 6 if vr >= 1.15 else 2)


def multi_bottom(S: List[list], i: int, d: int, h, l, c, atr, v, form: bool = False) -> Optional[Tuple[str, Dict, int]]:
    """Doble o triple suelo (d=+1) / techo (d=-1) cuyo cuello se rompe en la barra i (con form=True, i es la barra siguiente a la última y el cuello aún no se ha roto)."""
    lo, hi, cl = _view(d, h, l, c)
    a = float(atr[i - 1])
    tt = -d
    n = len(S)
    best = None
    for k in range(n - 1, 1, -1):
        if i - S[k][0] > 45:
            break
        if S[k][2] != tt or S[k - 1][2] != -tt or S[k - 2][2] != tt:
            continue
        for m in (3, 2):                                              # triple (3 mínimos) o doble (2)
            if k - 2 * (m - 1) < 0:
                continue
            idx = [k - 2 * q for q in range(m - 1, -1, -1)]
            if any(S[x][2] != tt for x in idx):
                continue
            pv = [d * S[x][1] for x in idx]
            js = [S[x][0] for x in idx]
            if js[-1] - js[0] < 8 or js[-1] - js[0] > (80 if m == 2 else 110):
                continue
            tol = max(0.025 * abs(min(pv)), 1.2 * a)
            if max(pv) - min(pv) > tol:
                continue
            neck = float(hi[js[0]:js[-1] + 1].max())
            if neck - max(pv) < 2.5 * a:
                continue
            if lo[js[-1]:i].min() < min(pv) - 0.5 * a:
                continue
            if form:
                if not (cl[i - 1] <= neck and neck - cl[i - 1] <= NEAR * a and cl[i - 1] - min(pv) >= a):       # el cuello aún no se ha roto y el precio ya se ha despegado de los mínimos
                    continue
            elif not (cl[i] > neck and cl[i - 1] <= neck):
                continue
            if cl[js[-1] + 1:i].max() > neck:                         # la ruptura cuenta la primera vez que el cierre pasa el cuello: un nuevo cruce tras un fallo no es la figura
                continue
            prior = _prior(hi, js[0], min(pv))
            if prior < 3.0 * a:
                continue
            brk, vr = (0.3, 1.2) if form else ((cl[i] - neck) / a, _vol_ratio(v, i))
            q = 15 + 20.0 * max(0.0, 1 - (max(pv) - min(pv)) / tol) + (15 if (neck - max(pv)) >= 5 * a else 10 if (neck - max(pv)) >= 3.5 * a else 6) \
                + (15 if 15 <= js[-1] - js[0] <= 70 else 8) + (15 if prior >= 6 * a else 10 if prior >= 4 * a else 6) + _q_common(brk, vr)
            if m == 3:
                q += 6
            jm = [S[x + 1][0] for x in idx[:-1]]
            g = dict(neck=d * neck, hgt=float(neck - min(pv)), pts=[(js[x], d * pv[x]) for x in range(m)], mids=[(jm[x], d * float(hi[jm[x]])) for x in range(m - 1)], q=int(min(q, 100)))
            if m == 2:
                g.update(p1=g['pts'][0], p2=g['pts'][1], mid=g['mids'][0])
            name = ('double_bottom' if d > 0 else 'double_top') if m == 2 else ('triple_bottom' if d > 0 else 'triple_top')
            if best is None or g['q'] > best[1]['q']:
                best = (name, g, js[0])
    return None if best is None else (best[0], best[1], best[2])


def head_shoulders(S: List[list], i: int, d: int, h, l, c, atr, v, form: bool = False) -> Optional[Tuple[str, Dict]]:
    """Hombro-cabeza-hombro (d=-1, techo) o invertido (d=+1, suelo) con ruptura del cuello en la barra i."""
    lo, hi, cl = _view(d, h, l, c)
    a = float(atr[i - 1])
    tt = -d
    n = len(S)
    best = None
    for k in range(n - 1, 3, -1):
        if i - S[k][0] > 35:
            break
        if S[k][2] != tt:
            continue
        ls, t1, hd, t2, rs = S[k - 4], S[k - 3], S[k - 2], S[k - 1], S[k]
        if ls[2] != tt or hd[2] != tt or t1[2] != -tt or t2[2] != -tt:
            continue
        pl_, ph_, pr_ = d * ls[1], d * hd[1], d * rs[1]
        k1, k2 = d * t1[1], d * t2[1]
        neck_mid = (k1 + k2) / 2.0
        depth = neck_mid - ph_
        if depth <= 0 or min(pl_, pr_) - ph_ < 1.2 * a or min(pl_, pr_) - ph_ < 0.12 * depth:
            continue
        if abs(pl_ - pr_) > 0.35 * depth or abs(k2 - k1) > 0.5 * depth:
            continue
        r = (hd[0] - ls[0]) / max(rs[0] - hd[0], 1)
        span = rs[0] - ls[0]
        if not (0.4 <= r <= 2.5 and 15 <= span <= 120):
            continue
        sl = (k2 - k1) / max(t2[0] - t1[0], 1)
        necki, neckp = k1 + sl * (i - t1[0]), k1 + sl * (i - 1 - t1[0])
        if form:
            if not (cl[i - 1] <= neckp and neckp - cl[i - 1] <= NEAR * a):
                continue
        elif not (cl[i] > necki and cl[i - 1] <= neckp):
            continue
        xs_ = np.arange(rs[0] + 1, i - (0 if form else 1))
        if len(xs_) and (cl[xs_] > k1 + sl * (xs_ - t1[0])).any():                  # el cuello se rompe la primera vez que el cierre lo pasa
            continue
        if lo[rs[0]:i].min() <= ph_:
            continue
        prior = _prior(hi, ls[0], pl_, 60)
        if prior < 3.0 * a:
            continue
        brk, vr = (0.3, 1.2) if form else ((cl[i] - necki) / a, _vol_ratio(v, i))
        q = 15 + 20.0 * min((min(pl_, pr_) - ph_) / (3.0 * a), 1.0) + 15.0 * max(0.0, 1 - abs(pl_ - pr_) / (0.35 * depth)) + 10.0 * max(0.0, 1 - abs(np.log(r)) / 0.9) \
            + 10.0 * max(0.0, 1 - abs(k2 - k1) / (0.5 * depth)) + (10 if prior >= 6 * a else 6) + _q_common(brk, vr)
        g = dict(ls=(ls[0], float(ls[1])), hd=(hd[0], float(hd[1])), rs=(rs[0], float(rs[1])), t1=(t1[0], float(t1[1])), t2=(t2[0], float(t2[1])), neck=float(d * necki),
                 hgt=float(abs((k1 + sl * (hd[0] - t1[0])) - ph_)), q=int(min(q, 100)))
        if best is None or g['q'] > best[1]['q']:
            best = ('hs_inverse' if d > 0 else 'hs_top', g)
    return best


# ── banderas y banderines ─────────────────────────────────────────────────
def flag(i: int, d: int, h, l, c, atr, v, form: bool = False) -> Optional[Tuple[str, Dict]]:
    """Bandera (canal paralelo) o banderín (convergente) tras un asta impulsiva; ruptura en la barra i en el sentido del asta (con form=True, i es la barra siguiente a la última y aún no ha roto)."""
    ll, hh, cl = _view(d, h, l, c)                                    # orientación alcista (mínimos, máximos, cierres)
    best = None
    for f in (4, 5, 6, 8, 10, 13, 16, 20):
        z0 = i - f
        if z0 < 25:
            continue
        fh, fl = hh[z0:i], ll[z0:i]
        top = float(fh.max())
        if form:
            if top - cl[i - 1] > NEAR * float(atr[i - 1]):
                continue
        elif not (cl[i] > top and cl[i - 1] <= top):
            continue
        for p in (4, 5, 6, 8, 10, 13, 15):
            pe, ps = z0 - 1, z0 - 1 - p
            if ps < 0:
                continue
            a = float(atr[pe])
            if not a > 0:
                continue
            pole = float(cl[pe] - cl[ps])
            if pole < 5.0 * a:
                continue
            path = float(np.abs(np.diff(cl[ps:pe + 1])).sum())
            if path <= 0 or pole / path < 0.45:
                continue
            ptop = float(hh[ps:z0 + 1].max())
            pbase = float(ll[ps:pe + 1].min())
            ph_ = ptop - pbase
            if ph_ <= 0:
                continue
            retr = (ptop - float(fl.min())) / ph_
            if retr > 0.5 or top > ptop + 0.5 * a or (top - float(fl.min())) > 0.6 * ph_:
                continue
            xs = np.arange(len(fh))
            if len(fh) >= 5:
                cu, cd = np.polyfit(xs, fh, 1), np.polyfit(xs, fl, 1)
                w_s, w_e = np.polyval(cu, 0) - np.polyval(cd, 0), np.polyval(cu, len(fh) - 1) - np.polyval(cd, len(fh) - 1)
                conv = bool(w_s > 0 and w_e <= 0.6 * w_s)
            else:
                conv = False
            slope = float(np.polyfit(xs, cl[z0:i], 1)[0])
            if not conv and slope > 0.1 * a:
                continue
            resid = float(np.std(cl[z0:i] - np.polyval(np.polyfit(xs, cl[z0:i], 1), xs))) / a
            brk, vr = (0.3, 1.2) if form else ((cl[i] - top) / a, _vol_ratio(v, i))
            q = 15 + (15 if pole >= 8 * a else 10) + 10.0 * min(pole / path, 1.0) + (15 if retr <= 0.3 else 8) + (10 if 5 <= f <= 15 else 5) + 10.0 * max(0.0, 1 - resid / 0.6) + _q_common(brk, vr)
            jb = ps + int(np.argmin(ll[ps:pe + 1]))
            jt = int(np.clip(z0 - 1 + int(np.argmax(hh[z0 - 1:z0 + 2])), ps, i - 1))
            g = dict(ps=(jb, float((l if d > 0 else h)[jb])), pe=(jt, float((h if d > 0 else l)[jt])), z0=int(z0), fh=float(h[z0:i].max()), fl=float(l[z0:i].min()),
                     pole=float(d * ph_), q=int(min(q, 100)), pen=bool(conv))
            name = ('bull_pennant' if conv else 'bull_flag') if d > 0 else ('bear_pennant' if conv else 'bear_flag')
            if best is None or g['q'] + 0.1 * f > best[1]['q'] + 0.1 * (i - best[1]['z0']):
                best = (name, g)
    return best


# ── taza con asa (con la curva pegada a los mínimos) ──────────────────────
def cup(S: List[list], i: int, h, l, c, atr, v, sma200, form: bool = False) -> Optional[Dict]:
    a = float(atr[i - 1])
    Hs = [s for s in S[-14:] if s[2] > 0]
    best = None
    for x in range(len(Hs) - 1):
        for y in range(x + 1, len(Hs)):
            ja, va = Hs[x][0], Hs[x][1]
            jb, vb = Hs[y][0], Hs[y][1]
            W = jb - ja
            if not (25 <= W <= 130) or i - jb > 35 or i - jb < 3:
                continue
            if not (0.92 * va <= vb <= 1.06 * va):
                continue
            seg = l[ja:jb + 1]
            bot = float(seg.min())
            bi = int(np.argmin(seg))
            depth = (min(va, vb) - bot) / min(va, vb)
            if not (0.12 <= depth <= 0.35) or (min(va, vb) - bot) < 3 * a or not (0.25 * W <= bi <= 0.75 * W):
                continue
            mid_level = bot + 0.5 * (min(va, vb) - bot)
            round_ = float(np.mean(c[ja:jb + 1] <= mid_level))
            if round_ < 0.55:                                          # una V pasa la mitad del tiempo bajo la mitad de la profundidad; una U, ~70 %
                continue
            xs = np.arange(W + 1)
            cf = np.polyfit(xs, seg, 2)
            if cf[0] <= 0:
                continue
            fit = np.polyval(cf, xs)
            r2 = 1.0 - ((seg - fit) ** 2).sum() / max(((seg - seg.mean()) ** 2).sum(), 1e-12)
            if r2 < 0.55:
                continue
            hj = jb + int(np.argmin(l[jb:i]))
            hlow = float(l[hj])
            hhigh = float(h[jb:i].max())
            pull = vb - hlow
            if pull > 0.4 * (min(va, vb) - bot) or pull < 0.3 * a:
                continue
            if form:
                if not (c[i - 1] <= hhigh and hhigh - c[i - 1] <= NEAR * a):
                    continue
            elif not (c[i] > hhigh and c[i - 1] <= hhigh):
                continue
            if c[jb + 1:i].max() > hhigh:                              # la ruptura del asa cuenta la primera vez
                continue
            ii = i - 1 if form else i
            if sma200[ii] == sma200[ii] and c[ii] < 0.9 * sma200[ii]:
                continue
            brk, vr = (0.3, 1.2) if form else ((c[i] - hhigh) / a, _vol_ratio(v, i))
            q = 15 + 15.0 * min(round_ / 0.7, 1.0) + 15.0 * max(0.0, (r2 - 0.55) / 0.4) + (15 if 0.15 <= depth <= 0.33 else 8) + (10 if 40 <= W <= 100 else 5) \
                + 10.0 * max(0.0, 1 - abs(vb / va - 1) / 0.06) + (10 if pull <= 0.25 * (min(va, vb) - bot) else 5) + _q_common(brk, vr)
            g = dict(left=(ja, float(va)), bottom=(ja + bi, bot), right=(jb, float(vb)), handle=(hj, hlow), fit=[float(x_) for x_ in cf], w0=int(ja), W=int(W + 1), hhigh=hhigh, q=int(min(q, 100)))
            if best is None or g['q'] > best['q']:
                best = g
    return best


# ── directrices con ≥ 3 toques ────────────────────────────────────────────
def trend_lines(S: List[list], i: int, h, l, c, atr) -> Tuple[Optional[Dict], Optional[Dict]]:
    """(resistencia descendente, soporte ascendente) con ≥ 3 toques hasta la barra i-1."""
    a = float(atr[i - 1])
    SW = [s for s in S[-14:] if i - s[0] <= MAXLEN]
    H = [(j, v) for j, v, t in SW if t > 0]
    Lw = [(j, v) for j, v, t in SW if t < 0]
    u = best_line(H, 1, h, l, c, a, i, min_touch=3, near=9.0) if len(H) >= 3 else None
    d = best_line(Lw, -1, h, l, c, a, i, min_touch=3, near=9.0) if len(Lw) >= 3 else None
    if u is not None and not (u['L'][2] * (u['tch'][-1][0] - u['tch'][0][0]) <= -1.0 * a and u['span'] >= 15):
        u = None
    if d is not None and not (d['L'][2] * (d['tch'][-1][0] - d['tch'][0][0]) >= 1.0 * a and d['span'] >= 15):
        d = None
    return u, d


# ── VCP, isla, diamante, soportes y resistencias de 3 toques ──────────────
def vcp(S: List[list], i: int, h, l, c, v, sma50, sma200, atr, form: bool = False) -> Optional[Dict]:
    """Tres contracciones cada vez menos profundas bajo máximos parecidos, con precio sobre sus medias; ruptura del último máximo."""
    H = [(s[0], s[1]) for s in S if s[2] > 0][-3:]
    ii = i - 1 if form else i
    if len(H) < 3 or H[-1][0] - H[0][0] > 120 or not (c[ii] > sma50[ii] > sma200[ii]):
        return None
    if form:
        if not (c[i - 1] <= H[-1][1] and H[-1][1] - c[i - 1] <= NEAR * float(atr[i - 1])):
            return None
    elif not (c[i] > H[-1][1] >= c[i - 1]):
        return None
    depths, lows = [], []
    for k in range(3):
        jh = H[k][0]
        jn = H[k + 1][0] if k < 2 else i
        kk = jh + int(np.argmin(l[jh:jn + 1]))
        depths.append((H[k][1] - float(l[kk])) / H[k][1])
        lows.append((kk, float(l[kk])))
    if not (depths[0] > depths[1] > depths[2] and depths[2] <= 0.08 and depths[0] <= 0.40):
        return None
    brk, vr = (0.3, 1.2) if form else ((c[i] - H[-1][1]) / float(atr[i - 1]), _vol_ratio(v, i))
    q = 30 + (15 if depths[1] <= 0.75 * depths[0] else 8) + (15 if depths[2] <= 0.75 * depths[1] else 8) + (15 if depths[2] <= 0.05 else 8) + 10 + _q_common(brk, vr)
    return dict(highs=H, lows=lows, depths=[float(d) for d in depths], q=int(min(q, 100)), key=(H[0][0], H[-1][0]))


def island(i: int, h, l, dn_state, up_state) -> Optional[Tuple[str, Dict]]:
    for aidx in range(i - 5, i):
        if aidx < 2:
            continue
        if h[aidx] < l[aidx - 1] and dn_state[aidx - 1] and l[i] > h[aidx:i].max():
            return 'island_bull', dict(a=aidx, gap_from=float(l[aidx - 1]), lo=float(l[aidx:i + 1].min()), hi=float(h[aidx:i].max()), key=(aidx,))
        if l[aidx] > h[aidx - 1] and up_state[aidx - 1] and h[i] < l[aidx:i].min():
            return 'island_bear', dict(a=aidx, gap_from=float(h[aidx - 1]), lo=float(l[aidx:i].min()), hi=float(h[aidx:i + 1].max()), key=(aidx,))
    return None


def diamond(i: int, h, l, c, dn_state, up_state) -> Optional[Tuple[str, Dict]]:
    if i < 60:
        return None
    W = 40
    hs_, ls_ = h[i - W:i], l[i - W:i]
    w = np.array([hs_[b * 5:(b + 1) * 5].max() - ls_[b * 5:(b + 1) * 5].min() for b in range(8)])
    pk = int(np.argmax(w))
    if not (2 <= pk <= 5 and w[0] <= 0.6 * w[pk] and w[-1] <= 0.55 * w[pk] and (np.diff(w[:pk + 1]) > -0.15 * w[pk]).all() and (np.diff(w[pk:]) < 0.15 * w[pk]).all()):
        return None
    top10, bot10 = float(h[i - 10:i].max()), float(l[i - 10:i].min())
    env_hi = [(i - W + b * 5 + 2, float(hs_[b * 5:(b + 1) * 5].max())) for b in range(8)]
    env_lo = [(i - W + b * 5 + 2, float(ls_[b * 5:(b + 1) * 5].min())) for b in range(8)]
    g = dict(hi=env_hi, lo=env_lo, top=top10, bot=bot10, width=float(w.max()), key=(i // 10,))
    if c[i] > top10 and dn_state[i - W - 1]:
        return 'diamond_bull', g
    if c[i] < bot10 and up_state[i - W - 1]:
        return 'diamond_bear', g
    return None


def levels3(S: List[list], i: int, o, h, l, c, atr) -> Optional[Tuple[str, Dict]]:
    """Rebote en un soporte (o rechazo en una resistencia) de 3 toques: tres pivotes a ≤ 0,8 ATR entre sí y una vela que lo prueba y se gira."""
    a = float(atr[i])
    for t, nm in ((-1, 'support_bounce'), (1, 'resistance_reject')):
        P = [(s[0], s[1]) for s in S if s[2] == t][-3:]
        if len(P) < 3:
            continue
        vals = [v for _, v in P]
        lvl = float(np.mean(vals))
        if max(vals) - min(vals) > 0.8 * a or P[-1][0] - P[0][0] < 15 or not (3 <= i - P[-1][0] <= 60):
            continue
        if t < 0 and lvl - 0.8 * a <= l[i] <= lvl + 0.3 * a and c[i] > lvl and c[i] > o[i]:
            return nm, dict(level=lvl, touches=P, key=(P[-1][0], i // 3))
        if t > 0 and lvl - 0.3 * a <= h[i] <= lvl + 0.8 * a and c[i] < lvl and c[i] < o[i]:
            return nm, dict(level=lvl, touches=P, key=(P[-1][0], i // 3))
    return None


# ── motor principal ───────────────────────────────────────────────────────
def _route(kind: str, d: int) -> Optional[str]:
    return {('asc_triangle', 1): 'asc_triangle', ('desc_triangle', -1): 'desc_triangle', ('sym_tri', 1): 'sym_tri_up', ('sym_tri', -1): 'sym_tri_dn',
            ('rising_wedge', -1): 'rising_wedge', ('falling_wedge', 1): 'falling_wedge', ('rect', 1): 'rect_up', ('rect', -1): 'rect_dn',
            ('channel_up', 1): 'channel_up_up', ('channel_up', -1): 'channel_up_dn', ('channel_dn', -1): 'channel_dn_dn', ('channel_dn', 1): 'channel_dn_up'}.get((kind, d))


def detect(F: Dict[str, np.ndarray]) -> Tuple[List[Dict], List[Dict]]:
    """Eventos de las figuras geométricas [{name, i, q, g}] y las figuras aún sin romper en la última barra (compresiones y demás, ver forming_figs)."""
    o, h, l, c, v, atr = F['o'], F['h'], F['l'], F['c'], F['v'], F['atr']
    sma50, sma200, sma20, atrp = F['sma50'], F['sma200'], F['sma20'], F['atrp']
    n = len(c)
    events: List[Dict] = []
    ph, pl = pivot_flags(h, l)
    S: List[list] = []
    ver, cver = 0, -1
    cmp_: Optional[Dict] = None
    tu = td = None
    dead: set = set()
    used: Dict[str, tuple] = {}
    with np.errstate(invalid='ignore'):
        dn_state = (c / shift(c, 5) - 1.0 < -1.5 * atrp / 100.0) & (c < sma20)
        up_state = (c / shift(c, 5) - 1.0 > 1.5 * atrp / 100.0) & (c > sma20)

    def fire(name: str, key, i: int, q: Optional[int], g: Dict) -> None:
        if (q is not None and q < QMIN) or used.get(name) == key:
            return
        used[name] = key
        events.append({'name': name, 'i': i, 'q': None if q is None else int(q), 'g': g})

    for i in range(max(30, K + 2), n):
        j = i - K
        a = float(atr[i])
        if not a > 0:
            continue
        ch = False
        if ph[j]:
            ch |= _add(S, j, float(h[j]), 1, AMP * a)
        if pl[j]:
            ch |= _add(S, j, float(l[j]), -1, AMP * a)
        if ch:
            ver += 1
        if ver != cver:
            cver = ver
            cmp_ = compression(S, i, h, l, c, atr)
            tu, td = trend_lines(S, i, h, l, c, atr)
        ci, cp = float(c[i]), float(c[i - 1])
        # compresiones (triángulos, cuñas, rectángulo)
        if cmp_ is not None and cmp_['key'] not in dead and i - cmp_['xl'] <= 45:
            st = cmp_
            ui, di, up_, dp = at(st['U'], i), at(st['D'], i), at(st['U'], i - 1), at(st['D'], i - 1)
            brk_up = ci > ui + CLOSE * a and cp <= up_ + CLOSE * a
            brk_dn = ci < di - CLOSE * a and cp >= dp - CLOSE * a
            expired = st['apex'] is not None and (i - st['x0']) > 1.15 * (st['apex'] - st['x0'])
            if (brk_up or brk_dn) and not expired:
                d = 1 if brk_up else -1
                nm = _route(st['kind'], d)
                if nm:
                    margin = (ci - ui) / a if d > 0 else (di - ci) / a
                    fire(nm, st['key'], i, compression_quality(st, margin, _vol_ratio(v, i)),
                         dict(U=st['U'], D=st['D'], tu=st['tu'], tl=st['tl'], x0=st['x0'], apex=st['apex'], kind=st['kind'], conv=st['conv']))
                dead.add(st['key'])
        # directrices
        for nm, tl_, sd in (('trendline_up', tu, 1), ('trendline_dn', td, -1)):
            if tl_ is None or i - tl_['tch'][-1][0] > 45:
                continue
            Lv, Lp = at(tl_['L'], i), at(tl_['L'], i - 1)
            if (sd > 0 and ci > Lv + CLOSE * a and cp <= Lp + CLOSE * a) or (sd < 0 and ci < Lv - CLOSE * a and cp >= Lp - CLOSE * a):
                nt = len(tl_['tch'])
                margin = abs(ci - Lv) / a
                q = {3: 38, 4: 48}.get(nt, 56) + 15.0 * max(0.0, 1 - tl_['dev'] / TOL) + min(tl_['span'], 60) / 6.0 + _q_common(margin, _vol_ratio(v, i)) - 8
                fire(nm, (tl_['tch'][0][0], tl_['tch'][-1][0]), i, int(round(min(q, 100))), dict(L=tl_['L'], tch=tl_['tch']))
        # dobles, triples, hombro-cabeza-hombro, banderas
        for d_ in (1, -1):
            r = multi_bottom(S, i, d_, h, l, c, atr, v)
            if r:
                fire(r[0], (r[2],), i, r[1]['q'], {k: x for k, x in r[1].items() if k != 'q'})
            r = head_shoulders(S, i, d_, h, l, c, atr, v)
            if r:
                fire(r[0], (r[1]['ls'][0], r[1]['hd'][0], r[1]['rs'][0]), i, r[1]['q'], {k: x for k, x in r[1].items() if k != 'q'})
            r = flag(i, d_, h, l, c, atr, v)
            if r:
                fire(r[0], (r[1]['z0'], r[1]['ps'][0]), i, r[1]['q'], {k: x for k, x in r[1].items() if k != 'q'})
        g = cup(S, i, h, l, c, atr, v, sma200)
        if g:
            fire('cup_handle', (g['left'][0], g['right'][0]), i, g['q'], {k: x for k, x in g.items() if k != 'q'})
        g = vcp(S, i, h, l, c, v, sma50, sma200, atr)
        if g:
            fire('vcp', g['key'], i, g['q'], {k: x for k, x in g.items() if k not in ('q', 'key')})
        r = island(i, h, l, dn_state, up_state)
        if r:
            fire(r[0], r[1]['key'], i, None, {k: x for k, x in r[1].items() if k != 'key'})
        r = diamond(i, h, l, c, dn_state, up_state)
        if r:
            fire(r[0], r[1]['key'], i, None, {k: x for k, x in r[1].items() if k != 'key'})
        r = levels3(S, i, o, h, l, c, atr)
        if r:
            fire(r[0], r[1]['key'], i, None, {k: x for k, x in r[1].items() if k != 'key'})
    forming: List[Dict] = []
    if n > 40:
        st = compression(S, n, h, l, c, atr)
        if st is not None:
            a = float(atr[n - 1])
            ui, di = at(st['U'], n - 1), at(st['D'], n - 1)
            expired = st['apex'] is not None and (n - 1 - st['x0']) > 1.15 * (st['apex'] - st['x0'])
            if di - CLOSE * a <= float(c[n - 1]) <= ui + CLOSE * a and not expired and n - 1 - st['xl'] <= 45:
                forming.append(dict(st, type='cmp', i=n - 1))
        forming += forming_figs(S, n, h, l, c, v, atr, sma50, sma200)
    return events, forming


def forming_figs(S: List[list], n: int, h, l, c, v, atr, sma50, sma200) -> List[Dict]:
    """Figuras que en el último cierre ya están dibujadas pero aún no han roto: [{type:'fig', name, d, g, q, trig, dist}] (trig = nivel que hay que cerrar más allá; dist en ATR).
    Se evalúan como si la barra n (la siguiente) fuese la de ruptura, con los mismos criterios que las rupturas, y sin usar nada posterior al último cierre."""
    a = float(atr[n - 1])
    if not a > 0:
        return []
    last = float(c[n - 1])
    found: Dict[str, Dict] = {}

    def keep(name: str, d: int, g: Dict, q: int, trig: float) -> None:
        if q < QFORM or (d > 0 and trig < last) or (d < 0 and trig > last):
            return
        cur = found.get(name)
        if cur is None or q > cur['q']:
            found[name] = dict(type='fig', name=name, d=d, g={k: x for k, x in g.items() if k != 'q'}, q=int(q), trig=float(trig), dist=float(abs(trig - last) / a))

    for d_ in (1, -1):
        r = multi_bottom(S, n, d_, h, l, c, atr, v, form=True)
        if r:
            keep(r[0], d_, r[1], r[1]['q'], r[1]['neck'])
        r = head_shoulders(S, n, d_, h, l, c, atr, v, form=True)
        if r:
            keep(r[0], d_, r[1], r[1]['q'], r[1]['neck'])
        r = flag(n, d_, h, l, c, atr, v, form=True)
        if r:
            keep(r[0], d_, r[1], r[1]['q'], r[1]['fh'] if d_ > 0 else r[1]['fl'])
    g = cup(S, n, h, l, c, atr, v, sma200, form=True)
    if g:
        keep('cup_handle', 1, g, g['q'], g['hhigh'])
    g = vcp(S, n, h, l, c, v, sma50, sma200, atr, form=True)
    if g:
        keep('vcp', 1, g, g['q'], g['highs'][-1][1])
    return sorted(found.values(), key=lambda f: (f['dist'], -f['q']))
