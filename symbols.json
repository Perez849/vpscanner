"""
jsmath.py — Utilidades para replicar EXACTAMENTE la aritmética de JavaScript.

El motivo: Number.prototype.toFixed() de JS y round() de Python difieren en el
redondeo del caso .5 y en cómo arrastran el epsilon de coma flotante. Para que
el scanner Python produzca cifras idénticas al HTML hasta el último decimal,
usamos to_fixed() en lugar de round() allí donde el HTML usa .toFixed().
"""

from __future__ import annotations


from decimal import Decimal, ROUND_HALF_UP


def to_fixed(x, digits: int):
    """
    Replica Number(x).toFixed(digits) de JavaScript y devuelve un float
    (equivalente a +x.toFixed(d) del HTML).

    Clave de fidelidad: JS opera sobre el DOUBLE real, no sobre el literal
    decimal. Ej: (21.205).toFixed(2) === "21.20" porque 21.205 almacenado como
    double es 21.20499999999999829...  Para reproducirlo, convertimos el float a
    Decimal EXACTO (Decimal(float) toma el valor binario real, no el literal) y
    redondeamos half-up sobre ese valor exacto — que es justo lo que ve V8.
    """
    if x is None:
        return None
    d = Decimal(x)  # valor binario EXACTO del double (como lo ve JS)
    q = Decimal(1).scaleb(-digits)  # 10^-digits
    rounded = d.quantize(q, rounding=ROUND_HALF_UP)
    return float(rounded)
