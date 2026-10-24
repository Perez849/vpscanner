"""
Catálogo único de patrones: nombre en español, sentido, familia, qué es y cómo se calcula su plan.
Lo leen el motor, el laboratorio, el estudio y la web (se exporta a data/catalog.json), así que un patrón se define en un solo sitio.
"""
from __future__ import annotations
from typing import Dict, NamedTuple


class Pat(NamedTuple):
    key: str
    label: str
    dir: int           # +1 alcista, -1 bajista, 0 neutral
    fam: str           # velas | rupturas | figuras | directrices | rebotes
    desc: str          # qué es
    rule: str          # cómo se calcula el plan (entrada, stop y objetivo)
    bars: int = 0      # velas que forman el patrón (solo familia velas)


CATALOG: Dict[str, Pat] = {}


def _add(key, label, d, fam, desc, rule, bars=0):
    CATALOG[key] = Pat(key, label, d, fam, desc, rule, bars)


GEN2R = 'Objetivo genérico de 2R. Stop más allá del extremo del patrón de velas.'
for key, label, d, bars, desc in (
    ('hammer', 'Martillo', 1, 1, 'Vela de cuerpo pequeño y mecha inferior larga tras una caída: los vendedores empujaron y los compradores recuperaron.'),
    ('inv_hammer', 'Martillo invertido', 1, 1, 'Cuerpo pequeño y mecha superior larga tras una caída: primer intento alcista.'),
    ('shooting_star', 'Estrella fugaz', -1, 1, 'Cuerpo pequeño y mecha superior larga tras una subida: los compradores no se sostuvieron.'),
    ('hanging_man', 'Hombre colgado', -1, 1, 'Martillo tras una subida: advierte de agotamiento.'),
    ('bull_engulf', 'Envolvente alcista', 1, 2, 'Vela verde que engulle el cuerpo de la roja anterior tras una caída.'),
    ('bear_engulf', 'Envolvente bajista', -1, 2, 'Vela roja que engulle el cuerpo de la verde anterior tras una subida.'),
    ('piercing', 'Línea penetrante', 1, 2, 'Vela verde que abre bajo el mínimo anterior y cierra por encima de la mitad de la roja.'),
    ('dark_cloud', 'Nube oscura', -1, 2, 'Vela roja que abre sobre el máximo anterior y cierra bajo la mitad de la verde.'),
    ('morning_star', 'Estrella de la mañana', 1, 3, 'Vela roja larga, vela pequeña con hueco y vela verde que recupera más de la mitad: suelo en tres velas.'),
    ('evening_star', 'Estrella de la tarde', -1, 3, 'Vela verde larga, vela pequeña con hueco y vela roja que devuelve más de la mitad: techo en tres velas.'),
    ('soldiers', 'Tres soldados', 1, 3, 'Tres velas verdes consecutivas, cada una abre dentro de la anterior y cierra más alto, tras una caída.'),
    ('crows', 'Tres cuervos', -1, 3, 'Tres velas rojas consecutivas, cada una abre dentro de la anterior y cierra más bajo, tras una subida.'),
    ('harami_bull', 'Harami alcista', 1, 2, 'Vela verde pequeña contenida en el cuerpo de una roja grande: la caída se frena.'),
    ('harami_bear', 'Harami bajista', -1, 2, 'Vela roja pequeña contenida en el cuerpo de una verde grande: la subida se frena.'),
    ('tweezer_bottom', 'Pinzas de suelo', 1, 2, 'Dos velas con el mismo mínimo, la segunda verde, tras una caída.'),
    ('tweezer_top', 'Pinzas de techo', -1, 2, 'Dos velas con el mismo máximo, la segunda roja, tras una subida.'),
    ('dragonfly', 'Doji libélula', 1, 1, 'Doji con una mecha inferior larga y sin mecha superior tras una caída.'),
    ('gravestone', 'Doji lápida', -1, 1, 'Doji con una mecha superior larga y sin mecha inferior tras una subida.'),
    ('marubozu_bull', 'Marubozu alcista', 1, 1, 'Vela verde larga sin mechas: dominio comprador durante toda la sesión.'),
    ('marubozu_bear', 'Marubozu bajista', -1, 1, 'Vela roja larga sin mechas: dominio vendedor durante toda la sesión.'),
    ('keyrev_bull', 'Reversión clave ↑', 1, 2, 'Marca un mínimo nuevo y cierra por encima del máximo de la vela anterior.'),
    ('keyrev_bear', 'Reversión clave ↓', -1, 2, 'Marca un máximo nuevo y cierra por debajo del mínimo de la vela anterior.'),
    ('doji_bull', 'Doji ↑ (tras caída)', 1, 1, 'Doji con mechas a ambos lados tras una caída: indecisión en el suelo.'),
    ('doji_bear', 'Doji ↓ (tras subida)', -1, 1, 'Doji con mechas a ambos lados tras una subida: indecisión en el techo.'),
    ('spin_bull', 'Peonza ↑', 1, 1, 'Cuerpo pequeño con mechas largas tras una caída.'),
    ('spin_bear', 'Peonza ↓', -1, 1, 'Cuerpo pequeño con mechas largas tras una subida.'),
    ('doji_star_bull', 'Estrella doji ↑', 1, 3, 'Estrella de la mañana cuya vela central es un doji.'),
    ('doji_star_bear', 'Estrella doji ↓', -1, 3, 'Estrella de la tarde cuya vela central es un doji.'),
    ('three_inside_up', 'Tres interiores ↑', 1, 3, 'Harami alcista confirmado por un cierre sobre la apertura de la primera vela.'),
    ('three_inside_dn', 'Tres interiores ↓', -1, 3, 'Harami bajista confirmado por un cierre bajo la apertura de la primera vela.'),
):
    _add(key, label, d, 'velas', desc, GEN2R, bars)

for p, nm in ((55, '55 sesiones'), (100, '100 sesiones'), (252, 'un año')):
    _add(f'donch{p}_up', f'Ruptura máx. {nm}' if p != 252 else 'Ruptura máx. anual', 1, 'rupturas', f'El cierre supera el máximo de {nm}.', 'Objetivo genérico de 2R. Stop 1,5 ATR más allá del nivel roto.')
    _add(f'donch{p}_dn', f'Ruptura mín. {nm}' if p != 252 else 'Ruptura mín. anual', -1, 'rupturas', f'El cierre pierde el mínimo de {nm}.', 'Objetivo genérico de 2R. Stop 1,5 ATR más allá del nivel roto.')
_add('bb_squeeze_up', 'Squeeze Bollinger ↑', 1, 'rupturas', 'Bandas de Bollinger en su anchura mínima de seis meses y cierre sobre la banda superior.', 'Objetivo genérico de 2R. Stop en la media de 20 sesiones.')
_add('bb_squeeze_dn', 'Squeeze Bollinger ↓', -1, 'rupturas', 'Bandas de Bollinger en su anchura mínima de seis meses y cierre bajo la banda inferior.', 'Objetivo genérico de 2R. Stop en la media de 20 sesiones.')
_add('atr_squeeze_up', 'Compresión ATR ↑', 1, 'rupturas', 'La volatilidad de 5 sesiones cae por debajo del 60 % de la de 50 y el precio rompe el máximo de 20.', 'Objetivo genérico de 2R. Stop en la media de 20 sesiones.')
_add('atr_squeeze_dn', 'Compresión ATR ↓', -1, 'rupturas', 'La volatilidad de 5 sesiones cae por debajo del 60 % de la de 50 y el precio rompe el mínimo de 20.', 'Objetivo genérico de 2R. Stop en la media de 20 sesiones.')
_add('box_up', 'Caja (Darvas) ↑', 1, 'rupturas', 'Caja de 15 sesiones de ≤ 3 ATR de alto y cierre por encima.', 'Objetivo = altura de la caja proyectada desde la ruptura. Stop al otro lado de la caja.')
_add('box_dn', 'Caja (Darvas) ↓', -1, 'rupturas', 'Caja de 15 sesiones de ≤ 3 ATR de alto y cierre por debajo.', 'Objetivo = altura de la caja proyectada desde la ruptura. Stop al otro lado de la caja.')
_add('base_near_high', 'Base junto a máximos', 1, 'rupturas', 'Base apretada (≤ 8 % de rango en 20 sesiones) a menos de un 5 % del máximo anual y ruptura de su techo.', 'Objetivo = altura de la base proyectada desde la ruptura. Stop bajo la base.')

_add('double_bottom', 'Doble suelo', 1, 'figuras', 'Dos mínimos iguales tras una caída y ruptura del máximo intermedio (el cuello).', 'Objetivo = altura de la figura (cuello − mínimos) proyectada sobre el cuello. Stop 0,25 ATR bajo el mínimo más bajo.')
_add('double_top', 'Doble techo', -1, 'figuras', 'Dos máximos iguales tras una subida y pérdida del mínimo intermedio (el cuello).', 'Objetivo = altura de la figura (máximos − cuello) proyectada bajo el cuello. Stop 0,25 ATR sobre el máximo más alto.')
_add('triple_bottom', 'Triple suelo', 1, 'figuras', 'Tres mínimos iguales y ruptura del máximo intermedio más alto.', 'Objetivo = altura de la figura (cuello − mínimos) proyectada sobre el cuello. Stop 0,25 ATR bajo el mínimo más bajo.')
_add('triple_top', 'Triple techo', -1, 'figuras', 'Tres máximos iguales y pérdida del mínimo intermedio más bajo.', 'Objetivo = altura de la figura (máximos − cuello) proyectada bajo el cuello. Stop 0,25 ATR sobre el máximo más alto.')
_add('hs_top', 'Hombro-cabeza-hombro', -1, 'figuras', 'Cabeza prominente entre dos hombros simétricos tras una subida; se activa al perder la línea de cuello.', 'Objetivo = distancia cabeza–línea de cuello proyectada bajo la ruptura. Stop sobre el hombro derecho.')
_add('hs_inverse', 'H-C-H invertido', 1, 'figuras', 'Cabeza prominente bajo dos hombros simétricos tras una caída; se activa al romper la línea de cuello.', 'Objetivo = distancia cabeza–línea de cuello proyectada sobre la ruptura. Stop bajo el hombro derecho.')
_add('asc_triangle', 'Triángulo ascendente', 1, 'figuras', 'Resistencia plana y soporte ascendente que se estrechan; ruptura de la resistencia.', 'Objetivo = altura de la base (en el inicio) proyectada sobre la resistencia. Stop bajo el último mínimo / la base.')
_add('desc_triangle', 'Triángulo descendente', -1, 'figuras', 'Soporte plano y resistencia descendente que se estrechan; pérdida del soporte.', 'Objetivo = altura de la base (en el inicio) proyectada bajo el soporte. Stop sobre el último máximo / el techo.')
_add('sym_tri_up', 'Triángulo simétrico ↑', 1, 'figuras', 'Máximos descendentes y mínimos ascendentes que convergen; ruptura al alza.', 'Objetivo = anchura del triángulo en su inicio proyectada desde la ruptura. Stop bajo el último mínimo / la base.')
_add('sym_tri_dn', 'Triángulo simétrico ↓', -1, 'figuras', 'Máximos descendentes y mínimos ascendentes que convergen; ruptura a la baja.', 'Objetivo = anchura del triángulo en su inicio proyectada desde la ruptura. Stop sobre el último máximo / el techo.')
_add('rising_wedge', 'Cuña ascendente', -1, 'figuras', 'Dos rectas ascendentes que convergen (la inferior sube más rápido); suele resolverse a la baja.', 'Objetivo = inicio de la cuña (la recta inferior en su arranque). Stop sobre el último máximo de la cuña.')
_add('falling_wedge', 'Cuña descendente', 1, 'figuras', 'Dos rectas descendentes que convergen (la superior baja más rápido); suele resolverse al alza.', 'Objetivo = inicio de la cuña (la recta superior en su arranque). Stop bajo el último mínimo de la cuña.')
_add('rect_up', 'Rectángulo ↑', 1, 'figuras', 'Rango lateral entre dos rectas casi horizontales con ≥ 5 toques; ruptura al alza.', 'Objetivo = altura del rectángulo proyectada desde la ruptura. Stop bajo el último mínimo / la base.')
_add('rect_dn', 'Rectángulo ↓', -1, 'figuras', 'Rango lateral entre dos rectas casi horizontales con ≥ 5 toques; ruptura a la baja.', 'Objetivo = altura del rectángulo proyectada desde la ruptura. Stop sobre el último máximo / el techo.')
_add('channel_up_up', 'Canal ascendente ↑', 1, 'figuras', 'Dos rectas paralelas ascendentes con ≥ 6 toques; ruptura de la superior: la tendencia se acelera.', 'Objetivo = anchura del canal proyectada desde la ruptura. Stop bajo la recta inferior (estructural, suele quedar lejos).')
_add('channel_up_dn', 'Canal ascendente ↓', -1, 'figuras', 'Dos rectas paralelas ascendentes con ≥ 6 toques; pérdida de la inferior: la tendencia se agota.', 'Objetivo = anchura del canal proyectada desde la ruptura. Stop sobre la recta superior (estructural, suele quedar lejos).')
_add('channel_dn_dn', 'Canal descendente ↓', -1, 'figuras', 'Dos rectas paralelas descendentes con ≥ 6 toques; pérdida de la inferior: la caída se acelera.', 'Objetivo = anchura del canal proyectada desde la ruptura. Stop sobre la recta superior (estructural, suele quedar lejos).')
_add('channel_dn_up', 'Canal descendente ↑', 1, 'figuras', 'Dos rectas paralelas descendentes con ≥ 6 toques; ruptura de la superior: la caída se agota.', 'Objetivo = anchura del canal proyectada desde la ruptura. Stop bajo la recta inferior (estructural, suele quedar lejos).')
_add('bull_flag', 'Bandera alcista', 1, 'figuras', 'Asta impulsiva al alza y consolidación corta en canal descendente o plano; ruptura del canal.', 'Objetivo = altura del asta proyectada desde la ruptura de la bandera. Stop bajo el mínimo de la bandera.')
_add('bear_flag', 'Bandera bajista', -1, 'figuras', 'Asta impulsiva a la baja y consolidación corta en canal ascendente o plano; ruptura del canal.', 'Objetivo = altura del asta proyectada desde la ruptura de la bandera. Stop sobre el máximo de la bandera.')
_add('bull_pennant', 'Banderín alcista', 1, 'figuras', 'Asta alcista seguida de una consolidación triangular pequeña; ruptura al alza.', 'Objetivo = altura del asta proyectada desde la ruptura de la bandera. Stop bajo el mínimo de la bandera.')
_add('bear_pennant', 'Banderín bajista', -1, 'figuras', 'Asta bajista seguida de una consolidación triangular pequeña; ruptura a la baja.', 'Objetivo = altura del asta proyectada desde la ruptura de la bandera. Stop sobre el máximo de la bandera.')
_add('cup_handle', 'Taza con asa', 1, 'figuras', 'Fondo redondeado (no en V) con bordes a la misma altura y un pequeño retroceso final (el asa); ruptura del asa.', 'Objetivo = profundidad de la taza proyectada sobre el borde. Stop bajo el mínimo del asa.')
_add('vcp', 'VCP (contracción)', 1, 'figuras', 'Tres retrocesos cada vez menos profundos bajo máximos parecidos, con el precio sobre sus medias de 50 y 200; ruptura del último máximo.', 'Objetivo genérico de 2R. Stop bajo el mínimo de la última contracción.')
_add('diamond_bull', 'Diamante de suelo', 1, 'figuras', 'Rango que se ensancha y luego se estrecha tras una caída; ruptura al alza.', 'Objetivo = anchura máxima del diamante proyectada desde la ruptura. Stop bajo el rango final.')
_add('diamond_bear', 'Diamante de techo', -1, 'figuras', 'Rango que se ensancha y luego se estrecha tras una subida; ruptura a la baja.', 'Objetivo = anchura máxima del diamante proyectada desde la ruptura. Stop sobre el rango final.')
_add('island_bull', 'Isla de suelo', 1, 'figuras', 'Hueco a la baja seguido, pocas sesiones después, de un hueco al alza que deja una isla aislada.', 'Objetivo = cierre del hueco (nivel previo a la isla). Stop bajo la isla.')
_add('island_bear', 'Isla de techo', -1, 'figuras', 'Hueco al alza seguido de un hueco a la baja que deja una isla aislada.', 'Objetivo = cierre del hueco (nivel previo a la isla). Stop sobre la isla.')
_add('trendline_up', 'Ruptura de directriz bajista', 1, 'directrices', 'El cierre supera una recta descendente con ≥ 3 toques.', 'Objetivo genérico de 2R con stop de 1,5 ATR (la ruptura de una directriz no tiene medida clásica).')
_add('trendline_dn', 'Ruptura de directriz alcista', -1, 'directrices', 'El cierre pierde una recta ascendente con ≥ 3 toques.', 'Objetivo genérico de 2R con stop de 1,5 ATR (la ruptura de una directriz no tiene medida clásica).')
_add('support_bounce', 'Rebote en soporte', 1, 'rebotes', 'Vela que toca un soporte de 3 toques y cierra por encima.', 'Objetivo genérico de 2R. Stop bajo el soporte y la mecha de rebote.')
_add('resistance_reject', 'Rechazo en resistencia', -1, 'rebotes', 'Vela que toca una resistencia de 3 toques y cierra por debajo.', 'Objetivo genérico de 2R. Stop sobre la resistencia y la mecha de rechazo.')

FAMILIES = {'figuras': 'Figuras', 'rupturas': 'Rupturas', 'velas': 'Velas', 'directrices': 'Directrices', 'rebotes': 'Rebotes en S/R'}
HQ = 75            # una figura es de «calidad alta» a partir de esta puntuación (0–100)
