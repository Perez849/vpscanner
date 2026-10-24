# VP Scanner · laboratorio de chartismo

Detecta **figuras, velas y rupturas** en acciones grandes de EE.UU., las **dibuja** sobre el gráfico con su plan (entrada, stop, objetivo) y **mide si funcionan**.

- **Radar**: qué figuras se están formando o se han completado, por calidad.
- **Gráfico**: velas con las figuras dibujadas (rectas con sus toques, ápice, medida), plan por patrón, resultado simulado y evidencia del estudio.
- **Evidencia**: ¿baten los patrones a entrar al azar? Estudio con corrección por probar muchos patrones.
- **Guía**: qué es cada patrón y cómo se calcula su plan.

> No es una recomendación de inversión. Dibujar bien una figura no demuestra que funcione: el estudio de la pestaña Evidencia dice qué ha aportado de verdad cada patrón.

## Cómo se reconocen las figuras (`vp/shapes.py`, causal)

Una figura no es «dos pivotes unidos por una recta». El motor:

- **Oscilaciones por zigzag de ATR**: un pivote solo cuenta si la oscilación mide ≥ 1 ATR, alternando máximo/mínimo; se confirma 3 barras después.
- **Rectas por toques**: entre todos los pares de pivotes se elige la recta con más **toques** (pivotes a ≤ 0,35 ATR) que el precio respeta (los cierres no la cruzan, las mechas no la superan en más de 0,6 ATR), que pasa por uno de los dos últimos pivotes y no queda lejos del precio actual.
- **Clasificación por pendientes y convergencia**: triángulos (simétrico, ascendente, descendente), cuñas (≥ 5 toques) y rectángulo; duración ≤ 80 barras, base ≤ 16 % del precio y ≤ 9 ATR, anchura al romper ≤ 70 % de la inicial, con ápice.
- **H-C-H** con cabeza ≥ 1,2 ATR más alta que los hombros, simetría en altura y tiempo y tendencia previa; **dobles y triples** techos/suelos; **banderas y banderines** con asta impulsiva (≥ 5 ATR) y retroceso ≤ 50 %; **taza con asa** con la parábola ajustada a los mínimos y prueba de redondez; **VCP**, **diamante**, **isla**, **directrices** con ≥ 3 toques y **rebotes en soportes y resistencias** de 3 toques.
- **Calidad 0–100** por figura (toques, ajuste, simetría, duración, forma, ruptura, volumen); por debajo de 35 no es figura.
- **Figuras en formación**: una compresión que aún no ha roto se dibuja con sus dos rectas y los dos escenarios (rompe arriba / rompe abajo).

Además, 30 patrones de velas y 13 rupturas (máximos y mínimos de 55/100/252 sesiones, Bollinger y ATR comprimidos, caja de Darvas, base junto al máximo anual). El catálogo completo (`vp/catalog.py`) alimenta el motor, el estudio y la Guía.

## El plan de cada patrón (`vp/plan.py`)

- **Objetivo** por la medida clásica de la figura (doble suelo: cuello + altura; triángulos: anchura de la base; banderas: el asta; taza: su profundidad…) y **T1** al 50 % del camino.
- **Stop** estructural más allá de la figura (0,25 ATR de margen), **nunca a menos de 1 ATR** de la entrada y, si queda a más de 3 ATR, un stop táctico de 1,5 ATR.
- **Obstáculos**: soportes y resistencias fuertes entre la entrada y el objetivo.
- **Resultado simulado**: entrada en la apertura siguiente, el stop gana si se tocan ambos el mismo día, cierre a las 40 sesiones.

## Estudio de evidencia (`vp/study.py`)

- Universo: miembros actuales del S&P 500 y el Nasdaq-100; se usan las **50 y las 300 más negociadas en cada fecha** (dinero medio de 60 sesiones).
- Cada señal se opera en la apertura siguiente y se cierra a 5, **10** (principal) y 20 sesiones, con un coste del 0,15 % de ida y vuelta; los patrones bajistas se miden en corto.
- «Exceso» = rendimiento − media de entrar al azar en el mismo universo el mismo año; **t con errores estándar agrupados por día**.
- **Umbral del mejor de N**: se desplazan al azar las señales de cada valor y se anota el mayor |t| entre todos los patrones; solo «supera el azar» quien pasa el percentil 95 de ese máximo. Se estudian también las señales de **calidad ≥ 75** de cada figura.
- **Limitación**: los miembros son los actuales (sesgo de supervivencia que favorece a los largos).

RESULTADOS_PENDIENTES

## Estructura

```
index.html            la web (lee data/*.json)
vp/
  catalog.py          catálogo de patrones: nombre, descripción, regla del plan
  ta.py               indicadores (ATR, medias, RSI…) con NumPy
  candles.py          30 patrones de velas
  breakouts.py        13 rupturas y compresiones
  shapes.py           motor geométrico de figuras (swings, rectas por toques, calidad, formación)
  engine.py           junta todo en eventos causales
  plan.py             entrada, stop, objetivo, T1, obstáculos, resultado simulado
  lab.py              analiza un valor → JSON para la web
  study.py            estudio de evidencia
  data.py, universe.py   descarga de Yahoo y universos
data/                 charts/<VALOR>.json, index.json, catalog.json, evidence.json
tests/test_vp.py      pruebas (sin red)
.github/workflows/    scan.yml (L–V 22:45 UTC) · evidence.yml (domingos) · research.yml (manual)
```

## Uso

```bash
pip install numpy pandas lxml
python tests/test_vp.py                       # pruebas
python -m vp lab                              # analiza la lista de seguimiento → data/
python -m vp lab --symbols AAPL,NVDA          # solo esos valores
python -m vp study                            # estudio de evidencia (tarda; mejor en Actions)
python -m http.server                         # abrir http://localhost:8000
```

Los precios vienen de Yahoo Finance (split aplicado, sin ajustar por dividendos para que los niveles coincidan con los del bróker); se descarta la barra de la sesión aún abierta.
