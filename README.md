# VP Scanner · laboratorio de chartismo

Detecta **figuras, velas y rupturas** en acciones grandes de EE.UU., las **dibuja** sobre el gráfico con su plan (entrada, stop, objetivo) y **mide si funcionan**.

- **Radar**: qué figuras se están **formando** (con el nivel que las activa y a cuántos ATR está el precio), cuáles son **nuevas hoy** y cuáles se completaron hace poco; filtros «cerca del disparo», sentido, calidad y ★ favoritos.
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
- **Canales**: dos rectas paralelas con pendiente y ≥ 6 toques; la ruptura a favor acelera la tendencia, la contraria la agota.
- **Primera ruptura**: el cuello (o la taza, o la figura de compresión) solo se rompe la primera vez que el cierre lo pasa; un nuevo cruce tras un fallo no es la figura otra vez.
- **Figuras en formación**: las figuras ya dibujadas que aún no han roto se devuelven aparte, evaluadas como si la próxima barra fuera la de ruptura y con los mismos criterios. Una compresión (triángulo, cuña, canal, rectángulo) tiene dos escenarios (rompe arriba / rompe abajo); doble y triple suelo o techo, H-C-H, bandera y banderín, taza con asa y VCP tienen uno, con su **disparador** (cuello, borde o techo) y la **distancia en ATR** (≤ 3 ATR y calidad ≥ 50). Es una lectura del último cierre: no aparece en el estudio, que solo mide figuras ya rotas.

Además, 30 patrones de velas y 13 rupturas (máximos y mínimos de 55/100/252 sesiones, Bollinger y ATR comprimidos, caja de Darvas, base junto al máximo anual). El catálogo completo (`vp/catalog.py`) alimenta el motor, el estudio y la Guía.

## El plan de cada patrón (`vp/plan.py`)

- **Objetivo** por la medida clásica de la figura (doble suelo: cuello + altura; triángulos: anchura de la base; banderas: el asta; taza: su profundidad…) y **T1** al 50 % del camino.
- **Stop** estructural más allá de la figura (0,25 ATR de margen), **nunca a menos de 1 ATR** de la entrada y, si queda a más de 3 ATR, un stop táctico de 1,5 ATR.
- **Obstáculos**: soportes y resistencias fuertes entre la entrada y el objetivo.
- **Resultado simulado**: entrada en la apertura siguiente, el stop gana si se tocan ambos el mismo día, cierre a las 40 sesiones.
- **Objetivo consumido**: si la ruptura ya recorrió la medida clásica (queda menos de 1 ATR hasta el objetivo) el patrón se marca como consumido y no cuenta como activo.

## Estudio de evidencia (`vp/study.py`)

- Universo: miembros actuales del S&P 500 y el Nasdaq-100; se usan las **50 y las 300 más negociadas en cada fecha** (dinero medio de 60 sesiones).
- Cada señal se opera en la apertura siguiente y se cierra a 5, **10** (principal) y 20 sesiones, con un coste del 0,15 % de ida y vuelta; los patrones bajistas se miden en corto.
- «Exceso» = rendimiento − media de entrar al azar en el mismo universo el mismo año; **t con errores estándar agrupados por día**.
- **Umbral del mejor de N**: se desplazan al azar las señales de cada valor y se anota el mayor |t| entre todos los patrones; solo «supera el azar» quien pasa el percentil 95 de ese máximo. Se estudian también las señales de **calidad ≥ 75** de cada figura.
- **Limitación**: los miembros son los actuales (sesgo de supervivencia que favorece a los largos).

## Resultados (estudio del 24-oct-2026: 500 acciones, 1,23 millones de barras, 2016–2026; se recalcula cada domingo)

Con las **300 más negociadas** de cada fecha y el umbral del mejor de 101 variantes (|t| ≥ 3,35), **8 superan el azar; seis de ellas rinden MENOS que entrar al azar**:

| patrón | exceso a 10 sesiones | t | años a favor |
|---|---|---|---|
| Base junto a máximos (compra) | −0,53 % | −5,4 | 1/10 |
| Ruptura del máximo de 100 sesiones (compra) | −0,44 % | −4,9 | 1/10 |
| Ruptura del máximo de 55 sesiones (compra) | −0,38 % | −4,5 | 1/10 |
| Ruptura del máximo anual (compra) | −0,45 % | −4,3 | 2/10 |
| VCP (compra) | −0,39 % | −3,5 | 0/10 |
| **Cuña ascendente (corto)** | **+0,60 %** | **+4,6** | **9/10** |

- **La única figura con ventaja es la cuña ascendente (bajista)**: tras romperse a la baja, la acción rinde 0,60 puntos menos de lo que rinde una acción cualquiera esos días. Pero la operación en corto **pierde de media un 0,12 %** (gana menos de lo que pierde el mercado al alza): no es rentable por sí sola, y el estudio no incluye coste de préstamo ni dividendos.
- **Comprar rupturas a máximos pierde frente al azar** (la reversión a corto plazo se come la ruptura); el VCP, la taza con asa y la base junto a máximos también.
- Con las **50 mayores** solo supera el umbral la base junto a máximos (negativa).
- Por familia (exceso mediano): figuras alcistas −0,22 % (solo el 20 % de ellas por encima del azar), figuras bajistas +0,03 %, rupturas alcistas −0,38 %, rupturas bajistas −0,37 %, velas alcistas 0,00 % y bajistas +0,22 %. Ninguna vela individual supera el umbral.
- **Los canales** (nuevos) no aportan nada: canal ascendente roto al alza −0,50 % (t −1,7), roto a la baja +0,36 % (t +1,3), descendente −0,73 % y +0,08 %; ninguno supera el umbral. Doble suelo −0,15 %, doble techo −0,18 %, H-C-H −0,18 % y su invertido +0,11 %, bandera alcista −0,06 %: todos indistinguibles del azar.
- **¿Ayuda dibujar mejor la figura?** Las señales de calidad ≥ 75 mejoran el exceso en 13 de 24 figuras (media +0,11 %), pero con 100–400 señales por celda eso es compatible con ruido, y no rescata a las que pierden (VCP −0,39 → −0,40, taza −0,82 → −0,83, triángulo ascendente −0,40 → −0,53).
- Una versión anterior del estudio (otro código y un universo con pertenencia histórica a índices) llegó a cifras casi idénticas (base junto a máximos −0,57 %, VCP −0,54 %, cuña ascendente +0,52 %): el resultado no depende de la implementación.

**Qué es entonces esta herramienta**: una forma rigurosa de *leer* el gráfico (qué figura hay, dónde está el stop, cuánto hay hasta el objetivo, qué barreras hay en medio) y de ver, con datos, que las figuras clásicas por sí solas no baten al mercado. No es un generador de señales.


## ¿Los stops y objetivos grandes explican el mal resultado? (`vp/planstudy.py`)

Una objeción razonable: «quizá no funcionan porque el stop y el objetivo son enormes». El estudio de evidencia no usa stops ni objetivos (salida a tiempo fijo), así que no puede ser la causa de su resultado; pero para comprobar el plan completo, `python -m vp plan` simula cada señal como la web (entrada en la apertura siguiente, stop y objetivo del plan, el stop gana si se tocan el mismo día, cierre a las 40 sesiones, coste del 0,15 %) y la compara con **8 entradas al azar en el mismo valor con el mismo sentido y las mismas distancias de stop y objetivo en ATR** (y volatilidad relativa parecida). Tres geometrías: el plan de la web, stop táctico de 1,5 ATR con el mismo objetivo, y una **fija igual para todos** (stop 1,5 ATR, objetivo 3 ATR). Informe en `data/plan_report.txt`.

Con las 300 más negociadas (203.000 señales, 2016–2026) el exceso medio sobre el azar con las mismas distancias es **−0,20 % con el plan de la web, −0,13 % con stop táctico y −0,11 % con la geometría fija**. Calibrado sobre paseos aleatorios, donde no puede haber ventaja, el método sesga el resultado en −0,05 a −0,07 %, así que lo que queda es una pérdida pequeña, no una ventaja escondida por el tamaño de los stops:

| grupo (300 más negociadas) | plan web | stop táctico | fijo 1,5/3 ATR |
|---|---|---|---|
| todos | −0,20 % (t −2,9) | −0,13 % (t −2,4) | −0,11 % (t −2,1) |
| figuras alcistas | −0,40 % (t −3,4) | −0,15 % | −0,13 % |
| figuras bajistas | −0,09 % | +0,05 % | +0,05 % |
| rupturas alcistas | −0,48 % (t −6,1) | −0,40 % (t −6,3) | −0,35 % (t −5,8) |
| velas | ≈ 0 | ≈ 0 | ≈ 0 |
| directrices alcistas | +0,42 % (t +2,6) | +0,42 % | +0,42 % |

- **Comprar rupturas a máximos** (55, 100 y 252 sesiones, base junto a máximos) rinde menos que entrar al azar con cualquier geometría (t entre −4,4 y −6,9): coincide con el estudio de evidencia.
- La única señal positiva que se repite en las tres geometrías es la **ruptura de una directriz bajista** (+0,42 %, t +2,6), pero con 75 patrones probados no supera el umbral del mejor de N (3,35).
- El tamaño del plan sí cambia la **tasa de acierto** (29 % de objetivos en las figuras con el plan de la web), pero la del azar con las mismas distancias es casi igual (26–36 %): la tasa de acierto la fija la geometría, no el patrón.

## Lista de seguimiento

`watchlist.txt` (raíz del repositorio) manda: un símbolo de Yahoo por línea o varios separados por espacios o comas, `#` comenta. El escaneo diario la lee entera; para añadir o quitar valores basta editarla desde GitHub (lápiz) y el siguiente escaneo, o lanzarlo a mano desde Actions, los incluye. Un valor que Yahoo no devuelve o con menos de 400 sesiones se avisa en el Radar. Cada valor añade ~75 KB de `data/charts` al repositorio en cada escaneo, así que conviene no pasar de unos 150.

Los ★ favoritos se guardan solo en el navegador (localStorage).

## Estructura

```
index.html            la web (lee data/*.json)
watchlist.txt         valores que vigila el escaneo diario (editable)
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
  planstudy.py        estudio del plan completo (stop y objetivo) frente a entradas al azar
  data.py, universe.py   descarga de Yahoo y universos (watchlist.txt)
data/                 charts/<VALOR>.json, index.json, catalog.json, evidence.json, plan_report.txt
tests/test_vp.py      pruebas (sin red)
.github/workflows/    scan.yml (L–V 22:45 UTC) · evidence.yml (domingos) · research.yml (manual)
```

## Uso

```bash
pip install numpy pandas lxml
python tests/test_vp.py                       # pruebas
python -m vp lab                              # analiza watchlist.txt → data/
python -m vp lab --symbols AAPL,NVDA          # solo esos valores
python -m vp study                            # estudio de evidencia (tarda; mejor en Actions)
python -m vp plan                             # estudio del plan completo con stop y objetivo
python -m http.server                         # abrir http://localhost:8000
```

Los precios vienen de Yahoo Finance (split aplicado, sin ajustar por dividendos para que los niveles coincidan con los del bróker); se descarta la barra de la sesión aún abierta.
