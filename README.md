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
- **Clasificación por pendientes y convergencia**: triángulos (simétrico, ascendente, descendente), cuñas y rectángulo con **≥ 5 toques y ≥ 3 en una de las rectas** (con 2 + 2 cualquier cuatro pivotes «forman» un triángulo); la recta «plana» se desvía ≤ 15 % de la base; duración ≤ 80 barras, base ≤ 16 % del precio y ≤ 9 ATR, anchura al romper ≤ 70 % de la inicial, con ápice.
- **Dobles y triples** techos y suelos con extremos de verdad iguales: dispersión ≤ 18 % de la altura de la figura y ≤ 0,75 ATR; en los triples, sin deriva en diagonal y con los valles intermedios (el cuello) también planos (≤ 35 % de la altura); al menos 10 barras entre extremos en los dobles y 16 en los triples.
- **H-C-H**: cabeza ≥ máx(1,5 ATR, 20 % de la profundidad) sobre ambos hombros, hombros a ≤ 22 % de la profundidad entre sí, cuello casi plano (≤ 33 %), simetría de tiempo entre 0,5 y 2 y tendencia previa.
- **Banderas y banderines** con asta impulsiva (≥ 5 ATR) y retroceso ≤ 50 %; **taza con asa** con la parábola ajustada a los mínimos y prueba de redondez; **VCP** con tres contracciones decrecientes bajo un techo casi plano (los tres máximos a ≤ 6 % y ≤ 2,5 ATR); **diamante**, **isla**, **directrices** con ≥ 3 toques y **rebotes en soportes y resistencias** de 3 toques.
- **Calidad 0–100** por figura (toques, ajuste, simetría, duración, forma, ruptura, volumen); por debajo de 35 no es figura.
- **Canales**: dos rectas paralelas con pendiente y ≥ 6 toques; la ruptura a favor acelera la tendencia, la contraria la agota.
- **Primera ruptura**: el cuello (o la taza, o la figura de compresión) solo se rompe la primera vez que el cierre lo pasa; un nuevo cruce tras un fallo no es la figura otra vez.
- **Figuras en formación**: las figuras ya dibujadas que aún no han roto se devuelven aparte, evaluadas como si la próxima barra fuera la de ruptura y con los mismos criterios. Una compresión (triángulo, cuña, canal, rectángulo) tiene dos escenarios (rompe arriba / rompe abajo); doble y triple suelo o techo, H-C-H, bandera y banderín, taza con asa y VCP tienen uno, con su **disparador** (cuello, borde o techo) y la **distancia en ATR** (≤ 3 ATR y calidad ≥ 50). Es una lectura del último cierre: no aparece en el estudio, que solo mide figuras ya rotas.

Además, **65 formaciones de vela** (30 clásicas y 31 más: cinturón, kicker, contraataque, harami cruz, paloma y halcón, tres exteriores, tres golpes, bebé abandonado, bloqueo de avance, tres métodos, tasuki, sándwich, vela exterior y ruptura de vela interior, entre otras) con **4 de consenso** (dos o más formaciones del mismo sentido y ninguna contraria en la misma sesión, o tres o más), y **13 rupturas** (máximos y mínimos de 55/100/252 sesiones, Bollinger y ATR comprimidos, caja de Darvas, base junto al máximo anual). El catálogo completo (`vp/catalog.py`, 110 patrones) alimenta el motor, el estudio y la Guía.

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

## Resultados (estudio del 25-oct-2026: 500 acciones, 1,23 millones de barras, 2016–2026; se recalcula cada domingo)

Con las **300 más negociadas** de cada fecha y el umbral del mejor de 130 variantes (|t| ≥ 3,46), **6 superan el azar y cuatro de ellas rinden MENOS que entrar al azar**:

| patrón | exceso a 10 sesiones | t | años a favor |
|---|---|---|---|
| Base junto a máximos (compra) | −0,53 % | −5,4 | 1/10 |
| Ruptura del máximo de 100 sesiones (compra) | −0,44 % | −4,9 | 1/10 |
| Ruptura del máximo de 55 sesiones (compra) | −0,38 % | −4,5 | 1/10 |
| Ruptura del máximo anual (compra) | −0,45 % | −4,3 | 2/10 |
| **Cuña ascendente (corto)** | **+0,60 %** | **+4,6** | **9/10** |
| **Triángulo simétrico roto a la baja (corto)** | **+0,99 %** | **+3,5** | **6/10** |

- **Comprar rupturas a máximos pierde frente al azar** (la reversión a corto plazo se come la ruptura).
- **Las dos figuras con ventaja son bajistas**: tras romperse a la baja, la cuña ascendente y el triángulo simétrico rinden 0,6 y 1,0 puntos menos que una acción cualquiera esos días. Pero la operación en corto de la cuña **pierde de media un 0,12 %** (gana menos de lo que pierde el mercado al alza): no es rentable por sí sola, y el estudio no incluye coste de préstamo ni dividendos. El triángulo simétrico bajista pasa el umbral por un pelo (3,5 contra 3,46) y 6 años de 10: está por confirmar.
- Con las **50 mayores** solo supera el umbral la base junto a máximos (negativa).
- **Triples y dobles con extremos realmente iguales** (n entre 200 y 1.600): triple suelo +0,48 % (t +1,0), triple techo −0,43 % (t −0,9), doble suelo −0,19 %, doble techo −0,04 %, H-C-H +0,10 % en corto (el invertido +0,26 %): indistinguibles del azar. Afinar la geometría deja la figura bien dibujada, no más fiable.
- **VCP con techo plano**: con la definición anterior (más laxa) salía significativamente negativo (−0,39 %, t −3,5); con tres máximos casi iguales queda en −0,27 % (t −1,7, n 1.600) y ya no supera el umbral.
- Por familia (exceso mediano a 10 sesiones): figuras alcistas −0,19 % (solo el 27 % de ellas por encima del azar), figuras bajistas +0,10 %, rupturas alcistas −0,38 %, rupturas bajistas −0,37 %, velas alcistas −0,11 % y bajistas +0,12 %. **Ninguna vela, de las 65, supera el umbral.** El mejor caso aislado es el kicker alcista (+1,76 %, t +2,4, 78 señales), que con 130 patrones probados es lo que sale por azar.
- **Consenso de velas**: dos formaciones del mismo sentido (+0,01 % alcista, +0,14 % bajista) o tres o más (+0,28 % y +0,24 %) mejoran algo el exceso de las velas sueltas (más con tres que con dos), pero ninguno es significativo (t entre 0,0 y 1,0): confluir ayuda a dibujar, no demuestra ventaja.
- **¿Ayuda dibujar mejor la figura?** Las señales de calidad ≥ 75 mejoran el exceso en 11 de 24 figuras (media +0,10 %): compatible con ruido.
- Una versión anterior del estudio (otro código y un universo con pertenencia histórica a índices) llegó a cifras casi idénticas en lo que no ha cambiado (base junto a máximos −0,57 %, cuña ascendente +0,52 %): el resultado no depende de la implementación.

**Qué es entonces esta herramienta**: una forma rigurosa de *leer* el gráfico (qué figura hay, dónde está el stop, cuánto hay hasta el objetivo, qué barreras hay en medio) y de ver, con datos, que las figuras clásicas por sí solas no baten al mercado. No es un generador de señales.

### ¿Se fía uno de estas estadísticas? Comprobaciones del propio método

- Con un efecto plantado de +3 % el estudio lo detecta (t = 16); con 24 paseos aleatorios no sale ningún patrón «significativo» de 66 (prueba automática).
- Control positivo con datos reales: en las mismas fechas, el factor clásico de impulso (rendimiento a 120 sesiones) deja al 10 % mejor +0,71 % por encima del resto a 10 sesiones (t +3,2): el método detecta efectos pequeños reales cuando los hay.
- El umbral del mejor de N corrige que, entre 130 patrones, alguno saldrá «significativo» por casualidad. Los porcentajes de acierto de casi todos los patrones rondan el 50 % porque las acciones grandes y líquidas son muy eficientes: es lo que cabe esperar de datos reales, no un fallo del cálculo.

## Aprendizaje automático (`vp/ml.py`): ¿combinar todo mejora?

`python -m vp ml` entrena un gradient boosting para ordenar las 300 acciones más negociadas de cada día por su rendimiento relativo a 10 sesiones, con **24 variables de contexto** (rendimientos a 1–250 sesiones, RSI, distancias a medias, rango anual, volatilidad, volumen, forma de la vela; todo como percentil del día) y **118 de patrones** (las 65 velas y su consenso de hoy; figuras y rupturas de las últimas 5 sesiones). Reglas para no engañarse:

- **Validación hacia delante**: para predecir el año Y se entrena solo con lo anterior a enero de Y (menos 15 días) y nunca se vuelve atrás.
- **Desarrollo y final separados**: 5 configuraciones fijadas de antemano; la elegida (la «grande») sale de los años 2020–2023 y se **juzga una sola vez en 2024–2026**, que no se toca.
- Cuatro conjuntos de variables (solo contexto, + velas, + figuras, todo) para ver si los patrones aportan algo, y factores simples (reversión, impulso, RSI, volatilidad) como control. Una puntuación al azar da IC ≈ 0.
- Medidas por fecha con t de Newey–West (los rendimientos a 10 sesiones de días contiguos se solapan).

Resultado en los **años finales (2024–2026)**:

| modelo | IC | mejor 10 % (a 10 sesiones) | t | neto de costes |
|---|---|---|---|---|
| solo contexto | +0,024 | +0,78 % | +2,6 | +0,63 % |
| contexto + velas (consenso incluido) | +0,026 | +0,77 % | +2,6 | +0,62 % |
| contexto + figuras y rupturas | +0,025 | +0,76 % | +2,6 | +0,61 % |
| todo | +0,024 | +0,73 % | +2,5 | +0,58 % |
| *referencia: ranking por impulso (r120)* | *+0,015* | *+1,18 %* | *+2,8* | *+1,03 %* |

- **Los patrones no añaden nada al modelo**: las diferencias de IC entre los cuatro conjuntos (±0,002) son ruido, tanto en los años finales como en los de desarrollo. Ni las velas, ni su consenso, ni las figuras.
- El contexto sí da una señal pequeña (IC ≈ 0,02 positivo los 7 años; el 10 % mejor rinde entre +0,3 % y +0,8 % más que el resto a 10 sesiones y gana en 6 de 7 años), pero **no mejora a un ranking por impulso** y 2022 salió plano. Con t entre 2 y 3 y varios modelos probados, es una pista, no una conclusión.
- No hay puntuación de aprendizaje automático en la web: no hay mejora demostrada que justifique enseñarla. El informe completo está en `data/ml_report.txt`.

## ¿Los stops y objetivos grandes explican el mal resultado? (`vp/planstudy.py`)

Una objeción razonable: «quizá no funcionan porque el stop y el objetivo son enormes». El estudio de evidencia no usa stops ni objetivos (salida a tiempo fijo), así que no puede ser la causa de su resultado; pero para comprobar el plan completo, `python -m vp plan` simula cada señal como la web (entrada en la apertura siguiente, stop y objetivo del plan, el stop gana si se tocan el mismo día, cierre a las 40 sesiones, coste del 0,15 %) y la compara con **8 entradas al azar en el mismo valor con el mismo sentido y las mismas distancias de stop y objetivo en ATR** (y volatilidad relativa parecida). Tres geometrías: el plan de la web, stop táctico de 1,5 ATR con el mismo objetivo, y una **fija igual para todos** (stop 1,5 ATR, objetivo 3 ATR). Informe en `data/plan_report.txt`.

Con las 300 más negociadas (220.000 señales, 2016–2026) el exceso medio sobre el azar con las mismas distancias es **−0,15 % con el plan de la web, −0,13 % con stop táctico y −0,11 % con la geometría fija**. Calibrado sobre paseos aleatorios, donde no puede haber ventaja, el método sesga el resultado en −0,05 a −0,08 %, así que lo que queda es una pérdida pequeña, no una ventaja escondida por el tamaño de los stops:

| grupo (300 más negociadas) | plan web | stop táctico | fijo 1,5/3 ATR |
|---|---|---|---|
| todos | −0,15 % (t −2,0) | −0,13 % (t −2,2) | −0,11 % (t −2,0) |
| figuras alcistas | −0,29 % (t −2,2) | −0,12 % | −0,09 % |
| figuras bajistas | +0,07 % | +0,14 % | +0,12 % |
| rupturas alcistas | −0,49 % (t −6,3) | −0,42 % (t −6,5) | −0,36 % (t −6,0) |
| velas | ≈ 0 (+0,09 / +0,10) | ≈ 0 (−0,07 / +0,07) | ≈ 0 (−0,10 / +0,10) |
| directrices alcistas | +0,33 % (t +2,1) | +0,33 % | +0,33 % |

- **Comprar rupturas a máximos** (55, 100 y 252 sesiones, base junto a máximos) rinde menos que entrar al azar con cualquier geometría (t entre −5,0 y −6,8): coincide con el estudio de evidencia.
- La única señal positiva que se repite en las tres geometrías es la **ruptura de una directriz bajista** (+0,33 %, t +2,1), pero con 104 patrones probados no supera el umbral del mejor de N (3,46).
- El tamaño del plan fija la **tasa de acierto**: con el plan de la web las figuras alcistas llegan al objetivo el 30 % de las veces, pero entrar al azar con las mismas distancias lo logra el 31 %. La tasa de acierto la decide la geometría, no el patrón.

### Stop que persigue (trailing) en lugar de objetivo fijo

La idea de muchos operadores discrecionales es «dejar correr las ganancias»: sin objetivo, con un stop que sube a 3 o 5 ATR del mejor cierre (60 sesiones como máximo). `python -m vp plan` lo simula (`trail3`, `trail5`) y lo compara con entradas al azar con la misma regla de salida:

| grupo (50 más negociadas) | stop 3 ATR | stop 5 ATR |
|---|---|---|
| todos | −0,36 % (t −2,4) | −0,62 % (t −3,3) |
| rupturas alcistas | −0,99 % (t −4,6) | −1,55 % (t −5,0) |
| velas bajistas | +0,37 % (t +1,8) | +0,73 % (t +2,6) |
| rebotes bajistas | +1,62 % (t +2,8) | +2,30 % (t +2,9) |

El stop que persigue no rescata ninguna familia: las rupturas alcistas siguen rindiendo menos que entrar al azar, y lo poco positivo (velas bajistas, rebotes bajistas) no supera el umbral del mejor de N con 86 patrones probados. Calibrado sobre paseos aleatorios el método sesga −0,12 a −0,17 %.

## Otros mercados: ¿es que las acciones grandes son demasiado eficientes?

`python -m vp study --universe smallmid|mixed` repite el estudio de evidencia con las mismas reglas en mercados menos eficientes (informes `data/evidence_report_smallmid.txt` y `data/evidence_report_mixed.txt`):

- **Acciones medianas y pequeñas** (S&P 400 + S&P 600, 989 valores, 2,3 millones de barras, coste del 0,30 %): lo que pasa el umbral del mejor de N son otra vez las **rupturas alcistas a máximos, por debajo del azar** (−0,5 %, t entre −4,6 y −5,5) y la base junto a máximos. Ninguna vela, figura ni consenso lo supera. Las velas bajistas salen todas algo positivas (+0,47 % de media, 88 % de ellas positivas) y las alcistas todas negativas (−0,84 %): que las dos direcciones sean «correctas» a la vez no es una ventaja direccional: lo más probable (no comprobado) es un efecto de selección, porque las velas grandes se dan en valores volátiles que rinden menos que la media del año, y el comparador del estudio de evidencia no iguala la volatilidad (el del plan sí).
- **Cripto, divisas, materias primas e índices** (49 símbolos): solo supera el umbral la base junto a máximos (negativa, t −6,5). Las rupturas alcistas de máximos salen algo positivas aquí (+0,8 a +0,9 %, t entre +2,2 y +2,6, 6 o 7 años de 10), lo contrario que en acciones: con 49 series y 103 patrones, una pista que tendría que repetirse en datos nuevos.

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
  candles.py          65 formaciones de vela y 4 de consenso
  breakouts.py        13 rupturas y compresiones
  shapes.py           motor geométrico de figuras (swings, rectas por toques, calidad, formación)
  engine.py           junta todo en eventos causales
  plan.py             entrada, stop, objetivo, T1, obstáculos, resultado simulado
  lab.py              analiza un valor → JSON para la web
  study.py            estudio de evidencia
  planstudy.py        estudio del plan completo (stop y objetivo) frente a entradas al azar
  ml.py               aprendizaje automático fuera de muestra (contexto, velas, figuras)
  data.py, universe.py   descarga de Yahoo y universos (watchlist.txt)
data/                 charts/<VALOR>.json, index.json, catalog.json, evidence.json, plan_report.txt, ml_report.txt
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
python -m vp ml                               # aprendizaje automático fuera de muestra (necesita scikit-learn)
python -m http.server                         # abrir http://localhost:8000
```

Los precios vienen de Yahoo Finance (split aplicado, sin ajustar por dividendos para que los niveles coincidan con los del bróker); se descarta la barra de la sesión aún abierta.
