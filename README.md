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

Además, **61 formaciones de vela** (30 clásicas y 31 más: cinturón, kicker, contraataque, harami cruz, paloma y halcón, tres exteriores, tres golpes, bebé abandonado, bloqueo de avance, tres métodos, tasuki, sándwich, vela exterior y ruptura de vela interior, entre otras) con **4 de consenso** (dos o más formaciones del mismo sentido y ninguna contraria en la misma sesión, o tres o más), y **13 rupturas** (máximos y mínimos de 55/100/252 sesiones, Bollinger y ATR comprimidos, caja de Darvas, base junto al máximo anual). El catálogo completo (`vp/catalog.py`, 110 patrones) alimenta el motor, el estudio y la Guía.

## El plan de cada patrón (`vp/plan.py`)

El plan que se dibuja, se simula y se cuenta es **el plan del operador** (el que describe quien opera chartismo con ganancias modestas: «con un 4–5 % soy feliz; si sube más lo dejo correr, pero pasado el 4 % pongo un stop para asegurarlo»). Las reglas están fijadas de antemano, no optimizadas con los resultados:

- **Entrada**: apertura de la sesión siguiente a la que completa el patrón.
- **Stop inicial**: el estructural (0,25 ATR al otro lado de la figura) si queda entre **1 y 2 ATR** de la entrada; si está más lejos se **acorta a 2 ATR** (en una acción de volatilidad media, ~5 %); nunca a menos de 1 ATR. El plan clásico daba stops del 10, 26 o 43 %.
- **Asegurar +4 %**: cuando una sesión **cierra a +4 % o más** a favor, el stop sube a **+4 %** (el nivel asegurado) y desde ahí **persigue al mejor cierre a 3 ATR**, sin bajar nunca de lo asegurado. No hay objetivo fijo: si sigue subiendo se deja correr. Máximo 60 sesiones; el stop gana si se tocan ambos el mismo día y un hueco de apertura sale a la apertura.
- **Ninguna señal se descarta**. Antes había un estado «no operable» (recompensa menor de 1,5 veces el riesgo, riesgo de más del 8 % del precio, medida consumida) que quitaba señales de la cuenta; era un error de planteamiento: un umbral de calidad estadística sirve para **graduar** cuánta evidencia hay, no para tirar operaciones. Ahora el riesgo alto (> 8 % del precio), un stop más lejos que el +4 % y la medida ya recorrida son **avisos informativos** en la ficha («la señal sigue contando»).
- **Tamaño**: la ficha dice cuánta posición corresponde a arriesgar el 1 % de la cuenta con ese stop (1 / riesgo %): lo que importa para el riesgo no es la distancia del stop sino el tamaño de la posición.
- El **plan clásico sin límites** (stop al otro lado de la figura y objetivo por la medida clásica, T1 al 50 %) se conserva como referencia y se ve con el interruptor «Plan clásico» del gráfico.
- **Obstáculos**: soportes y resistencias fuertes en el camino.
- **Quién gana y quién pierde**: gana la operación que **termina con ganancia neta** (después del 0,15 % de costes del estudio), llegue o no a asegurar. Asegurar +4 %, saltar el stop inicial o cerrar por tiempo es solo cómo terminó. De cada operación se guardan la ganancia neta, el **múltiplo del riesgo (R)**, el recorrido máximo a favor y en contra y si llegó a asegurar. La Lectura de cada valor da el porcentaje de ganadoras, las veces que se aseguró, la **ganancia media de las ganadoras y la pérdida media de las perdedoras**.
- **Una operación por valor** (`group_trades` en `vp/lab.py`): cuando varias señales saltan a la vez (velas y figuras, o varias velas) **no se cuenta cada una**. La primera señal (si coinciden varias en la misma barra, la de mayor rango: figura > ruptura > directriz > rebote > vela, y a igualdad la de mayor calidad) abre la operación y su plan es el que se simula; las del mismo sentido mientras está abierta son **confirmaciones** y las del sentido contrario son **contrarias** (aviso de conflicto, tampoco son operación propia). No se descarta ninguna señal: las confirmaciones y contrarias se muestran, pero no abren una segunda operación sobre el mismo valor.

## Estudio de evidencia (`vp/study.py`)

- Universo: miembros actuales del S&P 500 y el Nasdaq-100; se usan las **50 y las 300 más negociadas en cada fecha** (dinero medio de 60 sesiones).
- Cada señal se opera en la apertura siguiente y se cierra a 5, **10** (principal) y 20 sesiones, con un coste del 0,15 % de ida y vuelta; los patrones bajistas se miden en corto. «% ganadoras» es la parte de operaciones con ganancia neta, no la de las que llegan a ningún objetivo.
- «Exceso» = rendimiento − media de entrar al azar **ese mismo día** en el mismo universo y sentido: lo que haga el mercado en bloque (un patrón que salta justo antes de un rebote general) no cuenta como ventaja. Se guarda también el exceso frente a la media del **año**, que era la referencia de las primeras versiones (columna «vs año»).
- **t con errores estándar agrupados en bloques de 10 sesiones** (antes, por día): los rendimientos a 10 sesiones de días contiguos se solapan y las señales vienen en rachas. En los datos reales el cambio recorta la t casi a la mitad (ver abajo).
- **Umbral del mejor de N**: se desplazan al azar las señales de cada valor y se anota el mayor |t| entre todos los patrones; solo «supera el azar» quien pasa el percentil 95 de ese máximo. Se estudian también las señales de **calidad ≥ 75** de cada figura.
- **Coincidencias**: cada patrón se mide con todas sus señales, también las que saltan a la vez que otro patrón; por eso hay una tabla aparte donde **cada valor-día cuenta una sola vez** (solo velas, solo estructura, ambas, contradictorias).
- **Limitación**: los miembros son los actuales (sesgo de supervivencia que favorece a los largos).

## Resultados (estudio del 27-oct-2026: 500 acciones, 1,23 millones de barras, 2016–2026; se recalcula cada domingo)

**Ningún patrón supera el umbral del mejor de N**, ni con las 300 más negociadas (130 variantes, |t| ≥ 3,6) ni con las 50 (102 variantes, |t| ≥ 3,5). La versión anterior del estudio decía que **6 lo superaban**; al corregir dos defectos del método, se quedan en cero. Así cambió cada uno de esos seis (300 más negociadas, exceso a 10 sesiones):

| patrón | antes (vs el año, t por día) | solo con t por bloques (vs el año) | ahora (vs el mismo día, t por bloques) |
|---|---|---|---|
| Base junto a máximos (compra) | −0,53 % (t −5,4) | −0,55 % (t −2,8) | **−0,09 % (t −0,8)** |
| Ruptura del máximo de 100 sesiones (compra) | −0,44 % (t −4,9) | −0,43 % (t −2,6) | **−0,06 % (t −0,6)** |
| Ruptura del máximo de 55 sesiones (compra) | −0,38 % (t −4,5) | −0,39 % (t −2,5) | **−0,06 % (t −0,7)** |
| Ruptura del máximo anual (compra) | −0,45 % (t −4,3) | −0,46 % (t −2,4) | **−0,05 % (t −0,5)** |
| Cuña ascendente (corto) | +0,60 % (t +4,6) | +0,61 % (t +3,2) | **+0,28 % (t +2,7)** |
| Triángulo simétrico roto a la baja (corto) | +0,99 % (t +3,5) | +0,97 % (t +3,0) | **+0,91 % (t +3,5)** |

- **La t estaba exagerada.** Agrupar por bloques de 10 sesiones en vez de por día (columna central) baja la t de las rupturas de −4,5 a −2,5 y la de la cuña de +4,6 a +3,2: los rendimientos a 10 sesiones de días vecinos se solapan y no son independientes.
- **«Comprar rupturas pierde frente al azar» era sobre todo el momento del mercado.** Frente a entrar al azar en **el mismo año**, las rupturas alcistas rinden ~0,4 % menos; frente a entrar al azar **el mismo día** la diferencia es de ~0,06 % (t ≈ −0,6): saltan en días de mercado fuerte y justo después el mercado devuelve parte. Quien compra rupturas sin mirar el mercado lo sufre, pero no es que el valor que rompe rinda peor que otro cualquiera ese día.
- **Las dos figuras bajistas** son lo que más se acerca: la cuña ascendente (+0,28 %, t +2,7, 8 años de 10) y el triángulo simétrico roto a la baja (+0,91 %, t +3,5 con umbral 3,6, 10 años de 10 pero solo 239 señales). Pasan casi el umbral, no lo superan: «para confirmar» con más datos. Y siguen sin ser rentables por sí solas en corto (+0,22 % y −0,11 % de media neta, sin coste de préstamo).
- **Triples y dobles con extremos realmente iguales** (n entre 200 y 1.600): triple suelo +0,57 % (t +1,3), triple techo −0,17 % (t −0,4), doble suelo +0,01 %, doble techo −0,12 %, H-C-H +0,10 % en corto y el invertido +0,48 % (t +1,3): indistinguibles del azar. Afinar la geometría deja la figura bien dibujada, no más fiable.
- **VCP con techo plano**: −0,10 % (t −0,6, n 1.600).
- Por familia (exceso **mediano** a 10 sesiones): figuras alcistas −0,07 %, figuras bajistas −0,12 %, rupturas alcistas −0,06 %, rupturas bajistas −0,15 %, velas alcistas −0,05 % y bajistas −0,03 %. **Ninguna vela (61 formaciones y 4 de consenso) supera el umbral.** El mejor caso aislado es el kicker alcista (+0,78 %, t +1,0, 78 señales).
- **Consenso de velas**: dos o más formaciones del mismo sentido +0,05 % (alcista, t +0,4) y 0,00 % (bajista); tres o más +0,03 % y +0,12 %: sumar velas no añade nada.
- **Velas y figuras a la vez** (cada valor-día cuenta **una** vez; 300 más negociadas; 213.525 valor-días con señal, el 35 % con varias a la vez): señales alcistas solo velas −0,06 % (t −0,8, n 22.674), solo estructura −0,06 % (t −1,3, n 39.591), **ambas juntas +0,18 % (t +1,1, n 2.004)**, todas −0,06 % (t −1,6, n 64.269); bajistas −0,02 %, −0,11 %, 0,00 % y −0,06 %; las 2.144 contradictorias (alcistas y bajistas a la vez) tampoco anticipan nada (±0,18 %, t 1,0). Coincidir no mejora el resultado de forma apreciable, y no cuenta doble: es una sola operación.
- **¿Ayuda dibujar mejor la figura?** Las señales de calidad ≥ 75 no mejoran de forma consistente el exceso (compatible con ruido).
- Una versión anterior del estudio (otro código y un universo con pertenencia histórica a índices) llegó a cifras casi idénticas en lo que no ha cambiado: el resultado no depende de la implementación.

**Qué es entonces esta herramienta**: una forma rigurosa de *leer* el gráfico (qué figura hay, dónde está el stop inicial, dónde se asegura el +4 %, qué barreras hay en medio) y de ver, con datos, qué aporta cada patrón. No es un generador de señales.

**Sobre «el chartismo tiene evidencia estadística»**: la literatura es mixta, no vacía. Lo, Mamaysky y Wang (2000) encontraron que algunas figuras aportan información incremental pequeña sobre la distribución de rendimientos; otros trabajos que corrigen por el sesgo de probar muchas reglas y por costes (por ejemplo Bajgrowicz y Scaillet, 2012) concluyen que el beneficio neto desaparece en gran parte. Este laboratorio mide una pregunta concreta y acotada: con detectores automáticos fijados de antemano, en acciones grandes de EE.UU. de 2016 a 2026, con costes y frente a entrar sin patrón, ¿añaden algo las figuras, velas y rupturas? Su respuesta es que no se distingue de cero (potencia: se detectaría un efecto medio de ~0,1–0,2 % por familia, no uno menor). No mide la **habilidad discrecional** (elegir qué figuras valen, cuándo y con qué contexto, gestionar la posición), y esa no se puede descartar ni demostrar con un detector.

### ¿Y si el método fuera demasiado exigente? Potencia, controles y el conjunto de cada familia

«Ningún patrón supera el umbral» puede significar que no funcionan o que el estudio no tiene potencia para verlo. Se comprobó así (informe `data/evidence_report.txt`, 300 más negociadas):

- **El umbral es exigente a propósito** (|t| ≥ 3,6: el del mejor de 130 patrones). Con el error estándar real de cada patrón (mediana 0,28 % por operación), el estudio detecta con un 80 % de probabilidad efectos de **≥ 1,25 %** por operación en un patrón típico y de **≥ 8 %** en los 10 más raros. **No puede ver ventajas pequeñas en patrones concretos**, y la web muestra el intervalo del 95 % de cada patrón además de la t.
- **Control positivo con el mismo motor** (referencia del mismo día, bloques de 10 sesiones, mismo umbral): el 10 % de valores con mejor rendimiento a 120 sesiones (momentum) rinde **+0,54 % ± 0,32 (t +3,3, 71.580 señales)**: **tampoco supera el umbral** (con las 50 más negociadas, +1,01 % ± 0,56, t +3,5, queda justo en el umbral de 3,5). La volatilidad alta da +0,65 % (t +3,0), el momentum bajo +0,13 % y la sobreventa a corto +0,13 % (t +1,3). Un efecto real de ~0,5 % por operación queda en el límite: **«no supera el umbral» no significa «no funciona», significa «no se distingue del azar con estos datos»**.
- **Lo que sí se puede afirmar con precisión es el conjunto de cada familia** (todas sus señales juntas, cada valor-día una vez; frente a un valor cualquiera ese día; intervalos del 95 %): figuras alcistas +0,01 % ± 0,14, bajistas −0,04 % ± 0,17; rupturas alcistas −0,06 % ± 0,13, bajistas −0,15 % ± 0,20; velas alcistas −0,03 % ± 0,14, bajistas −0,02 % ± 0,11; directrices alcistas +0,08 % ± 0,22, bajistas −0,11 % ± 0,18; rebotes bajistas +0,35 % ± 0,28 (t +2,4: no pasa el umbral de Bonferroni de 2,8). **Ninguna familia entera mueve el resultado más de unas décimas de punto por operación, y el coste de ida y vuelta (0,15 %) es del mismo tamaño**: si existe una ventaja media, es demasiado pequeña para sobrevivir a los costes.
- **Lo que este estudio no puede probar**: que un operador con criterio añadido (qué patrones elige, contexto que no se puede escribir como regla, gestión de la posición) tenga ventaja. Mide reglas mecánicas.

### ¿Se fía uno de estas estadísticas? Comprobaciones del propio método

- Con un efecto plantado de +3 % en señales de valores distintos el estudio lo detecta (t = 34); con 24 paseos aleatorios no sale ningún patrón «significativo» de 66 (pruebas automáticas).
- La referencia del mismo día cancela lo que el mercado hace en bloque: si un efecto plantado saltara en todos los valores a la vez, el estudio lo trataría como mercado y no como patrón (por eso la prueba coloca el efecto en días distintos). Con simulaciones de paseos aleatorios y señales al azar (densidad del 2 %, 8 % y 30 %) tanto el t por día como el de bloques salen bien calibrados (|t| > 1,96 en 4,5–6 % de los casos); la diferencia aparece con datos reales, probablemente porque las señales vienen en rachas y el mercado tiene regímenes.
- Control positivo con datos reales: en las mismas fechas, el factor clásico de impulso (rendimiento a 120 sesiones) deja al 10 % mejor +0,71 % por encima del resto a 10 sesiones (t +3,2): el método detecta efectos pequeños reales cuando los hay.
- El umbral del mejor de N corrige que, entre 130 patrones, alguno saldrá «significativo» por casualidad. Los porcentajes de ganadoras de casi todos los patrones rondan el 50 % porque las acciones grandes y líquidas son muy eficientes: es lo que cabe esperar de datos reales, no un fallo del cálculo.
- Errores encontrados y corregidos en esta revisión: una «confluencia» que miraba también las dos sesiones siguientes (fuga de información); el recuento de ganancias por «llegar al objetivo» en lugar de por ganancia neta; señales solapadas contadas varias veces; la t por día y la referencia del año; «65 formaciones de vela» donde son 61 más 4 de consenso; descripciones de Bollinger y ATR que no decían lo que hace el código; un máximo por debajo del cierre en datos de Yahoo; y un parámetro `--reps` del contexto que se ignoraba.

## Aprendizaje automático (`vp/ml.py`): ¿combinar todo mejora?

`python -m vp ml` entrena un gradient boosting para ordenar las 300 acciones más negociadas de cada día por su rendimiento relativo a 10 sesiones, con **24 variables de contexto** (rendimientos a 1–250 sesiones, RSI, distancias a medias, rango anual, volatilidad, volumen, forma de la vela; todo como percentil del día) y **118 de patrones** (las 61 formaciones de vela y los 4 consensos de hoy; figuras y rupturas de las últimas 5 sesiones). Reglas para no engañarse:

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

## Tu regla de salida frente al azar: ¿asegurar +4 % y dejar correr cambia algo? (`vp/planstudy.py`)

La regla de muchos operadores («con un 4–5 % soy feliz; si sube más lo dejo correr, pero pasado el 4 % pongo un stop para asegurarlo») es razonable y es la que usa la web. Falta saber si, **aplicada a todas las señales sin descartar ninguna**, el patrón aporta algo frente a entrar sin patrón con esa misma regla. `python -m vp plan` simula cada señal como la web (entrada en la apertura siguiente, stop inicial acotado a 1–2 ATR, al cerrar a +4 % el stop sube a +4 % y persigue a 3 ATR, hasta 60 sesiones, el stop gana si se tocan el mismo día, coste del 0,15 %) y la compara con **8 entradas al azar en el mismo valor y sentido, con las mismas distancias de stop en ATR** (y volatilidad relativa parecida) y **exactamente la misma regla de salida**.

**Cómo se cuenta el éxito**: por la ganancia media neta y por el **% de operaciones ganadoras** (las que terminan con ganancia neta, lleguen o no a asegurar). El reparto asegurado / stop inicial / tiempo es solo descriptivo.

Resultado con todas las señales, estudio del 27-oct-2026 (`data/plan_report.txt`):

| regla de salida | universo | n | media neta | azar con la misma regla | exceso (t, bloques de 40–60 sesiones) | % ganadoras señal / azar |
|---|---|---|---|---|---|---|
| **asegurar +4 %** (stop inicial 1–2 ATR) | 300 más negociadas | 218.700 | −0,23 % | −0,04 % | **−0,18 % (t −2,8)** | 42 / 44 % |
| **asegurar +4 %** (stop inicial 1–2 ATR) | 50 más negociadas | 36.640 | −0,17 % | +0,13 % | **−0,30 % (t −3,8)** | 43 / 45 % |
| asegurar +5 % | 300 | 218.700 | −0,24 % | −0,04 % | −0,20 % (t −2,8) | 38 / 40 % |
| asegurar +4 % con stop inicial de 1,5 ATR para todos | 300 | 218.897 | −0,23 % | −0,07 % | −0,17 % (t −2,7) | 41 / 42 % |
| plan clásico sin límites (stops del 10–40 %) | 300 | 218.458 | −0,24 % | −0,09 % | −0,15 % (t −1,9) | 39 / 40 % |
| fijo 1,5 / 3 ATR | 300 | 219.795 | −0,24 % | −0,13 % | −0,12 % (t −2,1) | 34 / 35 % |
| stop que persigue 3 ATR (sin asegurar) | 300 | 220.333 | −0,31 % | −0,11 % | −0,20 % (t −1,4) | 36 / 37 % |
| stop que persigue 5 ATR (sin asegurar) | 300 | 220.333 | −0,33 % | +0,01 % | −0,34 % (t −1,8) | 41 / 42 % |

- **La regla en sí no es el problema ni la solución**: entrando al azar con ella también se gana el 44 % de las veces y se asegura el +4 % el 44 % (los patrones: 42 % y 43 %). Lo que da esa regla es una forma concreta de distribución (muchas pequeñas ganancias de ~+4 % y pérdidas de ~−3 a −5 %), no un valor esperado positivo: la media neta es ≈ 0 con o sin patrón, en torno al coste.
- **El patrón no mejora la entrada al azar con la misma regla** (exceso −0,18 % con t −2,8 en las 300, −0,30 % con t −3,8 en las 50). Calibrado sobre paseos aleatorios, donde no puede haber ventaja, el método sesga el exceso en −0,07 %; descontado, quedan −0,11 % (300) y −0,23 % (50): ligeramente peor que azar, no mejor.
- **Por familia (300 más negociadas, asegurar +4 %)**: figuras alcistas +0,34 % (azar +0,57 %, t −1,2), figuras bajistas −0,80 % (azar −0,84 %), velas alcistas +0,49 % (azar +0,50 %), velas bajistas −0,60 % (azar −0,65 %), directrices alcistas +0,63 % (azar +0,47 %, t +0,7), rebotes bajistas −0,42 % (azar −0,59 %, t +0,8). Ninguna familia supera al azar de forma distinguible del ruido. La excepción es negativa: las **rupturas alcistas** (compra a máximos) rinden **−0,53 % menos** que entrar sin patrón (t −3,7 en las 300 y −3,3 en las 50, con la misma regla de salida); el control es el mismo valor en cualquier fecha y no separa el momento del mercado (el estudio de evidencia, que compara con el mismo día, ve una diferencia mucho menor a 10 sesiones), así que es un aviso sobre comprar a máximos con esta salida, no una prueba firme.
- Los **«rebotes bajistas»** salen +1,14 % mejores que el azar en las 50 (t +2,4, n = 250) pero +0,17 % (t +0,8) en las 300: con 11 grupos por 7 reglas es lo que sale por azar.
- **Hueco de apertura**: de cada 100 operaciones que aseguran +4 %, unas 2 acaban con pérdida (un hueco contra la posición abre por debajo del stop, p. ej. −11 % de un día a otro tras resultados). Es un riesgo real de la regla y la simulación lo cuenta como pérdida.
- **Los stops enormes del plan clásico eran absurdos para operar, pero no eran la causa**: con un stop de 1–2 ATR el resultado frente al azar es el mismo.

### Stop que persigue (trailing) en lugar de objetivo fijo

La idea de muchos operadores discrecionales es «dejar correr las ganancias»: sin objetivo, con un stop que sube a 3 o 5 ATR del mejor cierre. El stop que persigue **no rescata ninguna familia**: las rupturas alcistas siguen por debajo del azar (−0,85 % y −1,14 %) y lo poco positivo (directrices alcistas +0,62 % y +0,49 %, velas bajistas +0,18 % y +0,52 %, t entre +0,6 y +1,0) no se distingue del ruido. Con un stop de 5 ATR el 46 % de las operaciones de velas alcistas se cierra por tiempo sin saltar el stop y el 50 % termina con ganancia (el azar con la misma regla, también el 50 %): el recuento «objetivo contra stop» no habría visto nada de eso.

## Otros mercados: ¿es que las acciones grandes son demasiado eficientes?

`python -m vp study --universe smallmid|mixed` repite el estudio de evidencia con las mismas reglas en mercados menos eficientes (informes `data/evidence_report_smallmid.txt` y `data/evidence_report_mixed.txt`). **En ninguno supera el umbral del mejor de N ni un solo patrón**:

- **Acciones medianas y pequeñas** (S&P 400 + S&P 600, 989 valores, 2,3 millones de barras, coste del 0,30 %, 100 y 500 más negociadas, umbrales 3,61 y 3,77): lo más negativo son el VCP (−1,1 %, t −3,0, con el techo plano) y la compresión de ATR bajista; ningún patrón, vela, figura o consenso pasa el umbral. Las velas bajistas salen casi todas positivas (+0,49 % de mediana, 73 % de ellas por encima de cero) y las alcistas algo negativas (−0,10 %) incluso frente al mismo día: las dos direcciones por debajo de la media del día no es una ventaja direccional y lo más probable (no comprobado) es un efecto de selección, porque las velas grandes se dan en valores volátiles que rinden menos. Contando cada valor-día una vez, las señales bajistas de solo velas dan +0,20 % (t +1,6) y las alcistas −0,11 % (t −1,7): nada que pase el umbral.
- **Cripto, divisas, materias primas e índices** (49 símbolos, 103 patrones, umbral 3,88): lo mayor es la caída del «piercing» (−3,2 %, t −3,3, 63 señales) y la ruptura del máximo de 55 sesiones (+0,79 %, t +3,0, 9 años de 10), que quedan bajo el umbral. Con 49 series es lo que sale por azar.

## ¿Y los operadores que «lo clavan» con dibujitos y velas? Contexto, confluencia y dónde se acaba el estudio (`vp/context.py`)

Un operador discrecional no compra cada patrón: lo mira *en contexto* (a favor de la tendencia, en un soporte, con volumen, con el mercado a favor, con varias señales a la vez). `python -m vp context` prueba exactamente eso con 12 filtros fijados de antemano sobre 219.000 señales (las 300 más negociadas, 2017–2026), por familia y sentido, con las mismas reglas de antes: desarrollo (≤ 2022) y final separados, t con errores estándar en bloques de 10 sesiones y **umbral del mejor de 156 combinaciones** (|t| ≥ 4,1, con 300 repeticiones). Informe completo en `data/context_report.txt`. Sobre 10 sesiones, en exceso sobre la media de todas las acciones ese día:

- **Ninguna combinación supera el umbral** (la mayor es |t| = 3,4: rebotes bajistas con volatilidad baja). Lo que más se acerca es siempre lo mismo: con **volatilidad baja** las señales alcistas salen peor que la media (−0,18 %, final −0,43 %) y las bajistas mejor (+0,25 %, final +0,40 %, t +2,6), es decir, las acciones tranquilas rinden menos que las movidas en este periodo alcista. Lo más probable es que sea una característica del valor (el factor de baja volatilidad) y no una ventaja del patrón: que funcione igual de bien en un sentido y en el contrario lo delata.
- **Una fuga de información que casi me engaña.** La primera versión definía la confluencia («al menos dos señales más del mismo sentido en ±2 sesiones») mirando también las **dos sesiones siguientes**, es decir, premiaba señales que ya habían subido. Con esa fuga la confluencia daba t = +3,9 en figuras alcistas, +5,3 en directrices bajistas y +5,0 en rebotes bajistas, y el meta-etiquetado un IC de 0,056 con el 10 % mejor en +0,78 % (t 4,6). Con la confluencia causal (la sesión y las dos anteriores): +0,14 % en figuras alcistas (t +1,3), +0,20 % en directrices bajistas (t +1,0), todas las alcistas −0,03 % y todas las bajistas −0,05 % (t −0,4 y −0,5). Hay una prueba que falla si la confluencia vuelve a mirar hacia delante (`test_confluence_causal`) y otra que comprueba que las variables del aprendizaje automático son causales.
- **«Todo a favor»** (al menos 4 de: tendencia, volumen, nivel, líder, mercado, confluencia): +0,06 % las alcistas y +0,09 % las bajistas (t +0,6 y +0,3): lo que los discrecionales llaman «setup perfecto» no se distingue del ruido.
- **Meta-etiquetado** (un gradient boosting que decide cuándo fiarse de cada señal; entrenado hacia delante, configuración elegida con 2020–2022 y juzgada en 2023–2026): IC +0,020 (t +1,9), 10 % mejor +0,31 % a 10 sesiones (+0,16 % neto, t +2,2), 10 % peor −0,41 %. **El mismo modelo sin saber qué patrón es** (solo contexto) da IC +0,019 y 10 % mejor +0,11 % (t +0,7): el patrón añade ≈ 0,001 de IC y los años de desarrollo no muestran nada (IC +0,011). Coincide con el estudio de aprendizaje automático: lo poco que hay viene del contexto del valor, no del dibujo.

**Qué dice todo esto sobre quien «nunca falla»**: no hay forma, con datos de 10 años de grandes acciones, de distinguir un patrón (solo o con contexto, con stop fijo o que persigue, en acciones grandes, medianas, cripto o divisas) de entrar al azar con el mismo riesgo. Este estudio no mide a personas, así que lo que sigue son hipótesis, no resultados; pero hay tres explicaciones razonables de que alguien parezca infalible que no requieren que el patrón funcione: (1) **sesgo de recuerdo**: se ven los aciertos dibujados a posteriori y las operaciones perdedoras se olvidan o se reinterpretan («la figura no era buena»); (2) **tasa de acierto alta con pérdidas grandes ocasionales** (objetivo cercano y stop lejano acierta el 70 %, y pierde en cuanto llega una mala racha), algo que la tasa de acierto sola no distingue; (3) **selección**: del grupo que operaba así, los que sobreviven y cuentan su historia son los que tuvieron suerte. Este laboratorio mide lo único que se puede medir: lo que pasó de media tras cada patrón.

### Qué pasa tras cada patrón (dibujado)

En la **Guía** y en el panel de cada gráfico se dibuja la **trayectoria media** de las 20 sesiones siguientes a cada patrón (con el signo de su sentido, menos la media de las acciones esos años) con su banda del 95 %. Si la línea y la banda se quedan alrededor de cero, el patrón no cambia la dirección esperada. Salen de `evidence.json` (`paths`) y se recalculan cada domingo.

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
  candles.py          61 formaciones de vela y 4 de consenso
  breakouts.py        13 rupturas y compresiones
  shapes.py           motor geométrico de figuras (swings, rectas por toques, calidad, formación)
  engine.py           junta todo en eventos causales
  plan.py             entrada, stop, objetivo, T1, obstáculos, resultado simulado
  lab.py              analiza un valor → JSON para la web
  study.py            estudio de evidencia
  planstudy.py        estudio del plan completo (stop y objetivo) frente a entradas al azar
  ml.py               aprendizaje automático fuera de muestra (contexto, velas, figuras)
  context.py          filtros de contexto, confluencia y meta-etiquetado
  data.py, universe.py   descarga de Yahoo y universos (watchlist.txt)
data/                 charts/<VALOR>.json, index.json, catalog.json, evidence.json, informes (evidence_report*, plan_report, ml_report, context_report)
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
python -m vp context                          # filtros de contexto, confluencia y meta-etiquetado
python -m vp study --universe smallmid        # otros mercados: smallmid (S&P 400 + 600) o mixed (cripto, divisas, materias primas, índices)
python -m http.server                         # abrir http://localhost:8000
```

Los precios vienen de Yahoo Finance (split aplicado, sin ajustar por dividendos para que los niveles coincidan con los del bróker); se descarta la barra de la sesión aún abierta.
