# VP Scanner · laboratorio de chartismo y reglas

El laboratorio tiene **dos partes separadas**, cada una con su sección en la web, sus datos y **sus propias estadísticas** (no se mezclan):

| | **Parte 1 · Figuras, velas y rupturas** | **Parte 2 · Regla de impulso con caída** |
|---|---|---|
| Qué es | Detecta **110 patrones chartistas**, los **dibuja** con su plan (entrada, stop inicial, escalera +4 % → +10 %) y **mide si funcionan** | Una **regla sin figuras ni velas**: comprar acciones de impulso alto a 12 meses que han caído, con la escalera del operador, un modelo de fiabilidad, disciplina de cartera y track record hacia delante |
| Web | «Figuras y velas»: Radar · Gráfico · Evidencia · Guía | «Regla de impulso»: Hoy (cribador y plan de hoy) · Seguimiento (track record) · Evidencia · Guía |
| Veredicto | Ningún patrón bate a entrar al azar con el mismo riesgo | Candidata con una ventaja **pequeña** que se repite en acciones distintas |
| Estadísticas | `data/evidence*.json`, `plan_report`, `ml_report`, `context_report` | `data/search*.json`, `search_confirm*.json`, `fiab.json`, `cartera.json`, `impulso.json`, `impulso_track.json` |
| Comandos | `lab`, `study`, `plan`, `ml`, `context` | `search`, `search --confirm`, `fiab`, `screen` |

> No es una recomendación de inversión. Dibujar bien una figura no demuestra que funcione, y una ventaja estadística pequeña no es una promesa de ganar: las dos partes dicen, con sus cifras, qué ha aportado de verdad cada cosa.

---

## PARTE 1 · Figuras, velas y rupturas (chartismo)

Detecta **figuras, velas y rupturas** en acciones grandes de EE.UU., las **dibuja** sobre el gráfico con su plan y **mide si funcionan**.

- **Radar**: qué figuras se están **formando** (con el nivel que las activa y a cuántos ATR está el precio), cuáles son **nuevas hoy** y cuáles se completaron hace poco; filtros «cerca del disparo», sentido, calidad y ★ favoritos.
- **Gráfico**: velas con las figuras dibujadas (rectas con sus toques, ápice, medida), plan por patrón (la escalera del operador: stop inicial razonable, asegurar +4 % y dejar correr, asegurar +10 % si llega; el clásico como referencia), resultado simulado y evidencia del estudio.
- **Evidencia**: ¿baten los patrones a entrar al azar? Estudio con corrección por probar muchos patrones.
- **Guía**: qué es cada patrón y cómo se calcula su plan.

**Veredicto de la parte 1** (detalle en las secciones siguientes): con 10 años de grandes acciones, ningún patrón, vela, figura, ruptura o consenso supera el umbral del mejor de N; aplicando tu regla de salida (asegurar +4 % y dejar correr) a **todas** las señales, el patrón no mejora a entrar al azar con la misma regla (−0,18 % por operación, t −2,8); el contexto, la confluencia y el aprendizaje automático tampoco los distinguen del azar. Las figuras se mantienen en la web, dibujadas con rigor, como forma de **leer** el gráfico; lo que se mide aquí es lo que pasó de media tras cada patrón.

### Cómo se reconocen las figuras (`vp/shapes.py`, causal)

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

### El plan de cada patrón (`vp/plan.py`)

El plan que se dibuja, se simula y se cuenta es **la escalera del operador** (la regla de quien opera chartismo con ganancias modestas: «con un 4 % soy feliz; si sigue subiendo lo dejo correr hasta +10 %; si pasa de ahí me lo aseguro, y si no, al menos el 4 %»). Las reglas están fijadas de antemano, no optimizadas con los resultados:

- **Entrada**: apertura de la sesión siguiente a la que completa el patrón.
- **Stop inicial**: el estructural (0,25 ATR al otro lado de la figura) si queda entre **1 y 2 ATR** de la entrada; si está más lejos se **acorta a 2 ATR** (en una acción de volatilidad media, ~5 %); nunca a menos de 1 ATR. El plan clásico daba stops del 10, 26 o 43 %.
- **Escalera**: cuando una sesión **cierra a +4 % o más** a favor, el stop sube a **+4 %** y **se queda ahí** (no persigue: una corrección normal no saca de la operación antes del segundo nivel). Cuando una sesión **cierra a +10 % o más**, el stop sube a **+10 %** y desde ahí **persigue al mejor cierre a 2 ATR**, sin bajar nunca de lo asegurado. No hay objetivo fijo: se asegura al menos +4 %, y +10 % si se llega. Máximo 60 sesiones; el stop gana si se tocan ambos el mismo día y un hueco de apertura sale a la apertura. (`plan.outcome_ladder`; el plan de un solo nivel, +4 % persiguiendo a 3 ATR, se conserva en `plan.outcome_lock` para el estudio del plan.)
- **Ninguna señal se descarta**. Antes había un estado «no operable» (recompensa menor de 1,5 veces el riesgo, riesgo de más del 8 % del precio, medida consumida) que quitaba señales de la cuenta; era un error de planteamiento: un umbral de calidad estadística sirve para **graduar** cuánta evidencia hay, no para tirar operaciones. Ahora el riesgo alto (> 8 % del precio), un stop más lejos que el +4 % y la medida ya recorrida son **avisos informativos** en la ficha («la señal sigue contando»).
- **Tamaño**: la ficha dice cuánta posición corresponde a arriesgar el 1 % de la cuenta con ese stop (1 / riesgo %): lo que importa para el riesgo no es la distancia del stop sino el tamaño de la posición.
- El **plan clásico sin límites** (stop al otro lado de la figura y objetivo por la medida clásica, T1 al 50 %) se conserva como referencia y se ve con el interruptor «Plan clásico» del gráfico.
- **Obstáculos**: soportes y resistencias fuertes en el camino.
- **Quién gana y quién pierde**: gana la operación que **termina con ganancia neta** (después del 0,15 % de costes del estudio), llegue o no a asegurar. Asegurar +4 % o +10 %, saltar el stop inicial o cerrar por tiempo es solo cómo terminó. De cada operación se guardan la ganancia neta, el **múltiplo del riesgo (R)**, el recorrido máximo a favor y en contra y el nivel asegurado. La Lectura de cada valor da el porcentaje de ganadoras, las veces que se aseguró, la **ganancia media de las ganadoras y la pérdida media de las perdedoras**.
- **Una operación por valor** (`group_trades` en `vp/lab.py`): cuando varias señales saltan a la vez (velas y figuras, o varias velas) **no se cuenta cada una**. La primera señal (si coinciden varias en la misma barra, la de mayor rango: figura > ruptura > directriz > rebote > vela, y a igualdad la de mayor calidad) abre la operación y su plan es el que se simula; las del mismo sentido mientras está abierta son **confirmaciones** y las del sentido contrario son **contrarias** (aviso de conflicto, tampoco son operación propia). No se descarta ninguna señal: las confirmaciones y contrarias se muestran, pero no abren una segunda operación sobre el mismo valor.

### Estudio de evidencia (`vp/study.py`)

- Universo: miembros actuales del S&P 500 y el Nasdaq-100; se usan las **50 y las 300 más negociadas en cada fecha** (dinero medio de 60 sesiones).
- Cada señal se opera en la apertura siguiente y se cierra a 5, **10** (principal) y 20 sesiones, con un coste del 0,15 % de ida y vuelta; los patrones bajistas se miden en corto. «% ganadoras» es la parte de operaciones con ganancia neta, no la de las que llegan a ningún objetivo.
- «Exceso» = rendimiento − media de entrar al azar **ese mismo día** en el mismo universo y sentido: lo que haga el mercado en bloque (un patrón que salta justo antes de un rebote general) no cuenta como ventaja. Se guarda también el exceso frente a la media del **año**, que era la referencia de las primeras versiones (columna «vs año»).
- **t con errores estándar agrupados en bloques de 10 sesiones** (antes, por día): los rendimientos a 10 sesiones de días contiguos se solapan y las señales vienen en rachas. En los datos reales el cambio recorta la t casi a la mitad (ver abajo).
- **Umbral del mejor de N**: se desplazan al azar las señales de cada valor y se anota el mayor |t| entre todos los patrones; solo «supera el azar» quien pasa el percentil 95 de ese máximo. Se estudian también las señales de **calidad ≥ 75** de cada figura.
- **Coincidencias**: cada patrón se mide con todas sus señales, también las que saltan a la vez que otro patrón; por eso hay una tabla aparte donde **cada valor-día cuenta una sola vez** (solo velas, solo estructura, ambas, contradictorias).
- **Limitación**: los miembros son los actuales (sesgo de supervivencia que favorece a los largos).

### Resultados (estudio del 27-oct-2026: 500 acciones, 1,23 millones de barras, 2016–2026; se recalcula cada domingo)

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

#### ¿Y si el método fuera demasiado exigente? Potencia, controles y el conjunto de cada familia

«Ningún patrón supera el umbral» puede significar que no funcionan o que el estudio no tiene potencia para verlo. Se comprobó así (informe `data/evidence_report.txt`, 300 más negociadas):

- **El umbral es exigente a propósito** (|t| ≥ 3,6: el del mejor de 130 patrones). Con el error estándar real de cada patrón (mediana 0,28 % por operación), el estudio detecta con un 80 % de probabilidad efectos de **≥ 1,25 %** por operación en un patrón típico y de **≥ 8 %** en los 10 más raros. **No puede ver ventajas pequeñas en patrones concretos**, y la web muestra el intervalo del 95 % de cada patrón además de la t.
- **Control positivo con el mismo motor** (referencia del mismo día, bloques de 10 sesiones, mismo umbral): el 10 % de valores con mejor rendimiento a 120 sesiones (momentum) rinde **+0,54 % ± 0,32 (t +3,3, 71.580 señales)**: **tampoco supera el umbral** (con las 50 más negociadas, +1,01 % ± 0,56, t +3,5, queda justo en el umbral de 3,5). La volatilidad alta da +0,65 % (t +3,0), el momentum bajo +0,13 % y la sobreventa a corto +0,13 % (t +1,3). Un efecto real de ~0,5 % por operación queda en el límite: **«no supera el umbral» no significa «no funciona», significa «no se distingue del azar con estos datos»**.
- **Lo que sí se puede afirmar con precisión es el conjunto de cada familia** (todas sus señales juntas, cada valor-día una vez; frente a un valor cualquiera ese día; intervalos del 95 %): figuras alcistas +0,01 % ± 0,14, bajistas −0,04 % ± 0,17; rupturas alcistas −0,06 % ± 0,13, bajistas −0,15 % ± 0,20; velas alcistas −0,03 % ± 0,14, bajistas −0,02 % ± 0,11; directrices alcistas +0,08 % ± 0,22, bajistas −0,11 % ± 0,18; rebotes bajistas +0,35 % ± 0,28 (t +2,4: no pasa el umbral de Bonferroni de 2,8). **Ninguna familia entera mueve el resultado más de unas décimas de punto por operación, y el coste de ida y vuelta (0,15 %) es del mismo tamaño**: si existe una ventaja media, es demasiado pequeña para sobrevivir a los costes.
- **Lo que este estudio no puede probar**: que un operador con criterio añadido (qué patrones elige, contexto que no se puede escribir como regla, gestión de la posición) tenga ventaja. Mide reglas mecánicas.

#### ¿Se fía uno de estas estadísticas? Comprobaciones del propio método

- Con un efecto plantado de +3 % en señales de valores distintos el estudio lo detecta (t = 34); con 24 paseos aleatorios no sale ningún patrón «significativo» de 66 (pruebas automáticas).
- La referencia del mismo día cancela lo que el mercado hace en bloque: si un efecto plantado saltara en todos los valores a la vez, el estudio lo trataría como mercado y no como patrón (por eso la prueba coloca el efecto en días distintos). Con simulaciones de paseos aleatorios y señales al azar (densidad del 2 %, 8 % y 30 %) tanto el t por día como el de bloques salen bien calibrados (|t| > 1,96 en 4,5–6 % de los casos); la diferencia aparece con datos reales, probablemente porque las señales vienen en rachas y el mercado tiene regímenes.
- Control positivo con datos reales: en las mismas fechas, el factor clásico de impulso (rendimiento a 120 sesiones) deja al 10 % mejor +0,71 % por encima del resto a 10 sesiones (t +3,2): el método detecta efectos pequeños reales cuando los hay.
- El umbral del mejor de N corrige que, entre 130 patrones, alguno saldrá «significativo» por casualidad. Los porcentajes de ganadoras de casi todos los patrones rondan el 50 % porque las acciones grandes y líquidas son muy eficientes: es lo que cabe esperar de datos reales, no un fallo del cálculo.
- Errores encontrados y corregidos en esta revisión: una «confluencia» que miraba también las dos sesiones siguientes (fuga de información); el recuento de ganancias por «llegar al objetivo» en lugar de por ganancia neta; señales solapadas contadas varias veces; la t por día y la referencia del año; «65 formaciones de vela» donde son 61 más 4 de consenso; descripciones de Bollinger y ATR que no decían lo que hace el código; un máximo por debajo del cierre en datos de Yahoo; y un parámetro `--reps` del contexto que se ignoraba.

### Aprendizaje automático (`vp/ml.py`): ¿combinar todo mejora?

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

### Tu regla de salida frente al azar: ¿asegurar +4 % y dejar correr cambia algo? (`vp/planstudy.py`)

La regla de muchos operadores («con un 4–5 % soy feliz; si sube más lo dejo correr, pero pasado el 4 % pongo un stop para asegurarlo») es razonable y es la que usa la web (ahora como escalera: +4 % plano y +10 % persiguiendo a 2 ATR). **Con la escalera** (`esc`, estudio del 27-oct-2026, 300 más negociadas, 218.700 señales): media neta −0,23 % frente a −0,06 % de entrar al azar con la misma escalera, **exceso −0,17 % (t −3,3)**; ganan el 42 % (azar 44 %); figuras alcistas −0,22 % (t −1,3), velas alcistas −0,05 %, velas bajistas +0,07 %, rupturas alcistas −0,45 % (t −3,3). El resultado de la parte 1 no cambia con la escalera: el patrón no mejora a entrar al azar con la misma regla de salida. Falta saber si, **aplicada a todas las señales sin descartar ninguna**, el patrón aporta algo frente a entrar sin patrón con esa misma regla. `python -m vp plan` simula cada señal como la web (entrada en la apertura siguiente, stop inicial acotado a 1–2 ATR, al cerrar a +4 % el stop sube a +4 % y persigue a 3 ATR, hasta 60 sesiones, el stop gana si se tocan el mismo día, coste del 0,15 %) y la compara con **8 entradas al azar en el mismo valor y sentido, con las mismas distancias de stop en ATR** (y volatilidad relativa parecida) y **exactamente la misma regla de salida**.

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

#### Stop que persigue (trailing) en lugar de objetivo fijo

La idea de muchos operadores discrecionales es «dejar correr las ganancias»: sin objetivo, con un stop que sube a 3 o 5 ATR del mejor cierre. El stop que persigue **no rescata ninguna familia**: las rupturas alcistas siguen por debajo del azar (−0,85 % y −1,14 %) y lo poco positivo (directrices alcistas +0,62 % y +0,49 %, velas bajistas +0,18 % y +0,52 %, t entre +0,6 y +1,0) no se distingue del ruido. Con un stop de 5 ATR el 46 % de las operaciones de velas alcistas se cierra por tiempo sin saltar el stop y el 50 % termina con ganancia (el azar con la misma regla, también el 50 %): el recuento «objetivo contra stop» no habría visto nada de eso.

### Otros mercados: ¿es que las acciones grandes son demasiado eficientes?

`python -m vp study --universe smallmid|mixed` repite el estudio de evidencia con las mismas reglas en mercados menos eficientes (informes `data/evidence_report_smallmid.txt` y `data/evidence_report_mixed.txt`). **En ninguno supera el umbral del mejor de N ni un solo patrón**:

- **Acciones medianas y pequeñas** (S&P 400 + S&P 600, 989 valores, 2,3 millones de barras, coste del 0,30 %, 100 y 500 más negociadas, umbrales 3,61 y 3,77): lo más negativo son el VCP (−1,1 %, t −3,0, con el techo plano) y la compresión de ATR bajista; ningún patrón, vela, figura o consenso pasa el umbral. Las velas bajistas salen casi todas positivas (+0,49 % de mediana, 73 % de ellas por encima de cero) y las alcistas algo negativas (−0,10 %) incluso frente al mismo día: las dos direcciones por debajo de la media del día no es una ventaja direccional y lo más probable (no comprobado) es un efecto de selección, porque las velas grandes se dan en valores volátiles que rinden menos. Contando cada valor-día una vez, las señales bajistas de solo velas dan +0,20 % (t +1,6) y las alcistas −0,11 % (t −1,7): nada que pase el umbral.
- **Cripto, divisas, materias primas e índices** (49 símbolos, 103 patrones, umbral 3,88): lo mayor es la caída del «piercing» (−3,2 %, t −3,3, 63 señales) y la ruptura del máximo de 55 sesiones (+0,79 %, t +3,0, 9 años de 10), que quedan bajo el umbral. Con 49 series es lo que sale por azar.

### ¿Y los operadores que «lo clavan» con dibujitos y velas? Contexto, confluencia y dónde se acaba el estudio (`vp/context.py`)

Un operador discrecional no compra cada patrón: lo mira *en contexto* (a favor de la tendencia, en un soporte, con volumen, con el mercado a favor, con varias señales a la vez). `python -m vp context` prueba exactamente eso con 12 filtros fijados de antemano sobre 219.000 señales (las 300 más negociadas, 2017–2026), por familia y sentido, con las mismas reglas de antes: desarrollo (≤ 2022) y final separados, t con errores estándar en bloques de 10 sesiones y **umbral del mejor de 156 combinaciones** (|t| ≥ 4,1, con 300 repeticiones). Informe completo en `data/context_report.txt`. Sobre 10 sesiones, en exceso sobre la media de todas las acciones ese día:

- **Ninguna combinación supera el umbral** (la mayor es |t| = 3,4: rebotes bajistas con volatilidad baja). Lo que más se acerca es siempre lo mismo: con **volatilidad baja** las señales alcistas salen peor que la media (−0,18 %, final −0,43 %) y las bajistas mejor (+0,25 %, final +0,40 %, t +2,6), es decir, las acciones tranquilas rinden menos que las movidas en este periodo alcista. Lo más probable es que sea una característica del valor (el factor de baja volatilidad) y no una ventaja del patrón: que funcione igual de bien en un sentido y en el contrario lo delata.
- **Una fuga de información que casi me engaña.** La primera versión definía la confluencia («al menos dos señales más del mismo sentido en ±2 sesiones») mirando también las **dos sesiones siguientes**, es decir, premiaba señales que ya habían subido. Con esa fuga la confluencia daba t = +3,9 en figuras alcistas, +5,3 en directrices bajistas y +5,0 en rebotes bajistas, y el meta-etiquetado un IC de 0,056 con el 10 % mejor en +0,78 % (t 4,6). Con la confluencia causal (la sesión y las dos anteriores): +0,14 % en figuras alcistas (t +1,3), +0,20 % en directrices bajistas (t +1,0), todas las alcistas −0,03 % y todas las bajistas −0,05 % (t −0,4 y −0,5). Hay una prueba que falla si la confluencia vuelve a mirar hacia delante (`test_confluence_causal`) y otra que comprueba que las variables del aprendizaje automático son causales.
- **«Todo a favor»** (al menos 4 de: tendencia, volumen, nivel, líder, mercado, confluencia): +0,06 % las alcistas y +0,09 % las bajistas (t +0,6 y +0,3): lo que los discrecionales llaman «setup perfecto» no se distingue del ruido.
- **Meta-etiquetado** (un gradient boosting que decide cuándo fiarse de cada señal; entrenado hacia delante, configuración elegida con 2020–2022 y juzgada en 2023–2026): IC +0,020 (t +1,9), 10 % mejor +0,31 % a 10 sesiones (+0,16 % neto, t +2,2), 10 % peor −0,41 %. **El mismo modelo sin saber qué patrón es** (solo contexto) da IC +0,019 y 10 % mejor +0,11 % (t +0,7): el patrón añade ≈ 0,001 de IC y los años de desarrollo no muestran nada (IC +0,011). Coincide con el estudio de aprendizaje automático: lo poco que hay viene del contexto del valor, no del dibujo.

**Qué dice todo esto sobre quien «nunca falla»**: no hay forma, con datos de 10 años de grandes acciones, de distinguir un patrón (solo o con contexto, con stop fijo o que persigue, en acciones grandes, medianas, cripto o divisas) de entrar al azar con el mismo riesgo. Este estudio no mide a personas, así que lo que sigue son hipótesis, no resultados; pero hay tres explicaciones razonables de que alguien parezca infalible que no requieren que el patrón funcione: (1) **sesgo de recuerdo**: se ven los aciertos dibujados a posteriori y las operaciones perdedoras se olvidan o se reinterpretan («la figura no era buena»); (2) **tasa de acierto alta con pérdidas grandes ocasionales** (objetivo cercano y stop lejano acierta el 70 %, y pierde en cuanto llega una mala racha), algo que la tasa de acierto sola no distingue; (3) **selección**: del grupo que operaba así, los que sobreviven y cuentan su historia son los que tuvieron suerte. Este laboratorio mide lo único que se puede medir: lo que pasó de media tras cada patrón.

#### Qué pasa tras cada patrón (dibujado)

En la **Guía** y en el panel de cada gráfico se dibuja la **trayectoria media** de las 20 sesiones siguientes a cada patrón (con el signo de su sentido, menos la media de las acciones esos años) con su banda del 95 %. Si la línea y la banda se quedan alrededor de cero, el patrón no cambia la dirección esperada. Salen de `evidence.json` (`paths`) y se recalculan cada domingo.


---

## PARTE 2 · Regla de impulso con caída (sin figuras ni velas)

Estadísticas **separadas** de las de la parte 1. Resumen: probando miles de reglas con selección hacia delante y control del procedimiento, y confirmando cinco candidatas fijadas de antemano en acciones distintas (S&P 400 y 600), la idea que se repite es **comprar acciones de impulso alto a 12 meses que han caído** (por debajo de su media de 200 sesiones, con RSI(2) ≤ 10 o con 2 o más señales bajistas a la vez), en largo y con las salidas del operador. Exceso estimado de **+0,3 a +0,5 % por operación** sobre el azar tras descontar el sesgo de supervivencia, **concentrado en el 40 % más volátil** (en las acciones tranquilas la ventaja es cero o negativa). Encima de la regla van **la escalera del operador** (+4 % plano → +10 % persiguiendo), la **búsqueda por fiabilidad**, un **modelo de fiabilidad** calibrado y validado hacia delante, la **disciplina del plan de hoy**, el **seguimiento con precios reales** y la **cartera simulada** (lo que da esa disciplina en una cuenta durante toda la historia, frente a carteras al azar y al mercado) con un **menú de salidas** sobre la misma entrada. En la web: «Regla de impulso» → Hoy · Seguimiento · Evidencia · Guía.

**Nombres.** Lo que se opera son **tres entradas** (todas en largo, todas con impulso 12-1 meses en el 20 % mejor del día) y **una salida**: «**bajo 200**» (precio bajo su media de 200), «**2+ bajistas**» (2 o más señales bajistas a la vez) y «**RSI(2) ≤ 10**», con la escalera. Los códigos **C1…C8** de las tablas de confirmación son la misma entrada medida con distintas salidas (C1, C2, C3, C6, C7 = «bajo 200» con asegura +4 %, asegura +6 %, cierre a 10 sesiones, escalera, escalera con stop al 6 %; C4 y C8 = «2+ bajistas» con cierre a 10 sesiones y escalera; C5 = «RSI(2) ≤ 10» con cierre a 10 sesiones) y **V1…V6** la de «bajo 200» separada por volatilidad baja o alta. La web y los informes enseñan el nombre (`search.LABELS`); el código queda en los ficheros de datos.

### ¿Existe alguna regla que gane de verdad al azar? Búsqueda con selección hacia delante (`vp/search.py`)

La pregunta de «maximiza la rentabilidad por encima del azar» tiene una trampa: si se prueban miles de reglas, la mejor siempre parece buena. `python -m vp search` busca igualmente, pero de forma que el resultado sea creíble:

- **Panel**: cada fila es una acción y una fecha (las 300 más negociadas, 712.000 filas, 2017–2026). Para cada fila, en largo y en corto, se simula la entrada en la apertura siguiente con 6 salidas fijadas de antemano: 4 del operador (stop inicial de 1,5 o 2,5 ATR; al cerrar a +4 % o +6 % el stop sube a ese nivel y persigue al mejor cierre a 3 o 4 ATR), un stop que persigue a 5 ATR sin asegurar y el cierre a 10 sesiones. Idéntica a la simulación de la web (se comprueba en las pruebas). **Exceso** = neto de costes menos la media de todas las filas de ese día.
- **Biblioteca de reglas fijada antes de mirar nada**: 32 condiciones (24 de contexto: tendencia, impulso, sobreventa, extremos anuales, volatilidad, volumen, huecos, mercado; 8 de patrón: velas, figuras y rupturas alcistas o bajistas y su confluencia), solas o de dos en dos: 400 combinaciones × 2 sentidos × 6 salidas = **4.800 reglas**.
- **Selección hacia delante**: para cada año de prueba se elige la mejor regla con lo anterior a ese año (menos 60 sesiones, para que no se cuele ningún resultado sin cerrar) y se mide en el año sin tocarla. La cifra «fuera de muestra» es la del procedimiento completo de buscar y elegir. Solo son elegibles reglas con ≥ 2.000 operaciones y operaciones en al menos el 60 % de los periodos del entrenamiento (una versión anterior eligió para 2020 una regla con t = 22 que solo operaba en unos pocos periodos de mercado bajista, y otra que ni siquiera operó en el año de prueba; ya no es posible).
- **Control del procedimiento**: se repite TODO con los resultados desplazados en el calendario (al menos 300 sesiones, más que la memoria de las variables de 12 meses). Lo que «encuentra» así es lo que da el azar al buscar entre miles de reglas. Un primer control con desplazamientos desde 120 sesiones salía demasiado exigente (el 18 % de los desplazamientos caía dentro de la memoria de la condición y la igualaba por construcción: p idéntico de 0,067 en las cinco reglas); se corrigió y se explica aquí porque cambia los p-valores.
- **Errores agrupados** en bloques de 10 sesiones (cierre a 10) o de 60 (salidas de hasta 60 sesiones: sus resultados se solapan hasta 60). Con bloques de 10 y salidas largas, el control daba «descubrimientos» enormes en datos sin ningún efecto.
- **Sesgo de supervivencia parcialmente corregido**: el universo son los miembros ACTUALES del S&P 500 y el Nasdaq-100, y las acciones que cayeron y salieron del índice no están (Yahoo no las tiene). Cada valor del S&P 500 solo cuenta desde su **fecha de incorporación** al índice (columna «Date added» de Wikipedia): se quitan 153.000 barras-valor (21 %). Los que salieron siguen sin estar: el sesgo restante favorece a los largos.

#### Resultado de la búsqueda (fuera de muestra, 2020–2026; 27-oct-2026, con las 10 salidas)

Con las cuatro salidas de la escalera la biblioteca pasa de 4.800 a **8.000 reglas** (400 combinaciones × 2 sentidos × 10 salidas). Dos criterios de selección, cada uno con su control:

| criterio · conjunto | operaciones | exceso (t) | fiabilidad (puntos sobre el día, t) | ganan (todas las filas) | +4 % / +10 % | control: p exceso · p fiabilidad |
|---|---|---|---|---|---|---|
| **rentabilidad** · todas | 16.123 | +0,24 % (+1,7) | +1,4 (+1,6) | 60,4 % (58,3 %) | 50 / 12 % | 0,45 · 0,25 |
| rentabilidad · solo contexto | 18.637 | +0,36 % (+2,5) | +1,9 (+2,2) | 59,5 % (57,3 %) | 50 / 12 % | 0,125 · 0,10 |
| rentabilidad · con patrón | 11.237 | +0,36 % (+1,9) | +1,6 (+1,3) | 63,3 % (57,5 %) | 43 / 12 % | 0,15 · 0,275 |
| **fiabilidad** · todas | 26.598 | +0,11 % (+0,6) | **+8,3 (+6,5)** | 48,3 % (41,8 %) | 49 / 7 % | 0,275 · **< 0,025** |
| fiabilidad · solo contexto | 28.300 | +0,13 % (+0,7) | +8,4 (+7,0) | 47,8 % (38,3 %) | 48 / 8 % | 0,275 · < 0,025 |
| fiabilidad · con patrón | 22.043 | +0,20 % (+1,4) | +6,6 (+5,6) | 49,0 % (43,0 %) | 50 / 8 % | 0,15 · < 0,025 |

Dos lecciones, las dos incómodas:

- **Con más salidas, la búsqueda automática por rentabilidad deja de ser significativa** (exceso +0,24 %, p 0,45; antes, con 6 salidas, +0,54 % y p 0,05). Al tener más libertad elige en 2021–2023 reglas en **corto** con la escalera y stop al 10 % que pierden fuera de muestra (−1,2 % en 2021). Más libertad = más sobreajuste; el procedimiento completo de «buscar la mejor» no gana al azar de buscar. Lo que se sostiene es una regla concreta (abajo), no el buscador.
- **Elegir por fiabilidad encuentra reglas que ganan más veces que el azar del día (+8 puntos, p < 0,025) pero no ganan dinero** (exceso +0,11 %, p 0,275): son cortos en valores volátiles con salidas de stop estrecho, que aciertan a menudo con ganancias pequeñas y pierden poco a menudo con pérdidas grandes. **La tasa de acierto sola no es un objetivo**: hay que mirar acierto × lo que se gana cuando se acierta × lo que se pierde cuando no. Es exactamente la trampa de «fiabilidad» en la que cae quien solo mira el porcentaje de trades ganados.

**Lo que se repite**: elegidas con los años ≤ 2022 y miradas una sola vez después, 10 de las 15 mejores reglas mantienen exceso positivo (final medio +0,42 %), y lo que aguanta de verdad es siempre la misma idea en largo, **valores de impulso alto a 12 meses (20 % mejor, sin el último mes) que han caído** (bajo su media de 200 o con 2 o más señales bajistas); los cortos que el entrenamiento ponía primeros se quedan en cero en el final:

| regla | salida | entrenamiento: exceso (t) | final: exceso (t) |
|---|---|---|---|
| impulso 12-1 alto Y bajo su media de 200 → largo | stop 2,5 ATR · asegura +6 % | +0,90 % (+4,1) | +1,62 % (+2,5) |
| impulso 12-1 alto Y bajo su media de 200 → largo | stop 2,5 ATR · asegura +4 % | +0,75 % (+4,2) | +1,55 % (+2,3) |
| impulso 12-1 alto Y bajo su media de 200 → largo | **escalera** (2,5 ATR; +4 → +10) | +0,62 % (+3,9) | **+1,27 % (+2,9)** |
| impulso 12-1 alto Y bajo su media de 200 → largo | escalera con stop al 10 % | +0,57 % (+3,3) | +0,85 % (+2,2) |
| impulso 12-1 alto Y 2+ señales bajistas → largo | cierre a 10 sesiones | +0,76 % (+3,5) | +0,76 % (+2,1) |
| bajo su media de 200 Y RSI(14) alto → **corto** | escalera con stop al 10 % | +0,59 % (+4,6) | −0,06 % (−0,4) |
| RSI(2) ≥ 90 Y hueco de apertura ≥ +2 % → corto | escalera con stop al 10 % | +1,00 % (+4,0) | −0,31 % (−0,9) |

Entrar al azar con cada salida (todas las filas, largo): con la escalera y stop a 2,5 ATR gana el **58,6 %** y asegura +4 % el 59 % (+10 % el 12 %); con stop al 10 % del precio, el **69 %**. Ganar 7 de cada 10 es barato con stops anchos en una década alcista: por eso todo se mide frente al azar del mismo día.

#### Confirmación en acciones que no se usaron para encontrar la regla

Las 5 mejores candidatas se fijaron **antes** de mirar otras acciones (`search.CANDIDATAS`: una hipótesis cada una, sin elegir nada). Se evalúan en el **S&P 400 y el S&P 600** (988 acciones distintas, las 300 más negociadas de ese universo, 2017–2026, coste 0,30 %), con el mismo control corregido (300 repeticiones):

| candidata (largo) | salida | exceso (t) | neto (azar mismas fechas) | % ganadoras (todas las filas) | años + | p exceso / p t |
|---|---|---|---|---|---|---|
| C1 impulso 12-1 alto Y bajo su media de 200 | asegura +4 % | **+0,76 %** (+2,4) | +1,66 % (+0,90 %) | 71,3 % (60,6 %) | 9/10 | 0,037 / 0,047 |
| C2 ídem | asegura +6 % | +0,87 % (+2,3) | +1,86 % (+0,99 %) | 65,5 % (53,5 %) | 8/10 | 0,063 / 0,040 |
| C3 ídem | cierre a 10 ses. | +0,63 % (+2,5) | +1,17 % (+0,54 %) | 54,6 % (49,9 %) | 8/10 | 0,007 / 0,023 |
| C4 impulso 12-1 alto Y 2+ patrones bajistas | cierre a 10 ses. | +1,02 % (+3,7) | +1,11 % (+0,08 %) | 53,8 % (49,9 %) | 8/10 | 0,000 / 0,000 |
| C5 impulso 12-1 alto Y RSI(2) ≤ 10 | cierre a 10 ses. | +0,78 % (+3,4) | +0,81 % (+0,03 %) | 52,9 % (49,9 %) | 9/10 | 0,000 / 0,000 |

Las cinco dan exceso positivo en acciones distintas, con 8 o 9 de 10 años positivos. Con 5 hipótesis (muy correlacionadas) y corrección de Bonferroni, **C4 y C5 siguen siendo significativas** (p < 0,017); C3 queda en 0,03–0,1 y C1 y C2 en el límite. En el universo grande, con los miembros desde su incorporación, las mismas reglas dan exceso de +0,68 a +1,10 % (t entre 3,4 y 4,2, p < 1/300) y 72 % de ganadoras con asegurar +4 % frente al 58,5 % de entrar al azar (`data/search_confirm.txt`, `data/search_confirm_smallmid.txt`).

#### ¿Solo en valores muy volátiles? Robustez por volatilidad

La búsqueda no distinguía por volatilidad, y el cribador real enseñó el motivo para hacerlo: las acciones que cumplen hoy la regla son casi todas muy volátiles (semiconductores, IA), con el stop inicial de 2,5 ATR a entre el 9 % y el 25 % del precio. Se midió la misma regla base (impulso 12-1 alto Y bajo su media de 200) según la volatilidad del valor, ATR/precio en el 40 % **más bajo** o en el 40 % **más alto** de ese día. **Es una comprobación descriptiva, hecha después de ver los resultados** (no estaba fijada de antemano); por eso se anota aquí y el seguimiento hacia delante la tratará como hipótesis nueva.

| volatilidad del valor | universo | salida | n | exceso (t) | p exceso / p t | años + | % ganadoras (todas las filas) |
|---|---|---|---|---|---|---|---|
| **alta** (40 % más volátil) | S&P 400 y 600 | asegura +4 % | 12.705 | **+1,21 %** (+2,2) | 0,037 / 0,077 | 9/10 | 71,8 % (60,6 %) |
| **alta** | S&P 400 y 600 | cierre a 10 ses. | 12.718 | **+0,95 %** (+2,9) | 0,003 / 0,017 | 9/10 | 52,5 % (49,9 %) |
| **alta** | S&P 500 + Nasdaq-100 | asegura +4 % | 9.249 | **+1,47 %** (+3,5) | < 0,003 / 0,010 | 9/10 | 72,8 % (58,5 %) |
| **alta** | S&P 500 + Nasdaq-100 | cierre a 10 ses. | 9.274 | **+1,10 %** (+4,3) | < 0,003 / < 0,003 | 9/10 | 58,5 % (53,3 %) |
| **baja** (40 % más tranquilo) | S&P 400 y 600 | asegura +4 % | 3.274 | −0,94 % (−1,6) | 0,96 / 0,87 | 2/6 | 71,9 % (60,6 %) |
| **baja** | S&P 400 y 600 | cierre a 10 ses. | 3.274 | −0,57 % (−1,4) | 0,90 / 0,74 | 1/6 | 62,4 % (49,9 %) |
| **baja** | S&P 500 + Nasdaq-100 | asegura +4 % | 2.385 | −0,11 % (−0,3) | 0,37 / 0,39 | 3/6 | 70,5 % (58,5 %) |
| **baja** | S&P 500 + Nasdaq-100 | cierre a 10 ses. | 2.388 | −0,07 % (−0,2) | 0,41 / 0,41 | 3/6 | 64,8 % (53,3 %) |

**Toda la ventaja está en las acciones muy volátiles; en las tranquilas es cero o negativa.** En las tranquilas hay pocas filas (un valor tranquilo con impulso alto y bajo su media de 200 es raro: 2.400–3.300 operaciones en 6 años), así que no se puede descartar una ventaja pequeña, pero la estimación en acciones distintas es negativa. En términos prácticos: la regla gana al azar donde el stop inicial a 2,5 ATR queda lejos (en valores con ATR de 5–9 %, a más del 12–20 % del precio), de modo que el **tamaño de la posición** es lo que controla el riesgo (el cribador lo da: con riesgo del 1 % de la cuenta, entre el 4 % y el 11 % de la cuenta por valor). Es el efecto conocido del rebote de las caídas en valores de beta alta; no es una propiedad del patrón. Con el sesgo de supervivencia (más grave justo en valores volátiles que cayeron y volvieron a subir) el exceso real es probablemente la mitad.

**Cautelas que no se pueden quitar con estos datos**
- El sesgo de supervivencia no está del todo corregido (los que salieron del índice no están; en el S&P 400 y 600 ni siquiera se quita la incorporación): comprar caídas de valores fuertes es justo lo que más se infla con él. Es razonable descontar la mitad: unos **+0,3 a +0,5 % por operación** sobre el azar, no +1 %.
- **No es chartismo**: la parte que aporta es el impulso y la caída previa. Con patrón o sin él da lo mismo (C3 sin patrón +0,63 % y C4 con patrón +1,02 % en el S&P 400/600; +0,74 % y +0,80 % en el universo grande): el patrón es un marcador de «ha caído», no una señal propia.
- Un exceso de +0,5 a +1 % por operación es pequeño: con costes realistas (comisión, deslizamiento en valores menos líquidos, impuestos) y operaciones simultáneas sobre los mismos valores (las filas de una regla vienen en rachas y comparten mercado), la ganancia real será menor y variará mucho de un año a otro (años con −1 % existen).
- Son rendimientos medios sobre muchas operaciones: una operación suelta pierde casi un tercio de las veces incluso con la mejor salida (C1: 71 % ganadoras).


### La escalera del operador y la fiabilidad

- **Salidas de la búsqueda** (`search.EXITS`, fijadas de antemano): a las seis anteriores se añaden cuatro con la **escalera** (`esc`: stop inicial 2,5 ATR; `esca`: 1,5 ATR; `esc6`: stop al 6 % del precio; `esc10`: al 10 %): al cerrar a +4 % el stop sube a +4 % y se queda; al cerrar a +10 % sube a +10 % y persigue a 2 ATR; 60 sesiones. La simulación vectorizada es idéntica a la de la web (`plan.outcome_ladder`) y se comprueba en las pruebas, incluidos los códigos de salida (stop inicial / aseguró +4 / aseguró +10 / tiempo).
- **Búsqueda por fiabilidad** (`search.walk_forward(mode='fiab')`): además de elegir cada año la regla de mayor t de exceso, se elige la de mayor **t de ganancia relativa** («gana» menos la fracción de filas del día que gana con esa salida), exigiendo exceso positivo. Es la pregunta del operador: cuántas veces sale bien, frente al azar del mismo día. Tiene su propio control del procedimiento.
- **Candidatas con la escalera** (`C6`, `C7`, `C8`) y robustez por volatilidad (`V5`, `V6`), que se confirman en el S&P 400 y 600 como las anteriores; la confirmación da ahora también fiabilidad (puntos y p), % que aseguró +4 % y +10 %, % de stops iniciales, ganancia y pérdida medias y factor de beneficio.
- **Modelo de fiabilidad** (`vp/fiab.py`, `python -m vp fiab`): un gradient boosting con parámetros fijados de antemano predice la probabilidad de que la escalera acabe con ganancia a partir del contexto del valor (percentiles del día), sus patrones recientes y la amplitud del mercado. **Walk-forward por años**: para cada año se entrena solo con operaciones cerradas al menos 95 días antes. Se juzga por AUC fuera de muestra, por **calibración** (si dice 70 %, ¿gana el 70 %?), por el decil más fiable de cada día (ganan, fiabilidad, neto, exceso, con t agrupada) y, sobre todo, frente a un **control «solo volatilidad»** (el mismo modelo con una sola variable, el percentil de ATR/precio): si no lo supera, la «fiabilidad» es solo que el valor se mueve. Con una reversión plantada el modelo la aprende fuera de muestra y está calibrado; en paseos aleatorios no inventa nada (pruebas).

#### Resultados con la escalera (confirmación del 27-oct-2026, 300 repeticiones del control)

La escalera (`esc`: stop inicial 2,5 ATR; al cerrar a +4 % stop plano a +4 %; al cerrar a +10 % stop a +10 % que persigue a 2 ATR) sobre la regla base, en los dos universos:

| | universo | n | ganan (todas las filas) | aseguró +4 % / +10 % / stop | gana media / pierde media · factor | exceso (t) | fiabilidad (puntos, t) | años + |
|---|---|---|---|---|---|---|---|---|
| **C6** impulso 12-1 alto Y bajo su media de 200 · esc | S&P 500 + N-100 (desde incorporación) | 13.702 | **72,2 %** (58,6 %) | 73 / 21 / 26 % | +6,2 % / −9,1 % · 1,78 | +0,84 % (3,8) | +7,6 (6,4) | 9/10 |
| **C6** ídem | **S&P 400 y 600** (acciones distintas) | 18.614 | **71,6 %** (60,8 %) | 73 / 21 / 27 % | +6,4 % / −12,1 % · 1,32 | +0,47 % (2,2; p 0,06) | +6,8 (6,1) | 9/10 |
| **C8** impulso 12-1 alto Y 2+ señales bajistas · esc | S&P 500 + N-100 | 10.252 | 66,8 % (58,6 %) | 68 / 17 / 32 % | +6,0 % / −7,8 % · 1,57 | +0,75 % (3,4) | +6,1 (6,4) | 9/10 |
| **C8** ídem | S&P 400 y 600 | 10.749 | 68,6 % (60,8 %) | 70 / 19 / 30 % | +6,4 % / −11,1 % · 1,25 | +0,75 % (3,9; p 0,003) | +6,8 (7,6) | 8/10 |
| C7 ídem que C6 con stop al 6 % del precio | S&P 500 + N-100 | 13.715 | 60,0 % (58,2 %) | 61 / 17 / 39 % | +6,1 % / −6,4 % · 1,44 | +0,50 % (3,0) | +1,5 (1,7) | 8/10 |
| C7 ídem | S&P 400 y 600 | 18.626 | 54,0 % (53,1 %) | 55 / 15 / 45 % | +6,1 % / −6,6 % · 1,08 | +0,21 % (1,6; p 0,11) | +0,7 (0,8) | 9/10 |
| V6 = C6 solo volatilidad alta | S&P 500 + N-100 | 9.254 | 73,0 % | — | — | +1,20 % (4,2) | — | 9/10 |
| V6 | S&P 400 y 600 | 12.707 | 72,2 % | — | — | +0,80 % (2,5) | — | 9/10 |
| V5 = C6 solo volatilidad baja | S&P 500 + N-100 | 2.385 | 70,5 % | — | — | −0,03 % (−0,1) | — | 2/6 |
| V5 | S&P 400 y 600 | 3.274 | 71,9 % | — | — | −0,81 % (−2,2) | — | 0/6 |

Lectura honesta, en el orden que importa a quien opera:

1. **Con la escalera, la regla gana 7 de cada 10 operaciones, asegura el +4 % en el 73 % y el +10 % en 1 de cada 5**, en acciones distintas de las que sirvieron para encontrarla y en 9 de 10 años. Eso es lo que pide un operador y está medido.
2. **Pero entrar al azar el mismo día con la misma escalera también gana el 59–61 %**: con un stop a 2,5 ATR y un primer nivel a solo +4 %, en una década alcista casi cualquier acción volátil llega antes a +4 % que a −2,5 ATR. La parte que pone la regla son **+7 a +8 puntos de acierto y +0,5 a +0,8 % de exceso por operación**, con t entre 2 y 4 y p-valores del control por debajo de 0,05 (C6 en el S&P 400/600 queda en el borde: p 0,06 del exceso, 0,047 de la t).
3. **La pérdida media es mayor que la ganancia media** (−9 a −12 % frente a +6 %): el R:R por operación es malo y lo compensa la tasa de acierto. Factor de beneficio 1,3 (acciones medianas) a 1,8 (grandes). **Un stop al 6 % del precio (C7) destroza la regla**: salta el 39–45 % de las veces y la ventaja desaparece. El stop tiene que ser ancho (2,5 ATR, del 10 al 25 % en estos valores) y el riesgo se controla con el tamaño de la posición, no con el stop.
4. **Solo en acciones volátiles** (V6 contra V5): en las tranquilas la escalera no aporta nada o resta. El plan de hoy solo las toma del 40 % más volátil.
5. El sesgo de supervivencia (los que salieron del índice no están) infla estas cifras, sobre todo en valores volátiles que cayeron. Lo prudente es contar con la mitad del exceso. El **seguimiento hacia delante** dirá si se sostiene.

#### Resultados del modelo de fiabilidad (27-oct-2026, 509.000 operaciones fuera de muestra, 2020–2026)

| modelo · filas | AUC | ganan (todas) | decil más fiable del día: ganan · fiabilidad (t) · neto · exceso | decil menos fiable: ganan |
|---|---|---|---|---|
| completo · todas | 0,564 | 58,8 % | 67,0 % · +8,1 (6,9) · +0,74 % · +0,29 % | 51,4 % |
| **solo volatilidad** · todas | 0,560 | 58,8 % | 68,3 % · +9,4 (8,1) · +0,85 % · +0,41 % | 49,2 % |
| completo · dentro de C6 | 0,603 | 72,8 % | 75,5 % · +13,9 (7,4) · +2,04 % · +1,25 % (t 3,7) | 69,6 % |
| solo volatilidad · dentro de C6 | 0,530 | 72,8 % | 75,6 % · +14,0 (6,4) · +2,28 % · +1,48 % | 68,2 % |

**Veredicto**: el modelo completo **no supera al control de solo volatilidad** en el conjunto (AUC 0,564 frente a 0,560; el decil más fiable gana lo mismo con una cosa que con la otra). Dentro de la regla separa algo más por AUC (0,603 frente a 0,530) pero no en el decil más fiable. Es decir: **la «fiabilidad» que se puede predecir es, casi toda, que el valor se mueve mucho**. Además el modelo es optimista en los extremos (cuando dice 88 % gana el 77 %; cuando dice 24 % gana el 56 %): la web muestra la probabilidad **calibrada** con lo que ganó de verdad cada tramo fuera de muestra. La columna sirve para ordenar candidatas, no como promesa. Esto es lo que había que encontrar, aunque no sea lo que se querría oír: con datos de precio y volumen no hay una señal fuerte de «este trade va a salir bien» más allá de la volatilidad y de la regla misma.

### ¿Da para vivir? La cartera simulada y el menú de salidas (`vp/cartera.py`)

Las estadísticas por operación no dicen qué pasa en una cuenta. `python -m vp cartera` simula **día a día, con los precios reales**, exactamente la cartera que calcula la pestaña Hoy: **5 huecos**; las posiciones abiertas ocupan hueco y solo se rellenan los huecos libres con las acciones de la regla de mayor **fiabilidad** ese día (el modelo de `vp/fiab.py`, **fuera de muestra**: entrenado cada año solo con lo cerrado 95 días antes; por eso la ventana empieza donde hay modelo, desde el cuarto año de datos), con volatilidad alta (percentil ≥ 60), sin valores con correlación ≥ 0,6 (rendimientos a 60 sesiones) con los abiertos; entrada en la apertura siguiente; **tamaño con el que el stop inicial a 2,5 ATR cuesta el 1 % de la cuenta** (nunca más del 20 % de la cuenta, sin apalancamiento); valoración diaria a cierre; cierre con el resultado real de la salida (el mismo simulador que la web y el seguimiento). Se compara con:

- **comprar todas las señales de la regla** (sin límite de posiciones ni de correlación, mismo tamaño, con lo que haya en caja): ¿elegir 5 por fiabilidad es mejor que comprarlo todo?;
- **carteras al azar del universo** (20): entran los mismos días, el mismo número de veces, en acciones al azar con la misma volatilidad alta y la misma salida. Es lo que pone la regla; p = fracción de carteras al azar que igualan o superan a la del plan;
- **las mismas 5 en orden de volatilidad y en orden al azar**: lo que pone la fiabilidad;
- **comprar y mantener el universo a partes iguales** (sin costes): el mercado;
- y, como contexto, toda la historia desde 2017 sin modelo (orden por volatilidad).

Se mide rentabilidad anual compuesta, caída máxima, Sharpe, rentabilidad por año, operaciones al año, % que ganan, exposición media, y variantes de la misma cartera (riesgo 0,5 % y 2 %, 3 y 10 posiciones, sin grupos, sin filtro de volatilidad). El **menú de salidas** opera la misma entrada con 13 salidas fijadas de antemano (la escalera; con colchón: asegura +4 % al cerrar a +6 % y +10 % al cerrar a +12 %; que deja correr hasta +20 %; con persecución a 3 ATR; alta: +6 % → +15 %; asegura +4 % o +6 % y persigue; stop que persigue a 3 o 5 ATR; stop fijo y 20 sesiones; cierre a 10, 20 o 40 sesiones), por operación (neto, azar del mismo día, exceso con t agrupada, ganan, niveles asegurados, ganancia y pérdida medias, sesiones y **neto por 100 sesiones de capital ocupado**) y en cartera. Es una comparación hecha después de ver los datos: la mejor fila es una referencia, no una regla nueva. Con una reversión plantada la cartera de la regla gana a las carteras al azar (p = 0) y en paseos aleatorios no (pruebas).

#### Resultados de la cartera simulada (27-oct-2026: 500 acciones desde su incorporación al S&P 500, mayo 2017 → octubre 2026, 20 carteras al azar)

| cartera | al año | caída máxima | Sharpe | anual / caída | años + | peor año | operaciones |
|---|---|---|---|---|---|---|---|
| **«bajo 200» con la escalera y la disciplina del plan de hoy** | **+7,0 %** (+89 % en total) | **−17,2 %** | 0,66 | 0,40 | 7/10 | −10,7 % | 665 (70 al año) · ganan 71 % · neto medio +1,1 % · 13 sesiones · cuenta invertida el 36 % de media |
| al azar en el universo (volatilidad alta), mismas fechas y mismo número de entradas | +2,4 % (p05 −0,5 %, p95 +6,3 %) · **p 0,05** | −21,0 % | | | | | 566 |
| las mismas acciones de la regla en orden al azar | +5,9 % (p 0,15) | −17,3 % | | | | | |
| comprar y mantener el universo a partes iguales (sin costes) | +12,7 % | −39,1 % | 0,63 | 0,33 | 8/10 | −11,3 % | — |

Por año (cartera / mercado / azar): 2017 −1,2 / +13,4 / +4,8 · 2018 −1,3 / −6,9 / −2,4 · 2019 +24,0 / +30,0 / +15,8 · 2020 +2,1 / +13,4 / +4,0 · 2021 +14,1 / +28,6 / +9,2 · 2022 −10,7 / −11,3 / −8,3 · 2023 +17,0 / +16,6 / +5,5 · 2024 +12,0 / +13,2 / −0,7 · 2025 +8,5 / +14,4 / +2,2 · 2026 +5,5 / +15,2 / +1,2. Variantes: riesgo 2 % → +11,3 %/año con caída −28,9 %; riesgo 0,5 % → +3,6 % y −8,9 %; 3 posiciones → +5,1 % y −13,3 %; 10 posiciones → +7,2 % y −27,3 %; sin la regla de un valor por grupo → +6,8 % y −19,9 %; sin el filtro de volatilidad → +6,7 % y −17,7 %.

Lectura:

1. **La regla gana a elegir al azar las mismas fechas** (+7,0 % frente a +2,4 % al año, p 0,05: en el borde) y con menos caída (−17 % frente a −21 %). Es la prueba de que la selección vale algo en una cuenta, no solo por operación.
2. **No gana al mercado a partes iguales en rentabilidad bruta** (+12,7 %) porque la cuenta está invertida solo el 36 % de media con riesgo del 1 % y stops anchos; por unidad de caída máxima sí (0,40 frente a 0,33). Subir el riesgo al 2 % da +11,3 %/año con −29 % de caída: la relación no cambia. **Para vivir de esto no da**: 7 % al año con dos años negativos seguidos (2017–2018) y un −11 % en 2022.
3. **La ventaja por operación está concentrada en rachas** (rebotes tras caídas del mercado, cuando muchas acciones cumplen la regla el mismo día): una cuenta con 5 huecos solo coge una parte. El neto medio de las operaciones que de verdad hace la cartera es +1,1 %, frente al +2,0 % de todas las filas de la regla.
4. **El orden por volatilidad aporta poco** (+7,0 % frente a +5,9 % en orden al azar, p 0,15), igual que el modelo de fiabilidad.
5. **Las otras dos entradas no se distinguen del azar en cartera** («2+ bajistas» +3,2 %/año, p 0,5; «RSI(2) ≤ 10» +3,2 %, p 0,45) aunque su exceso por operación sea positivo: con 5 huecos y la escalera, lo que aguanta es «bajo 200».

#### Menú de salidas sobre la entrada «bajo 200» (volatilidad alta; ordenado por lo que da en cartera)

| salida | n | neto (azar del día) | exceso (t) | ganan (azar) | 1º / 2º / stop | sesiones | neto / 100 ses. | cartera: al año | caída | azar (p) |
|---|---|---|---|---|---|---|---|---|---|---|
| asegura +6 % y persigue 4 ATR | 9.238 | +2,65 % (+1,06 %) | +1,59 % (3,8) | 67 % (54) | 67 / 0 / 32 | 19 | +13,7 | **+7,6 %** | −17,8 % | +2,4 % (< 0,05) |
| **tu escalera** (+4 % plano → +10 % persigue 2 ATR) | 9.254 | +1,97 % (+0,76 %) | +1,20 % (4,2) | **73 %** (61) | 74 / 22 / 26 | **13** | **+15,6** | **+7,0 %** | **−17,2 %** | +2,4 % (0,05) |
| escalera alta (+6 % → +15 %) | 9.244 | +2,28 % (+0,90 %) | +1,38 % (4,4) | 67 % (54) | 67 / 16 / 32 | 16 | +14,2 | +6,5 % | −19,4 % | +2,2 % (0,05) |
| escalera con persecución a 3 ATR | 9.254 | +2,04 % (+0,79 %) | +1,25 % (4,3) | 73 % (61) | 74 / 22 / 26 | 13 | +15,7 | +6,3 % | −17,1 % | +2,5 % (0,05) |
| escalera que deja correr (+4 → +10 → +20) | 9.254 | +1,80 % (+0,73 %) | +1,07 % (4,4) | 73 % (61) | 69 / 17 / 26 | 12 | +15,0 | +6,3 % | −17,1 % | +2,6 % (0,05) |
| stop 5 ATR que persigue a 5 ATR, sin asegurar | 9.126 | +5,91 % (+2,83 %) | +3,08 % (2,8) | 55 % (48) | 0 / 0 / 45 | 48 | +12,2 | +5,6 % | −24,8 % | +4,5 % (0,35) |
| asegura +4 % y persigue 3 ATR | 9.249 | +2,45 % (+0,98 %) | +1,47 % (3,5) | 73 % (61) | 74 / 0 / 26 | 16 | +15,3 | +5,5 % | −17,8 % | +3,4 % (0,05) |
| escalera con colchón (+6 % asegura +4; +12 % asegura +10) | 9.236 | +2,63 % (+1,07 %) | +1,56 % (3,9) | 67 % (54) | 67 / 30 / 32 | 19 | +13,9 | +4,9 % | −18,2 % | +3,0 % (0,2) |
| cierre a 20 sesiones, sin stop | 9.222 | +3,12 % (+1,60 %) | +1,52 % (3,8) | 59 % (56) | — | 20 | +15,6 | +3,9 % | −20,2 % | +3,0 % (0,3) |
| cierre a 40 sesiones, sin stop | 9.094 | +5,99 % (+3,31 %) | +2,68 % (3,5) | 62 % (58) | — | 40 | +15,0 | +3,6 % | −24,2 % | +4,7 % (0,6) |
| stop 2,5 ATR que persigue a 3 ATR, sin asegurar | 9.225 | +4,07 % (+1,91 %) | +2,17 % (2,7) | 46 % (40) | 0 / 0 / 80 | 31 | +13,1 | +2,3 % | −21,3 % | +3,8 % (0,85) |
| cierre a 10 sesiones, sin stop | 9.274 | +1,77 % (+0,68 %) | +1,09 % (4,3) | 58 % (55) | — | 10 | +17,7 | +1,9 % | −23,2 % | +2,1 % (0,55) |
| stop 2,5 ATR fijo y cierre a 20 sesiones | 9.244 | +2,38 % (+1,21 %) | +1,17 % (2,8) | 54 % (50) | 0 / 0 / 29 | 17 | +13,9 | +1,7 % | −20,6 % | +3,0 % (0,65) |

Lectura del menú: **tu escalera queda segunda de 13 en cartera** (+7,0 % frente a +7,6 % de asegurar +6 % y perseguir a 4 ATR, una diferencia dentro del ruido de una comparación hecha después) y es la que más gana por sesión de capital ocupado entre las que aseguran algo, con la caída más baja. Las salidas que ganan más por operación (perseguir a 5 ATR +5,9 %, cierre a 40 sesiones +6,0 %) duran 40–48 sesiones, en cartera rinden menos y con más caída, y **no se distinguen del azar** (p 0,35–0,85): su ganancia la pone el mercado de esos días. El colchón (asegurar +4 % al cerrar a +6 %) no mejora nada: asegura menos veces y en cartera da +4,9 %. Las salidas por tiempo sin stop, que por operación parecían buenas (sección de confirmación), en una cartera con 5 huecos no aportan nada sobre el azar. **No hay una salida mejor escondida**: con esta entrada, la escalera que ya usas es de las mejores que se pueden poner.

### El cribador diario, el plan de hoy y el seguimiento (`vp/screen.py`)

`python -m vp screen` (o el flujo «Cribador de la regla de impulso», de lunes a viernes a las 23:30 UTC) aplica al **último cierre** del universo (S&P 500 + Nasdaq-100, las 300 más negociadas ese día, 10 años de historia) las **tres entradas** con nombre (`search.ENTRADAS`: «bajo 200», «2+ bajistas», «RSI(2) ≤ 10»; salida: la escalera) con **exactamente las mismas condiciones y los mismos percentiles del día** que la búsqueda y la confirmación (`search.conditions`), de modo que una acción de la tabla es exactamente una fila de la estadística. Una prueba comprueba que da las mismas acciones que el panel completo en 5 fechas distintas, sin mirar el futuro.

- **Por acción**: cierre, impulso, distancia a la media de 200, RSI(2), ATR, percentil de volatilidad, señales bajistas, reglas que cumple (y cuáles son nuevas), **fiabilidad** (probabilidad de ganar con la escalera, con el modelo entrenado con todo lo cerrado hasta ayer; se lee frente a la media del universo ese día), **grupo de correlación** (valores con correlación de rendimientos ≥ 0,6 en 60 sesiones: comprar varios es repetir la apuesta) y el plan: stop inicial a 2,5 ATR (y al 6 % del precio como alternativa), niveles de +4 % y +10 %, R:R hasta +4 % y **tamaño con el que perder el stop inicial cuesta el 1 % de la cuenta**.
- **Plan de hoy: la cartera de 5 huecos, con estado** (`screen.daily_plan`). Las posiciones que la cartera del plan tiene abiertas (según el seguimiento) **ocupan hueco**; solo se rellenan los huecos libres con las acciones del día de mayor fiabilidad, con volatilidad alta (donde se midió la ventaja), que no estén ya abiertas ni tengan correlación ≥ 0,6 (rendimientos a 60 sesiones) con ninguna abierta ni con otra elegida; riesgo del 1 % de la cuenta cada una. **Con los 5 huecos llenos no entra nada** hasta que cierre alguna (por su stop o a las 60 sesiones). La pestaña Hoy enseña las abiertas (con el stop en vigor), los huecos libres y las entradas nuevas; la tabla de abajo lista todas las acciones que cumplen la regla, entren o no. Es exactamente la cartera que simula `vp/cartera.py`. En la web se puede cambiar el riesgo por operación y poner el tamaño de la cuenta para ver importes (se guardan solo en el navegador). El registro se puede **replanificar** (`python -m vp screen --replan`; entrada `replan` del flujo) cuando cambia la regla del plan: recalcula el plan de cada día en orden, solo con las candidatas guardadas ese día y las barras hasta esa fecha; con el mismo código es idempotente.
- **Ninguna acción se descarta** por tener un stop lejano (en valores muy volátiles llega al 15–25 %): se marca con ⚠ y el tamaño de posición ya lo compensa. Por defecto la tabla muestra solo las de **volatilidad alta** (percentil ≥ 60 de ATR/precio del día); las de volatilidad baja se pueden ver quitando el filtro y salen con un aviso.
- **Seguimiento hacia delante** (`data/impulso_track.json`, pestaña Seguimiento), dos registros: la **cartera del plan** (`plan_ops`: solo lo que el plan mandó comprar cada día, una posición por valor, con el stop en vigor de cada abierta) y **todas las señales**. Cada día se añade al registro (`impulso_hist.json`) qué acciones cumplían cada regla, con su cierre, ATR, fiabilidad y grupo, y qué entró en el plan, y se simula con los **precios reales posteriores** lo que pasó con cada una con la escalera (entrada en la apertura siguiente, stop 2,5 ATR del día de la señal, una operación por valor y día; si ya había una abierta no se abre otra). Da operaciones abiertas y cerradas, % que ganan, aseguradas +4 % / +10 %, neto medio, ganancia y pérdida medias, por regla, para el plan de hoy y para volatilidad alta. **Señales repetidas**: si un valor ya tiene una operación abierta y vuelve a cumplir la regla (o a entrar en el plan de hoy) en un día posterior, no se abre otra, pero la señal se anota en la operación abierta (`repite`, `plan_dias`): la web muestra «×N» y la etiqueta PLAN con la fecha, y la pestaña Hoy marca «ya abierta» en los valores del plan que el seguimiento ya tiene comprados, de modo que el plan y el seguimiento cuadran (el plan cuenta esas acciones como posiciones que ya tienes, no como compras nuevas). Una operación que no llega a abrirse (la apertura ya está más allá del stop) no bloquea futuras señales del valor. **Es el track record de la regla: no se puede ajustar a posteriori** y crece cada día. Mientras sea corto (semanas) no demuestra nada; con cientos de operaciones será la prueba definitiva, para bien o para mal.

## Común a las dos partes

### Lista de seguimiento

`watchlist.txt` (raíz del repositorio) manda: un símbolo de Yahoo por línea o varios separados por espacios o comas, `#` comenta. El escaneo diario la lee entera; para añadir o quitar valores basta editarla desde GitHub (lápiz) y el siguiente escaneo, o lanzarlo a mano desde Actions, los incluye. Un valor que Yahoo no devuelve o con menos de 400 sesiones se avisa en el Radar. Cada valor añade ~75 KB de `data/charts` al repositorio en cada escaneo, así que conviene no pasar de unos 150.

Los ★ favoritos se guardan solo en el navegador (localStorage).

### Estructura

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
  search.py           (parte 2) búsqueda de la mejor regla con selección hacia delante, nulo del procedimiento y confirmación
  fiab.py             (parte 2) modelo de fiabilidad walk-forward (probabilidad de ganar con la escalera) con control de solo volatilidad
  screen.py           (parte 2) cribador diario: acciones que cumplen cada regla, plan de la escalera, fiabilidad, grupos de correlación, plan de hoy y seguimiento con precios reales
  cartera.py          (parte 2) cartera simulada día a día con la disciplina del plan de hoy frente a carteras al azar y al mercado; menú de salidas sobre la misma entrada
  data.py, universe.py   descarga de Yahoo y universos (watchlist.txt)
data/                 parte 1: charts/<VALOR>.json, index.json, catalog.json, evidence.json, informes (evidence_report*, plan_report, ml_report, context_report)
                      parte 2: impulso.json, impulso_hist.json, impulso_track.json, search.json, search_confirm*.json, fiab.json, cartera.json, informes (search_report*, search_confirm*, fiab_report, cartera_report)
tests/test_vp.py      pruebas (sin red)
.github/workflows/    scan.yml (L–V 22:45 UTC) · impulso.yml (L–V 23:30 UTC) · evidence.yml (domingos) · research.yml (manual)
```

### Uso

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
python -m vp search                           # parte 2: búsqueda de la mejor regla (selección hacia delante + nulo)
python -m vp search --confirm --universe smallmid   # parte 2: confirmación de las 5 candidatas en otras acciones
python -m vp fiab                             # parte 2: modelo de fiabilidad walk-forward → data/fiab.json
python -m vp cartera [--reps 20]              # parte 2: cartera simulada con la disciplina del plan de hoy y menú de salidas → data/cartera.json
python -m vp screen                           # parte 2: cribador de hoy, plan de hoy y seguimiento → data/impulso*.json
python -m http.server                         # abrir http://localhost:8000
```

Los precios vienen de Yahoo Finance (split aplicado, sin ajustar por dividendos para que los niveles coincidan con los del bróker); se descarta la barra de la sesión aún abierta.
