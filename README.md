# VP Scanner v2

Escáner diario de ~2.000 activos (S&P 500/400/600, Nasdaq-100, ETF, Europa, cripto…) que avisa de **compras tras una caída dentro de una tendencia alcista** cuando un modelo estadístico, validado año a año, le da probabilidad de salir bien. Todo corre gratis en GitHub Actions; la web (`index.html`) solo lee los JSON.

> **Expectativas realistas.** La ventaja medida es modesta: **≈67 % de acierto y ≈ +0,7 % por operación neto de costes** (plan *Equilibrado*, las 3 mejores del día, positivo en los 8 años 2019–2026), o ≈70 % de acierto y ≈ +0,4 % (plan *Alta probabilidad*, sin ventaja apreciable sobre el S&P 500). **Ojo: el acierto de una operación concreta no se puede predecir** (ver «¿Están bien estimadas las probabilidades?»): lo estable es el acierto histórico de cada plan y puesto, no una probabilidad individual. No existe, en estos datos, una estrategia de ≥75 % de acierto *y* rentable: cuando el acierto sube, cada fallo (−4/−6 %) pesa más que cada acierto (+1/+1,5 %) y la ganancia media se evapora. La prueba definitiva es la pestaña **Seguimiento real**.

## Qué estaba mal en la versión anterior

| | Backtest antiguo | En vivo |
|---|---|---|
| Acierto | 63 % (+5,7 %/op) | **30 %** (−0,77 %/op) en 995 operaciones; los LONG, 8 % |

Causas (todas corregidas):

1. **Fuga de información del futuro.** El perfil de volumen de cada tramo pivote→pivote se calculaba con barras *posteriores* a la entrada (y un pivote necesita 20 barras futuras para existir). En vivo solo existe el tramo en formación → otra distribución.
2. **Sobreajuste.** Las «clases élite» se escogieron mirando el mismo histórico (los `lab_*`), con stops calculados solo con las operaciones ganadoras.
3. **Entradas no operables** (al cierre de la barra que dispara la señal) y empates stop/objetivo resueltos a favor.
4. **Comprar caídas en pánico** (VIX alto): con VIX en estrés el acierto cae a ~50 % y se pierde ≈1,2 %/op.

## Cómo funciona ahora

1. **Candidatos** (`setups.py`): 17 patrones de sobreventa dentro de tendencia (precio > SMA200 con RSI(2) bajo, rachas bajistas, mínimos de 5/10/20 sesiones, Bollinger, precio bajo el VAL del perfil de volumen móvil…). Solo largos: los cortos no tuvieron ventaja en ninguna variante.
2. **Probabilidad** (`model.py`): regresión logística lineal (poca capacidad) con 28 variables de contexto — VIX y SPY, **amplitud de mercado** (qué parte del universo está sobre su SMA200 y cuánta está en sobreventa), tamaño de la caída, posición respecto a máximos… Se reentrena cada semana con los últimos 3 años.
3. **Política**: cada día solo se alertan las **mejores** (3 en *Equilibrado*, 5 en *Alta probabilidad*) cuya probabilidad supera la tasa base de acierto; las mejor clasificadas rinden más. Los grupos de activos sin evidencia (p. ej. valores temáticos) se excluyen solos.
4. **Dos planes de salida** por alerta: *Equilibrado* (sale cuando RSI(2) > 70) y *Alta probabilidad* (objetivo +1 ATR). Stop a 4×ATR y máximo 10 sesiones en ambos.
5. **Operativa honesta** (`simulate.py`): señal con el cierre, **entrada en la apertura siguiente**, huecos de apertura, un día que toca stop y objetivo cuenta como **stop**, costes por tipo de activo (0,10–0,40 % ida y vuelta).
6. **Seguimiento en vivo** (`track.py`): cada alerta se registra y se simula con *exactamente* el mismo código que el backtest; la web compara acierto real vs esperado.
7. **Chequeo de salud**: si el rendimiento de los últimos 12 meses (fuera de muestra) no es rentable, ese plan se **pausa solo** y deja de alertar.

### Validación (walk-forward 2019–2026, cada año predicho solo con los 3 anteriores)

| Plan | Acierto | Media/op. | Factor beneficio | Desde 2024 | Costes ×2 |
|---|---|---|---|---|---|
| Equilibrado (3 mejores/día, ATR ≥ 2,5 %) | 66,6 % | +0,70 % | 1,38 | 64,9 % / +0,61 % | +0,49 % |
| Alta probabilidad (5 mejores/día, ATR ≥ 2,5 %) | 69,5 % | +0,38 % | 1,20 | 69,2 % / +0,31 % | +0,17 % |

Lo que mostró la investigación (`scanner/lab.py`, informes en `scanner/model/last_report.txt`):

- Comprar caídas sin filtrar: ~62 % de acierto y **≈0 % de media**; negativo en años de estrés (2018, 2022).
- Un modelo complejo con umbral fijo parecía dar 74 %/+1 % en validación, pero **fracasó en la prueba ciega 2024+** (60,8 %, +0,05 %): se había ajustado a los episodios 2020–2022. Lo estable fue: modelo simple + ventana móvil + «las N mejores del día».
- **El perfil de volumen no aporta capacidad predictiva** (resultado idéntico sin él). Quitar VIX/SPY o la amplitud de mercado sí destruye la ventaja.
- Las probabilidades mostradas están **calibradas** también en 2024+ (p. ej. 63,9 % predicho → 63,8 % real).
- Ya no queda periodo ciego: lo que valida el sistema a partir de ahora es el seguimiento en vivo.

## Ir tarde: el hueco de apertura y el aviso previo al cierre

Las alertas salen con el cierre y se compra en la apertura siguiente; si la acción abre con un hueco alcista grande, parte de la ganancia esperada ya se la ha llevado el mercado. Medido en 2019–2026:

- **El hueco no destruye la ventaja**, pero sí castiga al plan *Equilibrado* cuando abre muy por encima: con aperturas > referencia + 0,5 ATR perdió de media ≈ −0,8 % por operación (306 casos). El plan *Alta probabilidad* no se resintió. Por eso la web muestra un **precio máximo orientativo de entrada** (referencia + 0,5 ATR) y los niveles de stop/objetivo se miden **desde tu precio de entrada real**, no desde el cierre de ayer (el fallo original era de presentación: mostraba niveles del cierre).
- **Escaneo previo al cierre (MOC)**: `scan.yml` corre tras el cierre, pero `scan_preclose.yml` corre a las **15:20 ET** (19:20/20:20 UTC según el horario de verano; el script comprueba la hora de Nueva York y solo actúa entre 15:10 y 15:40 ET) y repite la misma lógica sobre la vela *provisional* de hoy, solo en EE.UU. (grandes, medianas, pequeñas y ETF). Avisa con tiempo para colocar una **orden MOC antes de las 15:50 ET (21:50 en España)** y entrar al precio de cierre, sin hueco. Los avisos van a `alerts_pre.json`, a la sección «Preliminares» de la web y a Telegram.
  - Validación (`research.py moc`, barras de 60 min reconstruyendo la vela de las 15:30 ET, ≈ 2 años): comprar al cierre con la señal provisional dio ≈ +0,68 % por operación frente a ≈ +0,56 % comprando a la apertura con la señal definitiva (plan *Equilibrado*): **≈ +0,1 % de mejora, dentro del margen de error (±0,1)**. Para *Alta probabilidad* no hay diferencia apreciable. Solo ≈ 30 % de las señales provisionales coinciden luego con la lista definitiva (el top-N se reordena mucho cerca del cierre), pero las que desaparecen rinden igual que las que se confirman. Ojo: el modelo ya vio esos años al entrenar y es una reconstrucción, no operativa real.
  - **Temáticos (p. ej. Equinor): no validado**, así que no entran en el aviso previo. Con ≈ 370 señales provisionales la diferencia cierre − apertura fue +0,07 % ± 0,34 (inconcluyente) y solo el 44 % se confirmaba al cierre. Además, el plan *Alta probabilidad* es el menos sensible al hueco, así que no hay ganancia que justifique el riesgo.
  - El seguimiento registra estas operaciones aparte (`mode: moc`, entrada = cierre final) y el escaneo posterior al cierre **no las duplica**; si una señal provisional no se confirma con el cierre final, sigue en el seguimiento (para medir el coste real de actuar antes).
  - Si no puedes poner la orden a tiempo: no pasa nada, esperas a las alertas definitivas y entras a la apertura.

## Qué se probó para mejorar la fiabilidad y la rentabilidad (y qué funcionó)

Todo con la misma tubería walk-forward (`research.py improve`, `improve2`, `index`; informes en la rama `claude/research-results-<cmd>`), comparando año a año con el plan desplegado. Con tantas pruebas, una mejora aislada puede ser azar: solo se adopta lo que mejora en casi todos los años, se replica con salidas vecinas y tiene una razón a priori.

| Prueba | Resultado | Decisión |
|---|---|---|
| **Puesto del día** (media por puesto, Equilibrado) | #1 +0,80 % · #2 +0,52 % · #3 +0,41 % · #4 +0,34 % · #5 +0,39 %; tendencia monótona, #1 positivo 7 de 8 años | **Adoptado: Equilibrado pasa a las 3 mejores del día** (cartera de 10 posiciones: Sharpe 1,42 frente a 1,31, menos caída, mismo CAGR). *Alta probabilidad* se queda en 5 (solo el #1 destaca: +0,48 % frente a +0,1–0,2 %). |
| Puertas de régimen a mano (VIX alto, SPY bajo su SMA200, amplitud baja, sobreventa amplia, ATR alto) | **Todas empeoran**: las señales descartadas rendían más (p. ej. VIX z > 1,5: +1,04 %; amplitud < 30 %: +1,29 %) | Descartado: el modelo ya elige bien en pánico |
| Ventana de entrenamiento 2/4/5 años o creciente; regularización | 3 años es igual o mejor en *Equilibrado* | Sin cambio |
| Variables nuevas (choques de noticias, momentum, corto plazo, mercado) | No mejoran *Equilibrado*; «mercado» mejoró *Alta probabilidad* (+0,13 %) pero no se replicó en todas las salidas vecinas (3 de 5) | Descartado (sospecha de azar) |
| Tope por sector y día | Casi nunca limita, sin efecto | Sin cambio |
| Modelo frente a elegir 5 al azar entre los candidatos | *Equilibrado*: +0,50 % frente a +0,08 %, mejor en **8 de 8 años**. *Alta probabilidad*: +0,24 % frente a +0,18 % (6 de 8 años) | El modelo aporta en *Equilibrado*; en *Alta probabilidad* aporta poco |
| Entrada al cierre (MOC) | ≈ +0,1 % por operación (ver arriba) | Aviso previo mantenido |

**Qué esperar con una cartera real** (simulación con capital limitado desde 2019, máx. 10 posiciones al 10 % cada una, netas de costes; la web la regenera cada semana):

| | Anual | Caída máx. | Sharpe | Meses + | Peor mes |
|---|---|---|---|---|---|
| Equilibrado (3 mejores/día) | +31,4 % | 27,6 % | 1,70 | 68 % | −21 % |
| Alta probabilidad (5 mejores/día) | +18,7 % | 22,0 % | 1,50 | 68 % | −13 % |
| Comprar y mantener el S&P 500 (2019–2026) | +17,5 % | 33,7 % | 0,94 | | |

- **Hallazgo incómodo**: en las mismas ventanas de cada operación, comprar el propio S&P 500 acertó casi igual (≈ 65 % frente a ≈ 67 %) y rindió +0,40 % frente a +0,70 % (*Equilibrado*). El «alfa» sobre el índice es ≈ +0,30 % (t ≈ 2,6) en *Equilibrado* y ≈ −0,04 % (t ≈ −0,4) en *Alta probabilidad*: **gran parte de la ventaja viene de *cuándo* se compra (tras caídas en un mercado alcista), no de *qué* acción se elige**, y la parte de selección queda además inflada por el sesgo de supervivencia.
- La misma idea aplicada **solo a índices** (SPY/QQQ/IWM/DIA, sin modelo y sin sesgo de supervivencia): 513 señales en 10 años, **71 % de acierto, +0,30 % por operación** (t = 2,5; 8 de 10 años positivos), y con VIX tranquilo (z < 0,5) 72 % y +0,36 %. Una cartera «solo SPY» (≈ 15 operaciones al año) dio ≈ +6 % anual con caída máxima ≈ 7 % y 79 % de meses positivos: fiable pero de poca rentabilidad si no se apalanca. No está en producción; es una opción de bajo riesgo.
- Un 65 % de operaciones ganadoras **no** implica una curva suave: en la simulación solo ≈ 6 de cada 10 meses son positivos y los peores meses superan el −10 %. Usa posiciones pequeñas.

## ¿Están bien estimadas las probabilidades? Estudio ampliado (`research.py reasons | reasons2`)

**No, las individuales no.** Se midió fuera de muestra (walk-forward, 2019–2026) si la probabilidad del modelo separa las operaciones que ganan de las que pierden:

| | *Equilibrado* | *Alta probabilidad* |
|---|---|---|
| AUC (0,5 = azar) | 0,52 (0,57–0,58 en 2019–22; **0,51–0,53 desde 2023**) | 0,51 (**≤ 0,51 en 2023–25**) |
| Brier frente a repetir la tasa base | **peor** (−4 % la cruda, −2 % calibrada solo con años previos) | **peor** (−7 % / −3 %) |
| Acierto real por décima de «probabilidad» | 61 % → 68 %, casi plano (**plano desde 2023**) | 70 % → 71 %, plano |
| Probabilidad mostrada − acierto real, por año | error medio 3,6 puntos, hasta −12,5 y +6,2 | 3,8 puntos, hasta −11 y +7,6 |

Es decir: casi todo el «65–72 %» que se mostraba era la **tasa base** con falsa precisión. **Decisión:** la web y los avisos muestran ahora el **acierto histórico del puesto del día** (con su rango por año) y la media histórica, y dicen claramente que no es la probabilidad de esa operación.

**¿Y los motivos de compra?** De los 17 patrones, solo unos pocos mejoran de forma estable la media de los candidatos (≥ 4–5 cierres bajistas seguidos, RSI(2) < 5, caída > 12 % desde máximos, capitulación); otros, como «mínimo de cierre de 5 o 10 sesiones», no aportan nada, y combinar varios patrones a la vez tampoco ayuda (acierto 63,4 % con uno, 63,1 % con cuatro o más). Lo que sí separa de forma estable (mismo signo 8 de 8 años) es la **volatilidad del valor** y la **profundidad de la caída** (ret. 1–3 sesiones, distancia a máximos de 20 días y 52 semanas): cuanto más violenta la caída dentro de una tendencia alcista, mayor el rebote *medio*, pero el porcentaje de aciertos apenas cambia (62–66 %). La web muestra ahora, para cada alerta, esos motivos medibles con lo que pasó históricamente en su quintil.

**Se probó para encontrar más fiabilidad (y no mejoró de forma robusta):** tendencia a revertir de cada valor, régimen de reversión de todo el mercado, caída frente a su sector (GICS), crédito (HYG, diferencial HYG/LQD), tipos (TLT) y dólar (UUP), gradient boosting, mezcla lineal+boosting, reglas de confluencia entre modelos, reglas simples (volatilidad + caída), ordenar por retorno esperado (ridge) y quedarse solo con los patrones que ya habían funcionado hasta cada año. La mayoría empeoró la media o el Sharpe; las pocas que mejoraban una prueba no se repetían con salidas vecinas. El modelo actual sí aporta frente a elegir al azar entre los candidatos (*Equilibrado*: +0,58 % frente a +0,01 %; Sharpe de cartera 1,4 frente a −0,1), pero **ordenando por el tamaño del rebote, no por la probabilidad de ganar**.

**Adoptado: suelo de volatilidad (ATR ≥ 2,5 % del precio).** Los costes son fijos en % y en valores poco volátiles se comen el rebote típico: las operaciones elegidas con ATR < 2 % rindieron −0,09 % de media y las de ATR > 6 % +1,93 % (8 de 8 años positivas). Con el suelo, *Equilibrado* pasa de +0,58 % a +0,70 % (mejor en 6 de 8 años; Sharpe de cartera 1,37 → 1,68) y *Alta probabilidad* de +0,25 % a +0,36 % (7 de 8 años). Se probaron 6 suelos entre 2 % y 5 % y se eligió el más bajo que mejora los dos planes; más alto sube la media pero también la caída máxima.

## Medio plazo: ¿más tiempo, menos ruido, más fiabilidad? (`research.py midterm`, pestaña «Medio plazo»)

Se probaron, con parámetros fijados de antemano y netas de costes (señal con el cierre, ejecución al cierre del día siguiente), tres ideas de semanas–meses. **Historia larga 2005–2026 (21 años, incluye 2008), ETF:**

| | Anual | Volatilidad | Sharpe | Caída máx. | Peor mes |
|---|---|---|---|---|---|
| S&P 500 comprar y mantener | +10,9 % | 18,9 % | 0,64 | −55,2 % | −16,5 % |
| Tendencia SPY (sobre su media de 200) | +8,5 % | 11,6 % | 0,76 | −25,1 % | −8,2 % |
| Tendencia QQQ | +12,2 % | 15,6 % | 0,82 | −26,5 % | −12,2 % |
| Rotación de ETF por momentum (3 mejores, mensual) | +10,3 % | 16,9 % | 0,66 | −25,2 % | −10,1 % |

- **Más tiempo ≠ más rentabilidad, pero sí menos caídas**: la tendencia y la rotación recortan la caída máxima a la mitad (−55 % → −25 %) y el peor mes, con Sharpe parecido o algo mejor. En 2008: S&P 500 −36,8 %, tendencia +4,0 %, rotación +7,2 %. **No son infalibles**: en 2022 la tendencia llegó tarde (−20,1 %; la rotación, +10,2 %) y en 2019–2026 por separado comprar y mantener ganó más (+17,5 % frente a +12,6 %).
- **Robustez**: la rotación con ventanas de 3, 6 o 12 meses y 2, 3 o 5 ETF dio entre +10 % y +12 % anual y Sharpe 0,6–0,8 en los 21 años (sin depender de una elección afortunada); en la muestra corta de 10 años la dispersión fue mucho mayor (+6 % a +17,7 %), por eso hacen falta muchos años para juzgarlas.
- **Momentum de acciones (12-1, 30 valores del S&P 500)**: +30,5 % anual frente a +17,9 % del igual-ponderado del mismo universo, **pero** con volatilidad 28,8 %, caída máxima −39,6 % y Sharpe 1,07 frente a 0,94, y el universo (componentes actuales) tiene sesgo de supervivencia: no hay ventaja clara ajustada al riesgo. No está en producción.
- **Combinación con el sistema de rebotes** (correlación diaria ≈ +0,1, 2019–2026, curva a precio realizado): 50 % rebotes + 50 % tendencia SPY → +23,0 % anual, caída máx. −18,7 %, Sharpe 1,93, peor mes −14 % (rebotes solos: +32,9 %, −27,6 %, 1,74, −21 %). A partes iguales con las tres ideas de medio plazo: +17,5 %, −16,8 %, Sharpe 1,53. Menos rentabilidad, bastante menos riesgo.
- **Producción:** la pestaña «Medio plazo» muestra a diario la tendencia de SPY/QQQ/IWM/EFA y la cartera vigente de la rotación (se revisa a fin de mes). Es **informativa**: no genera alertas ni entra en el seguimiento. Conclusión honesta: sirve para reducir el riesgo de una cartera, no para ganar más.

## Ideas con EMA 34 / 89 / 200 y RSI (`research.py ema | ema2 | ema3`; no desplegado)

3,47 millones de señales (2016–2026, todo el universo), 31 setups × 6 salidas de seguimiento de tendencia. Cada setup se compara con **entradas al azar en el mismo régimen y con la misma salida**; solo cuenta lo que supera a ese azar.

- **Idea 1: el precio cruza de abajo arriba la EMA 34 con el RSI(14) en torno a 50 y subiendo → sin ventaja.** 134.735 señales, 41 % de aciertos, +0,91 % por operación con stop móvil 3×ATR, pero entrar al azar con la misma salida da +1,02 % (Δ −0,11 %, mejor solo en 4–5 de 10 años). El RSI no aporta nada (frente al cruce solo: −0,01 a −0,09) y su zona es irrelevante (+1,50 % a +1,75 % tanto con RSI 45–50 como con 60–70). Cartera 20 × 5 % (2017–2026): +6,6 % a +7,5 % anual con caída −28 %, frente a +15,3 % del S&P 500. Los espejos cortos pierden (−1,7 % a −2,6 % por operación). Con filtro de tendencia (sobre EMA 200, pila alcista) mejora un poco frente al azar de su régimen (+0,13 % a +0,22 %, 8–9 años de 10) pero rinde menos en términos absolutos: las mejores entradas llegan **después de un susto del mercado** (con el S&P bajo su media de 200: +2,34 % por operación; sobre ella: +0,60 %).
- **Idea 2: cruces entre EMAs.** 34↗89, 89↗200 y 34↗200 son entradas **tardías**: con salidas por precio rinden menos que el azar (Δ −0,4 % a −0,5 %). Lo único que resiste es la **triple alineación** (se completa 34 > 89 > 200) con **RSI ≥ 60 (momentum, no ≈ 50)** y salida al **cerrar bajo la EMA 89**: 11.903 señales, 35 % de aciertos, +2,45 % por operación, **+0,69 % frente al azar con el mismo RSI** (mejor en 10 de 10 años). Pero (a) el margen **se encoge fuera de muestra** (2016–20: +1,44 % → 2021–26: +0,22 %); (b) es un perfil de cola gruesa: ganancia media +19 %, pérdida media −6,7 %, mediana −3,3 %, el 5 % mejor aporta el 45 % de las ganancias; (c) pierde en 2018 y 2022, y 2020 (+11,8 % por operación) tira de la media; (d) con acciones de EE.UU. la cartera sale bien (+18,6 % anual, caída −17 %, Sharpe 1,47 frente a 0,63 del azar con el mismo filtro), pero ese universo tiene sesgo de supervivencia, y con **solo ETF** (menos sesgo) rinde +7,6 % con caída −11 %, es decir, menos que el S&P 500 aunque con menos riesgo.
- **Conclusión:** el RSI útil es el **alto** (confirma momentum), no el de 50; las EMAs sirven para **seguir la tendencia** (pocos aciertos, ganancias grandes), no para acertar más a menudo; y nada de esto supera de forma clara y robusta al S&P 500 una vez descontado el sesgo de supervivencia. No se despliega; la rotación/tendencia de la pestaña «Medio plazo» cumple un papel parecido con menos complejidad.

## Medio plazo con acciones: búsqueda masiva de «miles de formas» (`research.py stockmid | stockmid_long`; no desplegado)

2.730 variantes por universo (91 señales × 5 tamaños de cartera × 1 o 3 meses × 3 filtros), señal con el cierre de fin de mes, ejecución al cierre del día siguiente, costes 0,15 % ida y vuelta, rentabilidades con dividendos. Como con miles de pruebas siempre hay una ganadora por azar, se mide la **habilidad como exceso sobre comprar todo el universo a partes iguales** y se aplican cuatro controles: contraste del «mejor de N» (remuestreo por bloques sin habilidad), **selección honesta año a año** (se elige con los años anteriores), **persistencia** entre mitades y correlación de rangos (IC). El motor se validó con datos sintéticos: sin habilidad el contraste da p≈0,9; con habilidad inyectada, p≈0,003 (`tests/test_engine.py`).

| | 10 años (1.285 acciones, 2018–2026) | Historia larga (378 del S&P 500 actual, 2005–2026, incluye 2008) |
|---|---|---|
| Universo a partes iguales / S&P 500 | +16,0 % / +15,4 % anual | +14,7 % / +11,1 % anual (caídas −48 % / −53 %) |
| Variantes que superan al universo en exceso | 44 % (mediana −1,3 pts/año) | 27 % (mediana −2,6 pts/año) |
| «Mejor de 2.730»: p-valor (S&P 500 actual / 300 más negociadas) | **0,043** / 0,278 | **0,150** |
| Idea clásica momentum 12-1 (50 acciones, mensual) | +23,6 % anual, Sharpe 1,08 (+23,0 %, 0,95 en las 300 más negociadas) | **+15,3 %, Sharpe 0,86, caída −52 %** (exceso +0,7 pts) |
| Selección honesta (mejor Sharpe previo) | +39,4 % anual, Sharpe 1,21 (S&P 500: +14,9 %) / en las 300 más negociadas **+20,5 %, Sharpe 0,73** | +16,5 % anual, Sharpe 0,89, caída −29 % (S&P 500: +11,6 %, caída −49 %) |

- **Resultado: no hay una forma de elegir acciones a medio plazo que resista los controles.** En 21 años, la mejor de miles no se distingue del azar (p=0,15), el momentum clásico rinde como el universo (+0,7 pts) y las familias que sí dan algo (momentum +2,3 pts/año, distancia a la media +1,3, reversión +2,5) son pequeñas frente a la dispersión. La selección honesta mejora algo la rentabilidad y recorta la caída, y el recorte de caída (2008: +1 % frente a −35 %) apunta al filtro de mercado más que a elegir mejores acciones (no se comprobó variante a variante).
- **En la ventana de 10 años sale un efecto grande, pero no se confirma**: momentum, distancia a la media y reversión añaden +4,8 a +8,3 pts/año sobre el universo de componentes actuales del S&P 500, y las carteras concentradas (10 acciones) dan +35 % a +45 % anual con caída −21 % a −24 %. Pero (a) el contraste del «mejor de N» está en el límite (p=0,043) y **no se repite** con las 300 acciones más negociadas en cada fecha (p=0,278; la selección honesta rinde +20,5 % con Sharpe 0,73, peor que comprar el S&P 500, Sharpe 1,05); (b) la correlación de rangos de las señales con el mes siguiente es ≈ 0 (momentum 12-1: IC +0,001, t=0,1), es decir, **no ordena bien el universo entero; el efecto vive en la cola**; (c) el universo «S&P 500 actual» ya contiene a los que subieron lo suficiente como para entrar o permanecer (sesgo de supervivencia), y la selección honesta gana sobre todo en 2024–2026 (+44 %, +58 %, +59 %), años en los que probablemente mandaron pocas acciones muy grandes (hipótesis, no comprobada); (d) las variantes de baja volatilidad y de cercanía a máximos rinden **menos** que el universo (−6 y −4 pts/año).
- **ETF**: la correlación de rangos del momentum con la rentabilidad del mes siguiente es ≈ 0 (IC −0,025 a +0,055, |t| ≤ 1,5, 22 ETF, 104 meses): los «mejores ETF» del pasado no son los mejores del futuro, y por eso la rotación no bate al índice (solo reduce la caída máxima).
- **Conclusión:** no se despliega nada. Un resultado así solo sería creíble con constituyentes históricos del índice en cada fecha (incluidos los que salieron), que no hay en estos datos.

## ¿Es real el mecanismo de los rebotes? Prueba en 19 índices y hasta 65 años (`research.py longidx`)

El sistema se afinó con acciones de 2016–2026, con sesgo de supervivencia y en una sola época. Un índice no tiene sesgo de supervivencia y hay décadas distintas. Se aplicaron los **mismos 17 patrones de sobreventa en tendencia alcista** (sin modelo ni ranking) a 19 índices (S&P 500 desde 1961, Nasdaq, Russell 2000, Dow, FTSE, DAX, CAC, Nikkei, Hang Seng, Bovespa…) y se compararon con **entrar al azar en tendencia alcista con la misma salida** (coste 0,10 %, entrada a la apertura siguiente).

| Salida | Operaciones | Acierto | Media neta | Al azar | Ventaja | Índices donde supera al azar |
|---|---|---|---|---|---|---|
| RSI(2) > 70 («Equilibrado») | 8.641 | 68,0 % (al azar 61–67 %) | +0,11 % (t=3,6) | +0,04 % | **+0,07 pts** | **17 de 19** |
| Objetivo +1×ATR («Alta probabilidad») | 8.641 | 72,5 % | +0,04 % (t=1,2) | +0,01 % | +0,03 pts (no significativa) | 13 de 19 |

- **El mecanismo es real y generaliza**: con la salida por RSI supera al azar en 17 de 19 índices y en todas las décadas desde 1980 (+0,06 a +0,13 pts por operación, error típico 0,04–0,07). En los años 60 y 70 (solo S&P 500, 84 y 99 operaciones) sale negativo pero dentro del ruido. Y confirma lo visto con acciones: **el plan con salida por RSI tiene ventaja de pago; el de objetivo +1×ATR solo acierta más a menudo, sin ganar más que entrar al azar** (en acciones: alfa frente al S&P 500 −0,04 %).
- **Es pequeño en un índice**: +0,07 pts por operación sobre un ATR de ~1 %. Una cartera «solo S&P 500» con este sistema estuvo el 26 % del tiempo en mercado y rindió +1,0 % anual frente a +7,5 % de comprar y mantener (sin dividendos, 65 años): **solo con un índice no se gana dinero**. La ventaja necesita activos volátiles (ATR ≥ 2,5 %) donde el mismo rebote paga más en % — de ahí el suelo de volatilidad y que las acciones den +0,7 % por operación.
- **No protege en mercados bajistas fuertes**: 2000–02 −0,34 % por operación (al azar −0,33 %), 2008 −0,94 % (al azar −0,75 %). El filtro de tendencia (sobre la SMA200) llega tarde en las caídas rápidas.

## Sesgo de supervivencia: cuánto de la ventaja era real (`research.py pit | pit_mid | pit_deep`)

El universo de las pruebas es la lista **actual** del S&P 500/400/600. Eso cuela dos sesgos: acciones que entraron al índice *después* de subir mucho (se las contaba como si siempre hubieran estado) y las que salieron (caídas, quiebras, compras). Se reconstruyó la pertenencia en cada fecha con las tablas de cambios de Wikipedia (S&P 500: 412 cambios desde 1976, «Historical components of the S&P 500»; S&P 400: 623 desde 2012; S&P 600: 491 desde 2019) y se descargaron las retiradas que aún cotizan (**198 de 717 con datos, el 28 %**: las quebradas y compradas ya no existen en Yahoo; sesgo residual que no se puede cerrar con estos datos). El 15,8 % de las señales de acciones de EE.UU. eran de valores que aún no estaban en el índice.

**Sistema de rebotes, plan Equilibrado (walk-forward 2019–2026):**

| | Operaciones | Acierto | Media | PF | Alfa vs S&P 500 | Cartera 10×10 % |
|---|---|---|---|---|---|---|
| A · lista actual (lo que se mostraba) | 4.083 | 66,5 % | +0,74 % | 1,41 | +0,35 % (t=3,0) | +32,1 % anual · caída −29 % · Sharpe 1,7 |
| B · solo cuando ya era miembro | 4.006 | 66,2 % | +0,57 % | 1,34 | +0,18 % (t=1,7) | +22,5 % · −25 % · 1,4 |
| C · B + retiradas que aún cotizan | 3.852 | 66,8 % | +0,53 % | 1,30 | **+0,13 % (t=1,1)** | **+19,2 % · −26 % · 1,3** |

El S&P 500 comprar y mantener: +17,5 % anual, caída −33,7 %. Con la pertenencia histórica **la ventaja por operación baja un 28 %, el alfa deja de ser significativo y la rentabilidad de la cartera se queda en la del índice con algo menos de caída**. Las pequeñas capitalizaciones (S&P 600) pasan de +0,73 % a −0,48 % por operación y el sistema las excluye solo. El plan «Alta probabilidad» no tiene alfa en ninguna versión (−0,06 %, +0,01 %, −0,01 %).

**Prueba dura 2005–2026 (incluye 2008, 2011, 2015, 2018, 2020, 2022):** Equilibrado con pertenencia histórica (y sin operaciones de S&P 400/600 anteriores a su tabla): 8.117 operaciones, 66,0 % de acierto, **+0,38 % por operación**, alfa +0,12 % (t=1,9); cartera 10×10 %: **+11,1 % anual con caída −36 %** frente a +11,4 % y −52 % del S&P 500 (con la lista actual: +14,0 % y −34 %). Gana en 17 de 19 años (2008: +0,32 %; pierde en 2015: −0,16 % y 2018: −0,61 %).

**Medio plazo con acciones, con pertenencia histórica:** el «efecto grande» de 2018–2026 era casi todo hindsight. Con el S&P 500 de cada fecha: contraste del «mejor de 2.730» p=0,24 (con las 300 más negociadas, p=0,48); el 85 % de las variantes no bate al S&P 500; persistencia entre mitades +0,15; la selección honesta rinde +15,7 % (Sharpe 0,91) frente a +14,9 % (1,05) del índice; el momentum clásico 12-1 pasa de +29,5 % (Sharpe 1,20) a +15,7 % (0,82). **No hay habilidad demostrable en elegir acciones a medio plazo.**

**Qué se hizo en producción:** la validación semanal (`research.py final`) usa ahora esta pertenencia histórica y aborta si no puede reconstruirla (no sustituye el registro por uno con sesgo). La web lo indica en «Cómo se validó».

## 🚀 Pelotazos (experimental)

Pestaña aparte para operaciones de **cola gruesa**: acierta poco, pero a veces gana mucho. Patrones de fuerza (ruptura de máximos de 55 sesiones con volumen, líder de fuerza relativa que retrocede, contracción de volatilidad…) en EE.UU. grandes/medianas/pequeñas, las 3 mejores por día según la probabilidad de superar +12 %, y salida con **stop de seguimiento de 5×ATR** hasta 60 sesiones, sin objetivo.

- En el histórico: **≈48 % de operaciones ganadoras, la mediana pierde** y la media positiva depende de que ≈1 de cada 4 operaciones supere +20 %.
- **Las cifras están infladas**: sesgo de supervivencia (el universo son las empresas que existen hoy; las que quebraron no están), mercado alcista 2019–2026, y un benchmark aleatorio sobre la SMA200 con la misma salida ya da una media positiva. La web muestra la media con un descuento por quiebras (2/5/8 % de operaciones perdiendo −60 %), con costes dobles y el benchmark.
- Arriesga muy poco por operación (≈0,25–0,5 % del capital hasta el stop). El seguimiento de los pelotazos se mide aparte y no contamina las estadísticas globales. Se pausa solo si los últimos 12 meses no son rentables. No se opera al cierre (solo se validó entrando a la apertura).

## Estructura

```
index.html                 web (lee scanner/data/*.json)
scanner/
  universe.py / .json      universo (S&P 500/400/600, Nasdaq-100, ETF, Europa, cripto, …)
  data.py                  descarga Yahoo con reintentos, ajuste por dividendos, descarta la barra incompleta
  feats.py                 indicadores causales, amplitud de mercado, perfil de volumen móvil
  setups.py                patrones candidatos y filtros de operabilidad (liquidez, precio, ATR)
  simulate.py              simulador vectorizado de operaciones (reglas honestas)
  model.py                 modelo de probabilidad (NumPy, serializable a JSON)
  research.py              construye la tabla de eventos y VALIDA → model/validated.json
  lab.py                   laboratorio exploratorio (cómo se llegó al diseño)
  emastudy.py              estudio de ideas con EMA 34/89/200 + RSI (research.py ema | ema2 | ema3)
  midsig.py                señales de medio plazo (tendencia, rotación de ETF) → data/midterm.json
  study.py / midterm.py    estudios ampliados (calibración, motivos, medio plazo); se lanzan con research.py reasons | reasons2 | midterm
  scan.py                  robot diario → data/alerts.json, tracking.json, candles.json, …
                           (--mode preclose: aviso previo al cierre → alerts_pre.json)
  track.py / notify.py     seguimiento en vivo / avisos (Telegram, resumen en Actions)
  model/validated.json     modelo y estadísticas validadas (se regenera cada domingo)
tests/                     test_engine.py (causalidad, simulador, sin ventaja en datos aleatorios) · test_pipeline.py
.github/workflows/         scan.yml (L–V 22:30 UTC) · scan_preclose.yml (L–V 15:20 ET) · revalidate.yml (domingos) · research.yml (manual)
```

## Puesta en marcha

1. Fusiona la rama en `main` (los *cron* de Actions solo corren desde la rama por defecto).
2. En *Settings → Pages* publica `main` (carpeta raíz) para ver `index.html`; lee los datos de `raw.githubusercontent.com/.../main/scanner/data/`.
3. **Alertas al móvil (opcional)**: crea un bot con @BotFather y añade los *secrets* `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` en *Settings → Secrets and variables → Actions*. Sin ellos, el resumen de cada día aparece igualmente en la pestaña *Actions → Scan diario*.
4. Lanzar a mano: *Actions → Scan diario → Run workflow*; el aviso previo al cierre: *Actions → Scan previo al cierre → Run workflow* (marca «force» para probarlo fuera de horario); revalidar: *Actions → Revalidación del modelo*.

En local: `pip install numpy pandas` · `python tests/test_engine.py` · `python tests/test_pipeline.py` · `cd scanner && python scan.py` (o `python scan.py --mode preclose --force`).

## Avisos

Las probabilidades son estimaciones estadísticas, no garantías. Sesgo de supervivencia: el universo son los componentes actuales de los índices, lo que flatea los resultados históricos (la prueba real es el seguimiento en vivo). La ventaja puede desaparecer; el chequeo de salud y la revalidación semanal existen para detectarlo. No es asesoramiento financiero.
