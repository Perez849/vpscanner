# VP Scanner v2

Escáner diario de ~2.000 activos (S&P 500/400/600, Nasdaq-100, ETF, Europa, cripto…) que avisa de **compras tras una caída dentro de una tendencia alcista** cuando un modelo estadístico, validado año a año, le da probabilidad de salir bien. Todo corre gratis en GitHub Actions; la web (`index.html`) solo lee los JSON.

> **Expectativas realistas.** La ventaja medida es modesta: **≈65 % de acierto y ≈ +0,4 % por operación neto de costes** (plan *Equilibrado*, positivo en los 8 años 2019–2026), o ≈70 % de acierto y ≈ +0,2 % (plan *Alta probabilidad*, marginal con costes dobles). No existe, en estos datos, una estrategia de ≥75 % de acierto *y* rentable: cuando el acierto sube, cada fallo (−4/−6 %) pesa más que cada acierto (+1/+1,5 %) y la ganancia media se evapora. La prueba definitiva es la pestaña **Seguimiento real**.

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
| Equilibrado | 65,8 % | +0,51 % | 1,32 | 64,9 % / +0,45 % | +0,32 % |
| Alta probabilidad | 70,1 % | +0,25 % | 1,16 | 69,2 % / +0,20 % | +0,06 % |

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
| Equilibrado (3 mejores/día) | ver web | | | | |
| Comprar y mantener el S&P 500 (2019–2026) | +17,5 % | 33,7 % | 0,94 | | |

- **Hallazgo incómodo**: en las mismas ventanas de cada operación, comprar el propio S&P 500 acertó casi igual (≈ 64 % frente a ≈ 66 %) y rindió +0,39 % frente a +0,51 %. El «alfa» sobre el índice del plan *Equilibrado* es ≈ +0,08 % (t ≈ 1) y el de *Alta probabilidad* ≈ −0,15 % (t ≈ −2): **gran parte de la ventaja viene de *cuándo* se compra (tras caídas en un mercado alcista), no de *qué* acción se elige**, y la parte de selección queda además inflada por el sesgo de supervivencia.
- La misma idea aplicada **solo a índices** (SPY/QQQ/IWM/DIA, sin modelo y sin sesgo de supervivencia): 513 señales en 10 años, **71 % de acierto, +0,30 % por operación** (t = 2,5; 8 de 10 años positivos), y con VIX tranquilo (z < 0,5) 72 % y +0,36 %. Una cartera «solo SPY» (≈ 15 operaciones al año) dio ≈ +6 % anual con caída máxima ≈ 7 % y 79 % de meses positivos: fiable pero de poca rentabilidad si no se apalanca. No está en producción; es una opción de bajo riesgo.
- Un 65 % de operaciones ganadoras **no** implica una curva suave: en la simulación solo ≈ 6 de cada 10 meses son positivos y los peores meses superan el −10 %. Usa posiciones pequeñas.

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
