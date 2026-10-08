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
3. **Política**: cada día solo se alertan las **5 mejores** cuya probabilidad supera la tasa base de acierto. Los grupos de activos sin evidencia (p. ej. valores temáticos) se excluyen solos.
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
  track.py / notify.py     seguimiento en vivo / avisos (Telegram, resumen en Actions)
  model/validated.json     modelo y estadísticas validadas (se regenera cada domingo)
tests/                     test_engine.py (causalidad, simulador, sin ventaja en datos aleatorios) · test_pipeline.py
.github/workflows/         scan.yml (L–V 22:30 UTC) · revalidate.yml (domingos) · research.yml (manual)
```

## Puesta en marcha

1. Fusiona la rama en `main` (los *cron* de Actions solo corren desde la rama por defecto).
2. En *Settings → Pages* publica `main` (carpeta raíz) para ver `index.html`; lee los datos de `raw.githubusercontent.com/.../main/scanner/data/`.
3. **Alertas al móvil (opcional)**: crea un bot con @BotFather y añade los *secrets* `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` en *Settings → Secrets and variables → Actions*. Sin ellos, el resumen de cada día aparece igualmente en la pestaña *Actions → Scan diario*.
4. Lanzar a mano: *Actions → Scan diario → Run workflow*; revalidar: *Actions → Revalidación del modelo*.

En local: `pip install numpy pandas` · `python tests/test_engine.py` · `python tests/test_pipeline.py` · `cd scanner && python scan.py`.

## Avisos

Las probabilidades son estimaciones estadísticas, no garantías. Sesgo de supervivencia: el universo son los componentes actuales de los índices, lo que flatea los resultados históricos (la prueba real es el seguimiento en vivo). La ventaja puede desaparecer; el chequeo de salud y la revalidación semanal existen para detectarlo. No es asesoramiento financiero.
