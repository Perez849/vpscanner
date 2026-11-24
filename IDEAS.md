# Ideas por coyuntura · protocolo del analista

La parte principal de **The Many Saints of Wall St** no es un cribador ni una regla de precio. Es un analista que, tres veces al día (9:00, 14:00 y 21:30, hora española, de lunes a viernes), lee lo que ha pasado en el mundo y razona **quién gana y quién pierde** con ello. Lo publica en la pestaña **Ideas** de la web.

## Qué hace cada sesión

1. **Leer el estado**: `data/ideas.json` (ideas vivas, con su disparador y su invalidación) y `data/ideas_playbook.json` (lecciones del entrenamiento).
1b. **Alertas**: ejecutar `python -m vp.ideas alertas`. Mide las condiciones del día (miedo, caída del S&P 500, rachas, tipos, petróleo, grano, oro y acciones de la lista que se mueven un 10 % o más; y en Europa, el Euro Stoxx 50, el Ibex 35 y las grandes europeas que se mueven un 10 %) y dice qué lecciones se activan. **Si Wall Street está en plena sesión (la edición de las 21:30), usa el precio del momento**: la barra de hoy entra aunque no haya cerrado y las caídas o saltos del 10 % se miran ahora mismo, no en el escaneo de la madrugada. La primera línea dice con qué precios se ha medido. Se puede forzar con `alertas vivo` o `alertas cierre`. El miedo previo (VIX) siempre es el del cierre anterior a la sesión que se mide. **Cada alerta se atiende en la edición**: con una idea, un descarte o una línea en `sesion.cambios`. Las alertas no sustituyen al razonamiento; evitan que se olvide una lección.
2. **Hechos**: buscar qué ha pasado desde la última sesión (geopolítica, materias primas, bancos centrales, resultados, escándalos, regulación, catástrofes). Solo cuenta lo confirmado por más de una fuente; un rumor se apunta como rumor.
3. **Revisar las ideas vivas**: ¿ha saltado el disparador? → `abierta`, con `entrada` = fecha de hoy. ¿Ha ocurrido la invalidación o se ha cumplido el horizonte? → `cerrada`, con `salida`. Si no, una nota de seguimiento solo cuando haya algo nuevo que decir.
4. **Barrido amplio antes de elegir**: mirar como mínimo estos frentes, no solo el titular del día: geopolítica · materias primas · tipos y bancos centrales · resultados y guías de la última semana · fusiones y compras · regulación, sentencias y política · aranceles · accidentes, averías y catástrofes · salud y farmacia · tecnología · consumo · **Europa y España** (Euro Stoxx 50, Ibex 35, BCE, Bruselas, Gobierno español). La idea tiene que salir de un hecho **de las últimas una o dos sesiones**. Un tema que ya tuvo su momento no se propone ni se lista como descarte «porque llegamos tarde»: simplemente no está. Los ejemplos que el usuario pone para explicarse (petróleo y aerolíneas, Nvidia y su competencia) son eso, ejemplos, y no temas a seguir.
5. **Ideas nuevas**: para cada hecho, primer efecto (el obvio) y segundo efecto (el que tarda). Mirar cuánto se ha movido ya cada valor. Pasar el filtro de las lecciones (abajo). **Máximo 2 ideas nuevas por sesión; lo normal es ninguna.** No se rellena por rellenar.
6. **Descartes**: las ideas que parecían buenas y no pasan el filtro se publican en `descartes` con la lección que las tumba. La lista se rehace en cada sesión (solo las de hoy).
7. **Publicar**: reescribir `sesion` (`titular`: una frase corta de periódico sobre lo más importante del día, con la parte clave entre asteriscos; pulso de 2–4 frases y 3–6 hechos), añadir una línea a `sesiones`, ejecutar `python -m vp.ideas` (precios y marcador), `python tests/test_vp.py ideas` y subir los cambios.

## Lo que lleva cada edición además de las ideas

- **`sesion.hoy`**: la respuesta a «¿qué hago hoy?» en una frase. `accion` es `nada`, `vigilar`, `comprar` o `vender`; `texto` dice qué y cuándo («comprar Indra y Leidos al cierre»). Sale en una caja arriba del todo, así que tiene que poder leerse sola.
- **`sesion.cambios`**: de dos a cinco frases con lo que ha cambiado desde la edición anterior (un disparador que salta, un hecho nuevo, una idea que se rompe, un descarte nuevo). Si no ha cambiado nada, una sola frase que lo diga.
- **`nuevo: true`** en cada hecho, descarte, idea o nota que no estaba en la edición anterior, y se quita en la siguiente. La web lo resalta; nada se pliega ni se esconde.
- **`caduca`** (fecha) en toda idea en vigilancia: el día a partir del cual, si el disparador no ha saltado, la idea se cierra con `motivo_cierre: "no saltó"`. Se elige al darla, según el hecho (una votación, unos resultados, un plazo anunciado), y como norma no más de cuatro semanas. `python -m vp.ideas` avisa de las vencidas.
- **`motivo_cierre`** al cerrar una idea: `no saltó`, `se rompió` (ocurrió la invalidación), `horizonte cumplido` o el hecho concreto.
- **`en_contra`** en cada idea: el mejor argumento para no hacerla, escrito con el mismo cuidado que la tesis y sin rebajarlo. Si el argumento en contra convence más que la tesis, la idea no se da.
- **Hemeroteca**: `python -m vp.ideas` guarda cada edición entera en `data/ediciones/`. No se edita a mano.

## Las tres ediciones

| Edición | Para qué sirve |
|---|---|
| **9:00** | Qué ha pasado desde el cierre de ayer (Asia, apertura europea, noticias de la noche). |
| **14:00** | Preapertura de Wall Street: resultados y datos publicados antes de abrir. |
| **21:30** | Media hora antes del cierre: qué ha pasado **durante la sesión** y qué habría que comprar al cierre para adelantarse a la apertura de mañana. Es la edición en la que una idea puede pasar a «abierta» con entrada al cierre de hoy. |

## Cómo aprende

1. **Cada idea lleva su `tipo`**, que es el `id` de una lección de `data/ideas_playbook.json`: `estructural`, `decision`, `susto`, `panico`, `miedo`, `europa`, `norma`, `ataque`, `demanda`, `rival`, `contagio`, `segundo`, `caido`, `obvio`, `opa`, `fusion_rota`, `aranceles`, `bruselas`, `recompra`, `empresa`, `impulso_corto`, `bajista`, `agricola`. Se admiten también las variantes con nombre propio en la web (`jefe`, `farmaco`, `activista`, `huelga`, `split`, `indice`, `victima`, `canario`), que se miden con su lección (`empresa`, `impulso_corto`, `norma` u `opa`). **Cada descarte lleva también `tipo` y `valores`** (los tickers que se habrían comprado). Sin eso no se puede medir si descartar fue acertado. Los descartes solo de valores europeos se miden frente al Euro Stoxx 50; el resto, frente al SPY.
2. `python -m vp.ideas` apunta cada idea dada y cada descarte en `data/ideas_casos.json` y, cuando pasan 5, 20 y 60 sesiones, mide lo que hicieron sus valores frente al SPY (o frente al Euro Stoxx 50, si todos son europeos), igual que el entrenamiento (compra en la apertura siguiente). Los casos no se borran nunca.
3. **Cada lunes, en la edición de las 9:00**, se lee el resumen de `ideas_casos.json`. Si un tipo de idea que se viene dando no funciona, o uno que se viene descartando acaba subiendo, se reescribe la lección correspondiente en `data/ideas_playbook.json` (veredicto, texto y cifra) y se dice en la nota de la sesión.
4. **La primera edición de cada mes** se vuelve a ejecutar `python -m vp.ideas_train`, que incorpora las sesiones nuevas del mercado a los ensayos, y se añaden a `CASES` (en `vp/ideas_train.py`) los hechos con nombre propio del mes que hayan sido importantes, con su fecha (el primer día en que el mercado pudo reaccionar), sus valores y su letra de tipo. Los casos con valores europeos se miden solos frente a las grandes europeas. Después se revisan las lecciones con las cifras nuevas: si una cifra citada en una lección (o en la tabla de abajo) ha cambiado, se corrige en los dos sitios.
5. **La convicción también se mide.** `data/ideas_casos.json` guarda la convicción de cada idea dada y si llegó a comprarse, y la sección `calibracion` agrupa las compradas por convicción con su resultado a 20 y 60 sesiones (sale en la pestaña Entrenamiento, en «Casos propios»). En la revisión mensual: si las de convicción 3 no rinden más que las de 2, la convicción está mal puesta; se dice en la nota de la sesión y se corrige el criterio de abajo («La convicción…»).

## Cómo pasar las ediciones a otro chat

Todo lo que hace falta para seguir está en este repositorio: el protocolo (este fichero), las ideas vivas, las lecciones, los casos propios y la hemeroteca. Ningún chat guarda nada que no esté aquí, así que cambiar de chat no pierde nada.

Para pasar el relevo, en un chat nuevo abierto sobre este repositorio basta con escribir: **«Hazte cargo de las ediciones de Many Saints»**. El chat nuevo tiene que:

1. Leer este fichero entero y `data/ideas.json`.
2. Trabajar en su propia rama (`claude/…`). El flujo `ideas-sync` publica en `main` los datos de ideas que suba cualquier rama `claude/…`, así que no hay que tocar nada más.
3. Crear tres tareas programadas que se ejecuten en ese mismo chat, de lunes a viernes, a las 8:39, 13:34 y 21:09 de `Europe/Madrid`, con estos textos:
   - «Edición de la MAÑANA de The Many Saints of Wall St. Sigue el protocolo de IDEAS.md del repositorio vpscanner, en tu rama, después de traer lo último de main. Termina con un resumen breve en español.»
   - «Edición de PREAPERTURA de The Many Saints of Wall St. …» (igual)
   - «Edición de CIERRE de The Many Saints of Wall St. …» (igual; además dice si hay algo que comprar al cierre)
4. Borrar las tres tareas antiguas (las que se llamen «Many Saints …» y pertenezcan al chat anterior), para que no haya dos chats escribiendo a la vez.
5. Hacer una edición de prueba y comprobar que aparece en la web.

Antes de cada edición, el chat que la hace trae lo último: `git fetch origin main && git merge origin/main`.

## Quién hace la lectura

Las lecturas las hace el modelo de IA que esté sirviendo la conversación en ese momento. Este protocolo está escrito para que cualquier modelo capaz lo siga igual: mismos pasos, mismo filtro, misma ficha. Si la lectura no puede hacerse con el rigor que pide (sin acceso a noticias o a precios, o sin poder contrastar los hechos), **no se inventa nada**: se deja la edición anterior y se añade una línea a `sesiones` diciendo que esa edición no se pudo hacer y por qué.

## El filtro: qué ideas se dan y cuáles no

Sale de simular cada tipo de idea sobre hechos pasados desde 2000, en EE. UU. y en Europa (`python -m vp.ideas_train`; cifras en la pestaña **Entrenamiento**):

| Tipo (`tipo`) | Veredicto | Por qué (cifras completas en la lección del mismo nombre) |
|---|---|---|
| Hecho consumado que cambia el mundo: guerra, paz, pandemia, vacuna, vuelco electoral, fármaco que funciona (`estructural`) | **Sí** | 28 casos: +6,1 % sobre el S&P 500 en 20 sesiones comprando al día siguiente; sube el 61 %. |
| Decisión anunciada: banco central, ley firmada, plan de empresa (`decision`) | **No** | 36 casos: −2,4 % en 20 sesiones; solo sube el 36 %. Se compra antes y se vende cuando llega. (En Europa, los rescates de los bancos centrales sí dieron: pista con cuatro casos, lección `europa`.) |
| Susto o anuncio de un día: huracán, atentado, ataque, estímulo prometido (`susto`) | **No** | 65 casos: −0,7 % en 20 sesiones; sube el 45 %. El beneficiario pierde (−2,5 % a 60 sesiones); el castigado apenas gana (+3,4 %). |
| Pánico general (`panico`) | **Solo con el miedo alto** | Con el VIX previo ≥ 28, tras un día de −2 % o una racha de tres días que sume −3 %: +10,7 % y +8,0 % en 60 sesiones. Sin ese miedo, nada: cualquier día de −2 % da +2,3 %, igual que un tramo cualquiera (+2,6 %) desde 2000. Da igual comprar al cierre que en la apertura siguiente. |
| El miedo alto como multiplicador (`miedo`) | **Con condiciones** | Ver «Antes de descartar, cruzar con el miedo», abajo. |
| Europa y España (`europa`) | **Sí, solo el pánico con miedo alto** | Euro Stoxx 50 o Ibex 35 tras un −2 % con el VIX previo ≥ 28: +5,6 % y +5,1 % en 60 sesiones sobre lo normal, en las dos mitades del periodo; también tras un −2 % de Wall Street, comprando en la apertura europea. La grande europea que se hunde un 10 % con miedo alto, como en EE. UU. (+18,0 %, sin problema propio). Los hechos consumados también (Ucrania → defensa europea), y la racha de tres días con miedo alto. En calma, los sustos, el gigante caído, lo evidente (petróleo, gas, euro), las competidoras y el contagio: nada, como en EE. UU. El contagio desde Wall Street no se compra (chips de EE. UU. hundidos → ASML e Infineon, −6,6 % en 60 sesiones). Son unas 12–16 crisis distintas. |
| Sector castigado por una norma o sanción, comprando el castigo (`norma`) | **Con condiciones** | 7 casos: +16,4 % en 60 sesiones (86 %), pero pocos y casi todos de chips en años de subida. |
| Gigante que cae desde cerca de máximos (`caido`) | **No** | 581 casos: −1,6 % en 20 sesiones; 54 caídas famosas: −6,0 %. Solo rebotan las ya muy castigadas, y esa cifra está inflada. |
| Competencia del que tropieza (`rival`) | **Solo si suben ese día** | Si las competidoras suben ≥ 2 % el mismo día del tropiezo (113 casos): +2,7 % en 20 sesiones (t 2,1). Sin eso, nada. |
| Ataque informático que obliga a gastar en protección (`ataque`) | **Con condiciones** | 10 casos: +5,5 % en 20 sesiones (80 %). Vale para ciberseguridad; el único caso de avería física (FAA, 2023) no funcionó. |
| Demanda antimonopolio contra un gigante, comprando al gigante (`demanda`) | **Con condiciones** | 8 casos: −1,2 % a 20 sesiones, +3,6 % a 60. Pista con muy pocos casos. |
| Víctimas de la novedad de otro, comprando el castigo (`victima`, se mide con `norma`) | **Con condiciones** | 12 casos: +3,4 % en 60 sesiones, sube el 58 % a 20. |
| Noticias de la propia empresa (`empresa`): nuevo consejero delegado (`jefe`), fármaco aprobado (`farmaco`), entra un activista (`activista`), huelga o bloqueo (`huelga`), unos resultados | **No** | Suben el primer día y lo devuelven: cambio de jefe, 15 casos, −7,2 % en 60 sesiones; fármaco aprobado, 6 casos, −5,3 % en 20. |
| Anuncio de split (`split`) o de entrada en el S&P 500 (`indice`), los dos con la lección `impulso_corto` | **Con condiciones** | Empuje de unas semanas (split: +3,2 % en 20 sesiones, sube el 67 %), que se diluye a tres meses. El día en que entra de verdad en el índice, ya tarde: −0,9 % en 5 sesiones. |
| Informe de un fondo bajista, comprando el castigo (`bajista`) | **Con condiciones** | Sin patrón: unas veces tenía razón y otras no. Solo tras una respuesta seria (auditor, regulador, cuentas a tiempo). |
| Café, cacao, azúcar, ganado (`agricola`) | **No** | Nada que se distinga del azar; el cacao caro sigue castigando al chocolate después. |
| Tras una gran compra, las siguientes candidatas (`opa`) · arrastradas por el aviso de otra (`canario`, se mide con `opa`) | **No** | 16 y 11 casos: las candidatas, +1,1 % a 60 sesiones; las arrastradas, −7,9 %. |
| Fusión u OPA que se rompe, comprando a la que iba a ser comprada (`fusion_rota`) | **No** | 6 casos: −13,3 % el día de la ruptura y −2,0 % en las 20 sesiones siguientes. Unas veces valía más sola y otras el comprador sabía algo. |
| Aranceles anunciados, comprando a los castigados (`aranceles`) | **No** | 4 casos: +1,3 % en 20 sesiones, casi todo por la pausa del «Día de la Liberación». Depende de una pausa o un acuerdo que nadie sabe el día del anuncio. Si llegan en pleno pánico, manda `panico`. |
| Multa o norma de Bruselas contra un gigante (`bruselas`) | **No** | 9 casos: +0,3 % el día de la multa. No hay castigo que comprar ni motivo para vender. |
| Gran recompra de acciones (`recompra`) | **No, sola** | 8 casos, casi todos junto a los resultados; la media la sostienen dos. Suma si acompaña a un hecho que ya es idea. |
| Beneficiario evidente de una materia prima o de los tipos (`obvio`) | **No** | Ni en un día ni en una semana: después del movimiento no queda nada. Tampoco en Europa (IAG tras un −4 % del petróleo: −0,3 %). |
| Segundo efecto (`segundo`) | **Con condiciones** | Solo en pares concretos: trigo → fertilizantes, tipos a la baja → constructoras. |
| El líder se dispara (`contagio`) | **Solo al revés** | Las competidoras no suben por contagio. Pero si el mercado las castiga ese día (≥ 1 % abajo, 307 casos), lo recuperan: +1,9 % en 20 sesiones (t 2,6) y +2,6 % en 60. |

Antes de dar una idea de tipo `estructural`, comprobar que no es en realidad una `decision`: si el hecho es que alguien **decide o anuncia** algo que se veía venir (un banco central, un parlamento, una empresa), es `decision` y no se da. Si es algo que **ha ocurrido** y nadie descontaba, es `estructural`.

**Antes de descartar, cruzar con el miedo.** Al combinar lecciones (validando cada combinación por separado en 2000–2013 y en 2014–2026) sale un multiplicador común: el índice del miedo (VIX). Con el VIX en 28 o más:

- Pánico del mercado: comprar **semiconductores (SMH)** ha dado más que el Nasdaq entero (+10,4 % frente a +8,9 % en 60 sesiones).
- Una acción grande que cae un 10 % en un día de tormenta, **sin un problema propio** detrás: +18,6 % en 60 sesiones. En calma, lo contrario (−1,3 %). En Europa se repite: +18,0 % (87 casos), aunque después de 2014 solo hay 16.
- Competidoras castigadas el día en que el líder se dispara: +9,0 %.
- El castigado de un susto: +4,8 % (16 casos).

Así que un descarte de tipo `caido`, `susto` (comprar al castigado) o `contagio` se revisa si ese día el VIX está en 28 o más. Si la caída es por un escándalo o por malos resultados propios, sigue siendo «no» aunque haya tormenta.

La pregunta que separa un cambio duradero de un susto: ¿el motivo puede desaparecer con el siguiente titular, o cambia lo que esas empresas ganarán durante varios trimestres?

La **convicción** (1–5) dice cuánto encaja la idea con un tipo de veredicto «sí» y cuánto queda sin recoger en el precio. Se mide (ver «Cómo aprende», punto 5).

## Cuándo salir

Sale de seguir día a día cada tipo de idea que funciona durante sus 60 sesiones (lección `salidas`, grupo «Salidas» de los ensayos):

- **Las ideas buenas caen por el camino antes de dar fruto.** Comprar el Nasdaq en un pánico con miedo alto ha caído de mediana un 4 % antes de subir (una de cada cinco veces, más de un 13 %); los semiconductores, un 5 % (más de un 16 %); la grande castigada en plena tormenta, un 18 %.
- **Un stop de precio las estropea:** con un stop al −10 %, el pánico con miedo alto pasa de +10,7 % a +7,4 %, y la grande castigada en plena tormenta, de +23,6 % a +7,1 %. Por eso **se sale por un hecho (la invalidación) o por el horizonte, no por el precio**.
- **El horizonte importa y depende del tipo** (grupo «Horizontes» de los ensayos, medido de 5 a 120 sesiones). No se recogen beneficios por un buen primer día.

  | Tipo | Horizonte | Por qué |
  |---|---|---|
  | `panico`, `miedo` (y la grande europea castigada con miedo) | **90 sesiones** (unos 4 meses) | El Nasdaq pasa de +8,2 % a 60 a +11,4 % a 90, y es donde más se distingue del azar; la grande castigada, de +18,6 % a +21,1 %. |
  | `rival` (competidoras con trasvase) | **90 sesiones** | +3,6 % a 60 y +6,9 % a 90. |
  | `norma` | **60 sesiones** | Es donde mejor se distingue del azar (+16,4 %, siete casos). |
  | `europa`: Ibex 35 tras un pánico | **60 sesiones** | +4,7 % a 60; después devuelve algo. El Euro Stoxx 50 da lo más fiable la primera semana y sigue despacio hasta 90. |
  | `estructural` | **20 a 40 sesiones** | +6,1 % a 20 y +8,7 % a 40; a 60 ya no añade. |
  | `impulso_corto` (split, entrada en el índice) | **De 2 a 4 semanas** | El split da lo suyo en 10–20 sesiones y lo devuelve a 40. |
- **La señal de alarma es la caída en calma:** de las grandes castigadas que caen otro 10 % después de comprarlas, en calma solo acaba en positivo una de cada cuatro; en plena tormenta, la mitad.

Cada idea viva lleva en **`camino`** la caída que es normal esperar según su tipo (de la lección `salidas`), para que una caída de ese tamaño no se confunda con una idea rota.

## Ficha de una idea (`data/ideas.json`)

`id`, `fecha`, `estado` (`vigilancia` · `abierta` · `cerrada`), `conviccion`, `tipo` (uno de los de «Cómo aprende»), `caduca`, `en_contra`, `camino` (la caída normal por el camino y cuándo se sale), `indice` (solo si no es el SPY: `^STOXX50E` o `^IBEX` para las ideas europeas), `motivo_cierre`, `titulo`, `resumen`, `hecho`, `cadena` (pasos de la lógica), `valores` (`sym`, `nombre`, `papel`, `por_que`), `en_precio`, `disparador`, `invalidacion`, `no_hacer`, `horizonte`, `precedentes` (`caso`, `que_paso`), `ref` (precio de cada valor y de su índice el día que se dio), `entrada`, `salida`, `notas` (`fecha`, `texto`), `fuentes` (`t`, `u`).

Reglas del registro:

- Una idea **no se borra ni se retoca** aunque salga mal: solo cambia de estado y gana notas. (Excepciones: el 16-nov-2026 se retiraron a petición del usuario las dos ideas de arranque, que eran sus ejemplos y no temas del día; y el horizonte puede cambiar si el entrenamiento demuestra otro mejor, siempre con una nota que lo diga y sin tocar el marcador.)
- La invalidación es un **hecho**, no un stop de precio. No hay tope de ganancia.
- El marcador (`data/ideas_track.json`) compara cada idea con su índice (el SPY, o el Euro Stoxx 50 o el Ibex 35 si la ficha lleva `indice`): abierta, desde el cierre del día de entrada; en vigilancia, desde la referencia (solo orientativo).
- Cada descarte: `fecha`, `titulo`, `por_que`, `leccion`, `tipo` y `valores` (lista de tickers).
- No es asesoramiento personalizado: es un razonamiento público con su registro.
