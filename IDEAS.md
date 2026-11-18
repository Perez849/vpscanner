# Ideas por coyuntura · protocolo del analista

La parte principal de **The Many Saints of Wall St** no es un cribador ni una regla de precio. Es un analista que, tres veces al día (9:00, 14:00 y 21:30, hora española, de lunes a viernes), lee lo que ha pasado en el mundo y razona **quién gana y quién pierde** con ello. Lo publica en la pestaña **Ideas** de la web.

## Qué hace cada sesión

1. **Leer el estado**: `data/ideas.json` (ideas vivas, con su disparador y su invalidación) y `data/ideas_playbook.json` (lecciones del entrenamiento).
2. **Hechos**: buscar qué ha pasado desde la última sesión (geopolítica, materias primas, bancos centrales, resultados, escándalos, regulación, catástrofes). Solo cuenta lo confirmado por más de una fuente; un rumor se apunta como rumor.
3. **Revisar las ideas vivas**: ¿ha saltado el disparador? → `abierta`, con `entrada` = fecha de hoy. ¿Ha ocurrido la invalidación o se ha cumplido el horizonte? → `cerrada`, con `salida`. Si no, una nota de seguimiento solo cuando haya algo nuevo que decir.
4. **Barrido amplio antes de elegir**: mirar como mínimo estos frentes, no solo el titular del día: geopolítica · materias primas · tipos y bancos centrales · resultados y guías de la última semana · fusiones y compras · regulación, sentencias y política · accidentes, averías y catástrofes · salud y farmacia · tecnología · consumo. La idea tiene que salir de un hecho **de las últimas una o dos sesiones**. Un tema que ya tuvo su momento no se propone ni se lista como descarte «porque llegamos tarde»: simplemente no está. Los ejemplos que el usuario pone para explicarse (petróleo y aerolíneas, Nvidia y su competencia) son eso, ejemplos, y no temas a seguir.
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

1. **Cada idea lleva su `tipo`** (`estructural`, `decision`, `susto`, `panico`, `norma`, `ataque`, `demanda`, `victima`, `opa`, `canario`, `jefe`, `farmaco`, `activista`, `huelga`, `split`, `indice`, `bajista`, `agricola`, `caido`, `rival`, `obvio`, `segundo`, `contagio`), y **cada descarte lleva también `tipo` y `valores`** (los tickers que se habrían comprado). Sin eso no se puede medir si descartar fue acertado.
2. `python -m vp.ideas` apunta cada idea dada y cada descarte en `data/ideas_casos.json` y, cuando pasan 5, 20 y 60 sesiones, mide lo que hicieron sus valores frente al SPY, igual que el entrenamiento (compra en la apertura siguiente). Los casos no se borran nunca.
3. **Cada lunes, en la edición de las 9:00**, se lee el resumen de `ideas_casos.json`. Si un tipo de idea que se viene dando no funciona, o uno que se viene descartando acaba subiendo, se reescribe la lección correspondiente en `data/ideas_playbook.json` (veredicto, texto y cifra) y se dice en la nota de la sesión.
4. **La primera edición de cada mes** se vuelve a ejecutar `python -m vp.ideas_train`, que incorpora las sesiones nuevas del mercado a los ensayos, y se añaden a `CASES` (en `vp/ideas_train.py`) los hechos con nombre propio del mes que hayan sido importantes, con su fecha, sus valores y su tipo. Después se revisan las lecciones con las cifras nuevas.

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

Sale de simular cada tipo de idea sobre hechos pasados (`python -m vp.ideas_train`; cifras en la pestaña **Entrenamiento**):

| Tipo (`tipo`) | Veredicto | Por qué |
|---|---|---|
| Hecho consumado que cambia el mundo: guerra, paz, pandemia, vacuna, vuelco electoral, fármaco que funciona (`estructural`) | **Sí** | 22 casos: +8,3 % sobre el S&P 500 en 20 sesiones comprando al día siguiente; sube el 73 %. |
| Decisión anunciada: banco central, ley firmada, plan de empresa (`decision`) | **No** | 29 casos: −3,4 % en 20 sesiones; solo sube el 31 %. Se compra antes y se vende cuando llega. |
| Susto o anuncio de un día: huracán, atentado, arancel, ataque, estímulo prometido (`susto`) | **No** | 51 casos: −2,5 % en 20 sesiones; sube el 35 %. El beneficiario pierde (−6,0 % a 60 sesiones); el castigado apenas gana. |
| Pánico general (`panico`) | **Sí, según el miedo** | Vale para un día de −2 %, una racha de tres días que sume −3 % o un hueco grande a la baja. Con el VIX previo ≥ 28: +10,0 % en 60 sesiones (78 %); racha con miedo alto: +8,8 %. Viniendo de calma: +2,3 %, menos que un tramo cualquiera. Da igual comprar al cierre que en la apertura siguiente. |
| Sector castigado por una norma o sanción, comprando el castigo (`norma`) | **Con condiciones** | 7 casos: +16,4 % en 60 sesiones (86 %), pero pocos y casi todos de chips en años de subida. |
| Gigante que cae desde cerca de máximos (`caido`) | **No** | 417 casos: −1,5 % en 20 sesiones; 42 caídas famosas: −5,6 %. Solo rebotan las ya muy castigadas, y esa cifra está inflada. |
| Competencia del que tropieza (`rival`) | **Solo si suben ese día** | Si las competidoras suben ≥ 2 % el mismo día del tropiezo (51 casos): +7,5 % en 60 sesiones (t 2,5). Sin eso, nada. |
| Ataque informático que obliga a gastar en protección (`ataque`) | **Con condiciones** | 6 casos: +6,4 % en 20 sesiones (83 %). Vale para ciberseguridad; el único caso de avería física (FAA, 2023) no funcionó. |
| Demanda antimonopolio contra un gigante, comprando al gigante (`demanda`) | **Con condiciones** | 5 casos: nada a 20 sesiones, +5,0 % a 60 (80 %). Pista con muy pocos casos. |
| Víctimas de la novedad de otro, comprando el castigo (`victima`) | **Con condiciones** | 7 casos: +6,4 % en 60 sesiones, sube el 57 %. |
| Noticias de la propia empresa: nuevo consejero delegado (`jefe`), fármaco aprobado (`farmaco`), entra un activista (`activista`), huelga o bloqueo (`huelga`) | **No** | Suben el primer día y lo devuelven: cambio de jefe, 12 casos, −9,0 % en 60 sesiones; fármaco aprobado, 6 casos, −5,3 % en 20. |
| Anuncio de split (`split`) o de entrada en el S&P 500 (`indice`) | **Con condiciones** | Empuje de unas semanas (split: +4,2 % en 20 sesiones, sube el 70 %), que se diluye a tres meses. El día en que entra de verdad en el índice, ya tarde: −0,9 % en 5 sesiones. |
| Informe de un fondo bajista, comprando el castigo (`bajista`) | **Con condiciones** | Sin patrón: unas veces tenía razón y otras no. Solo tras una respuesta seria (auditor, regulador, cuentas a tiempo). |
| Café, cacao, azúcar, ganado (`agricola`) | **No** | Nada que se distinga del azar; el cacao caro sigue castigando al chocolate después. |
| Tras una gran compra, las siguientes candidatas (`opa`) · arrastradas por el aviso de otra (`canario`) | **No** | 11 y 7 casos: nada a 20 sesiones y negativo a 60. |
| Beneficiario evidente de una materia prima o de los tipos (`obvio`) | **No** | Ni en un día ni en una semana: después del movimiento no queda nada. |
| Segundo efecto (`segundo`) | **Con condiciones** | Solo en pares concretos: trigo → fertilizantes, tipos a la baja → constructoras. |
| El líder se dispara (`contagio`) | **Solo al revés** | Las competidoras no suben por contagio. Pero si el mercado las castiga ese día (≥ 1 % abajo, 145 casos), lo recuperan: +4,2 % en 60 sesiones (t 2,6). |

Antes de dar una idea de tipo `estructural`, comprobar que no es en realidad una `decision`: si el hecho es que alguien **decide o anuncia** algo que se veía venir (un banco central, un parlamento, una empresa), es `decision` y no se da. Si es algo que **ha ocurrido** y nadie descontaba, es `estructural`.

La pregunta que separa un cambio duradero de un susto: ¿el motivo puede desaparecer con el siguiente titular, o cambia lo que esas empresas ganarán durante varios trimestres?

La **convicción** (1–5) dice cuánto encaja la idea con un tipo de veredicto «sí» y cuánto queda sin recoger en el precio.

## Ficha de una idea (`data/ideas.json`)

`id`, `fecha`, `estado` (`vigilancia` · `abierta` · `cerrada`), `conviccion`, `tipo` (uno de los de «Cómo aprende»), `caduca`, `en_contra`, `motivo_cierre`, `titulo`, `resumen`, `hecho`, `cadena` (pasos de la lógica), `valores` (`sym`, `nombre`, `papel`, `por_que`), `en_precio`, `disparador`, `invalidacion`, `no_hacer`, `horizonte`, `precedentes` (`caso`, `que_paso`), `ref` (precio de cada valor y del SPY el día que se dio), `entrada`, `salida`, `notas` (`fecha`, `texto`), `fuentes` (`t`, `u`).

Reglas del registro:

- Una idea **no se borra ni se retoca** aunque salga mal: solo cambia de estado y gana notas. (Única excepción hasta hoy: el 16-nov-2026 se retiraron a petición del usuario las dos ideas de arranque, que eran sus ejemplos y no temas del día.)
- La invalidación es un **hecho**, no un stop de precio. No hay tope de ganancia.
- El marcador (`data/ideas_track.json`) compara cada idea con el SPY: abierta, desde el cierre del día de entrada; en vigilancia, desde la referencia (solo orientativo).
- Cada descarte: `fecha`, `titulo`, `por_que`, `leccion`, `tipo` y `valores` (lista de tickers).
- No es asesoramiento personalizado: es un razonamiento público con su registro.
