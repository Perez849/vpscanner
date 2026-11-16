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

1. **Cada idea lleva su `tipo`** (`estructural`, `susto`, `obvio`, `panico`, `rival`, `caido`, `segundo`, `contagio`), y **cada descarte lleva también `tipo` y `valores`** (los tickers que se habrían comprado). Sin eso no se puede medir si descartar fue acertado.
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

| Tipo | Veredicto | Por qué |
|---|---|---|
| Hecho que cambia las cuentas durante trimestres (guerra, paz, vacuna, gobierno, ley, tecnología) | **Sí** | 19 casos: +7,7 % sobre el S&P 500 en 20 sesiones comprando al día siguiente; sube el 79 %. |
| Susto o anuncio de un día (huracán, atentado, arancel anunciado, tregua) | **No** | 18 casos: −4,7 % en 20 sesiones; solo sube el 28 %. |
| Beneficiario evidente de una materia prima o de los tipos, al día siguiente | **No** | El primer efecto lo recoge el precio el mismo día; después no se distingue del azar. |
| Pánico general (S&P 500 −2 % o −3 % en un día) | **Sí** | Nasdaq-100 +7,6 % en 60 sesiones tras un −3 % (gana el 73 %); +5,6 % tras un −2 %. |
| Competencia del que tropieza, cuando el problema es de la empresa y no del sector | **Sí** | 12 casos con nombre: +3,9 % en 60 sesiones (75 %); 1.846 caídas: +2,9 % (t 6,5). |
| Gigante que se hunde por un problema propio | **Con condiciones** | 21 caídas famosas: −6,7 % en 20 sesiones. Hace falta una prueba de negocio, no dejar pasar el tiempo. |
| Segundo efecto | **Con condiciones** | Funciona cuando el siguiente eslabón gana más de verdad (trigo → fertilizantes); no es regla general. |
| Buena noticia de una grande | **Con condiciones** | Solo mientras las competidoras sigan quietas (+1,6 % en 60 sesiones); si ya saltaron, tarde. |

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
