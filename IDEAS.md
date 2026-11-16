# Ideas por coyuntura · protocolo del analista

La parte principal de VP Scanner no es un cribador ni una regla de precio. Es un analista que, dos veces al día (9:00 y 14:00, hora española, de lunes a viernes), lee lo que ha pasado en el mundo y razona **quién gana y quién pierde** con ello. Lo publica en la pestaña **Ideas** de la web.

## Qué hace cada sesión

1. **Leer el estado**: `data/ideas.json` (ideas vivas, con su disparador y su invalidación) y `data/ideas_playbook.json` (lecciones del entrenamiento).
2. **Hechos**: buscar qué ha pasado desde la última sesión (geopolítica, materias primas, bancos centrales, resultados, escándalos, regulación, catástrofes). Solo cuenta lo confirmado por más de una fuente; un rumor se apunta como rumor.
3. **Revisar las ideas vivas**: ¿ha saltado el disparador? → `abierta`, con `entrada` = fecha de hoy. ¿Ha ocurrido la invalidación o se ha cumplido el horizonte? → `cerrada`, con `salida`. Si no, una nota de seguimiento solo cuando haya algo nuevo que decir.
4. **Barrido amplio antes de elegir**: mirar como mínimo estos frentes, no solo el titular del día: geopolítica · materias primas · tipos y bancos centrales · resultados y guías de la última semana · fusiones y compras · regulación, sentencias y política · accidentes, averías y catástrofes · salud y farmacia · tecnología · consumo. La idea tiene que salir de un hecho **de las últimas una o dos sesiones**. Un tema que ya tuvo su momento no se propone ni se lista como descarte «porque llegamos tarde»: simplemente no está. Los ejemplos que el usuario pone para explicarse (petróleo y aerolíneas, Nvidia y su competencia) son eso, ejemplos, y no temas a seguir.
5. **Ideas nuevas**: para cada hecho, primer efecto (el obvio) y segundo efecto (el que tarda). Mirar cuánto se ha movido ya cada valor. Pasar el filtro de las lecciones (abajo). **Máximo 2 ideas nuevas por sesión; lo normal es ninguna.** No se rellena por rellenar.
6. **Descartes**: las ideas que parecían buenas y no pasan el filtro se publican en `descartes` con la lección que las tumba. La lista se rehace en cada sesión (solo las de hoy).
7. **Publicar**: reescribir `sesion` (`titular`: una frase corta de periódico sobre lo más importante del día, con la parte clave entre asteriscos; pulso de 2–4 frases y 3–6 hechos), añadir una línea a `sesiones`, ejecutar `python -m vp.ideas` (precios y marcador), `python tests/test_vp.py ideas` y subir los cambios.

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

`id`, `fecha`, `estado` (`vigilancia` · `abierta` · `cerrada`), `conviccion`, `tipo`, `titulo`, `resumen`, `hecho`, `cadena` (pasos de la lógica), `valores` (`sym`, `nombre`, `papel`, `por_que`), `en_precio`, `disparador`, `invalidacion`, `no_hacer`, `horizonte`, `precedentes` (`caso`, `que_paso`), `ref` (precio de cada valor y del SPY el día que se dio), `entrada`, `salida`, `notas` (`fecha`, `texto`), `fuentes` (`t`, `u`).

Reglas del registro:

- Una idea **no se borra ni se retoca** aunque salga mal: solo cambia de estado y gana notas. (Única excepción hasta hoy: el 16-nov-2026 se retiraron a petición del usuario las dos ideas de arranque, que eran sus ejemplos y no temas del día.)
- La invalidación es un **hecho**, no un stop de precio. No hay tope de ganancia.
- El marcador (`data/ideas_track.json`) compara cada idea con el SPY: abierta, desde el cierre del día de entrada; en vigilancia, desde la referencia (solo orientativo).
- No es asesoramiento personalizado: es un razonamiento público con su registro.
