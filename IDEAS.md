# Ideas por coyuntura · protocolo del analista

La parte principal de VP Scanner no es un cribador ni una regla de precio. Es un analista que, dos veces al día (9:00 y 14:00, hora española, de lunes a viernes), lee lo que ha pasado en el mundo y razona **quién gana y quién pierde** con ello. Lo publica en la pestaña **Ideas** de la web.

## Qué hace cada sesión

1. **Leer el estado**: `data/ideas.json` (ideas vivas, con su disparador y su invalidación) y `data/ideas_playbook.json` (lecciones del entrenamiento).
2. **Hechos**: buscar qué ha pasado desde la última sesión (geopolítica, materias primas, bancos centrales, resultados, escándalos, regulación, catástrofes). Solo cuenta lo confirmado por más de una fuente; un rumor se apunta como rumor.
3. **Revisar las ideas vivas**: ¿ha saltado el disparador? → `abierta`, con `entrada` = fecha de hoy. ¿Ha ocurrido la invalidación o se ha cumplido el horizonte? → `cerrada`, con `salida`. Si no, una nota de seguimiento solo cuando haya algo nuevo que decir.
4. **Ideas nuevas**: para cada hecho, primer efecto (el obvio) y segundo efecto (el que tarda). Mirar cuánto se ha movido ya cada valor. Pasar el filtro de las lecciones (abajo). **Máximo 2 ideas nuevas por sesión; lo normal es ninguna.** No se rellena por rellenar.
5. **Descartes**: las ideas que parecían buenas y no pasan el filtro se publican en `descartes` con la lección que las tumba. La lista se rehace en cada sesión (solo las de hoy).
6. **Publicar**: reescribir `sesion` (pulso de 2–4 frases y 3–6 hechos), añadir una línea a `sesiones`, ejecutar `python -m vp.ideas` (precios y marcador), `python tests/test_vp.py ideas` y subir los cambios.

## El filtro: qué ideas se dan y cuáles no

Sale de simular cada tipo de idea sobre hechos pasados (`python -m vp.ideas_train`; cifras en la pestaña **Entrenamiento**):

| Tipo | Veredicto | Por qué |
|---|---|---|
| Beneficiario evidente de una materia prima o de los tipos, al día siguiente | **No** | El primer efecto lo recoge el precio el mismo día; después no queda nada (petróleo −4 % → aerolíneas: +0,1 % en 5 sesiones, 99 casos). |
| Hecho que cambia las cuentas durante trimestres (guerra, paz, vacuna, cambio de gobierno, clientes que cambian de proveedor) | **Sí** | Los beneficiarios siguieron subiendo semanas después del día de la noticia. Los sustos de un día se dieron la vuelta. |
| Segundo efecto con poder de precio | **Sí** | Petróleo +4 % → refino: +4,7 % en 60 sesiones (t 2,6), frente a +0,5 % en 20 de las petroleras. |
| Pánico general (S&P 500 −3 % en un día) | **Sí** | Nasdaq-100 +9,1 % en 60 sesiones (t 4,8), el doble de lo normal. |
| Gigante que se hunde por un problema propio | **Esperar a la prueba** | Rebota de media, pero gana menos de la mitad de las veces y las primeras semanas suelen ser malas. |
| Competencia del que tropieza | **Solo el primer día y si sube** | Una semana después es llegar tarde. |
| Buena noticia de una grande que aún no se ha contagiado al sector | **Sí (pequeña)** | +1,3 % en 60 sesiones (t 3,3). |

La **convicción** (1–5) dice cuánto encaja la idea con un tipo de veredicto «sí» y cuánto queda sin recoger en el precio.

## Ficha de una idea (`data/ideas.json`)

`id`, `fecha`, `estado` (`vigilancia` · `abierta` · `cerrada`), `conviccion`, `tipo`, `titulo`, `resumen`, `hecho`, `cadena` (pasos de la lógica), `valores` (`sym`, `nombre`, `papel`, `por_que`), `en_precio`, `disparador`, `invalidacion`, `no_hacer`, `horizonte`, `precedentes` (`caso`, `que_paso`), `ref` (precio de cada valor y del SPY el día que se dio), `entrada`, `salida`, `notas` (`fecha`, `texto`), `fuentes` (`t`, `u`).

Reglas del registro:

- Una idea **no se borra ni se retoca** aunque salga mal: solo cambia de estado y gana notas.
- La invalidación es un **hecho**, no un stop de precio. No hay tope de ganancia.
- El marcador (`data/ideas_track.json`) compara cada idea con el SPY: abierta, desde el cierre del día de entrada; en vigilancia, desde la referencia (solo orientativo).
- No es asesoramiento personalizado: es un razonamiento público con su registro.
