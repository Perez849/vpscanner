# Ideas por coyuntura (parte 3 · sin reglas de precio)

No es un cribador. Es un analista que lee lo que ha pasado en el mundo y razona **quién gana y quién pierde** con ello: baja el petróleo por un acuerdo → aerolíneas y cruceros; se hunde el líder de un sector por un problema propio → sus competidores; etc.

## Método de cada sesión

1. **Hechos**: qué ha pasado de verdad desde la última sesión (geopolítica, materias primas, bancos centrales, resultados, escándalos, regulación). Solo hechos confirmados por más de una fuente; un rumor se apunta como rumor.
2. **Cadena lógica**: para cada hecho, primer efecto (el obvio) y segundo efecto (el que tarda más en verse). Se escribe en una frase que entendería cualquiera.
3. **¿Ya está en el precio?**: se mira cuánto se ha movido ya cada valor desde el hecho. Si el efecto obvio ya está recogido, la idea es el segundo efecto o no hay idea.
4. **Pocas**: como máximo 2 ideas nuevas por sesión, y lo normal es **ninguna**. Una idea necesita: hecho, lógica, valores, disparador (qué tiene que pasar para entrar), invalidación (qué hecho la rompe; no un stop de precio) y horizonte.
5. **Dejar correr**: sin tope de ganancia. Se sale cuando el hecho que la sostenía cambia o cuando se cumple el horizonte.

## Registro

Cada idea se guarda en `data/ideas.json` el día que se da, con el precio de ese día, y no se borra ni se retoca aunque salga mal. `python -m vp.ideas` la compara con el SPY desde su fecha. Es la única forma de saber si este sentido común vale dinero.

Estados: `vigilancia` (hay tesis pero falta el disparador), `abierta` (disparador cumplido; cuenta desde el cierre de ese día), `cerrada`.
