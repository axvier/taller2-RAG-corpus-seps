Python 3.12.x (el entorno con 3.14 falló por falta de wheels de scikit-network)

# Parte 0.a:

Salida:
embeddings: local · sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 · tope 128 tokens · Qdrant :memory:
AVISO ingesta: instructivo_escaneado.pdf: 0 caracteres útiles en 2 página(s). ¿Escaneado o fotografiado? NO se indexa: sin texto no hay nada que recuperar (OCR aparte).
ingesta: nimbus_gastos.md: 1 página(s), 294 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_remoto.md: 1 página(s), 296 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_vacaciones.md: 1 página(s), 363 caracteres útiles → 1 fragmento(s)
indexados 3 fragmentos de 3 documento(s)
  0.681  nimbus_vacaciones.md / nimbus_vacaciones-0000  # Política de vacaciones de NimbusSoft Los empleados a tiempocompleto acumulan …
  0.502  nimbus_remoto.md / nimbus_remoto-0000  # Política de trabajo remoto de NimbusSoft El trabajo remoto está permitido hast…
  0.485  nimbus_gastos.md / nimbus_gastos-0000  # Política de reembolso de gastos de NimbusSoft Los gastos de viaje se reembolsa…

[modo inspección (sin clave ni Ollama)] abstuvo=None
[modo inspección: no hay clave de API ni Ollama; este es el prompt]
Responde usando solo el contexto recuperado.
Si el contexto no contiene la respuesta, di exactamente: "El corpus no contiene información suficiente."

Contexto:
[1] nimbus_vacaciones.md / nimbus_vacaciones-0000
# Política de vacaciones de NimbusSoft Los empleados a tiempo completo acumulan 1.5 días de vacaciones por mes trabajado, hasta un máximo de 18 días por año. Las vacaciones deben solicitarse con al menos 15 días de anticipación a través del portal interno. Los días no utilizados pueden transferirse al año siguiente hasta un máximo de 5 días. Durante el primer año, los días solo pueden tomarse después de superar el período de prueba de 3 meses.

[2] nimbus_remoto.md / nimbus_remoto-0000
# Política de trabajo remoto de NimbusSoft El trabajo remoto está permitido hasta 3 días por semana para todos los roles excepto soporte de infraestructura on-site. Los días remotos se coordinan con el líder de equipo. Para trabajar desde el exterior del país se requiere aprobación de Recursos Humanos con 30 días de anticipación y un máximo de 60 díaspor año.

[3] nimbus_gastos.md / nimbus_gastos-0000
# Política de reembolso 

# Parte 0.b: Truncamiento a 128
# Salida:
=== 1) Ingesta de ejemplos/ pidiendo chunk_tokens=900 ===
AVISO ingesta: chunk_tokens=900 supera el tope del modelo (128 tokens): lo que pase de 126 tokens no llega al índice —el MiniLM lo trunca en silencio; la H200 rechaza la petición—.
AVISO ingesta: instructivo_escaneado.pdf: 0 caracteres útiles en 2 página(s). ¿Escaneado o fotografiado? NO se indexa: sin texto no hay nada que recuperar (OCR aparte).
ingesta: nimbus_gastos.md: 1 página(s), 294 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_remoto.md: 1 página(s), 296 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_vacaciones.md: 1 página(s), 363 caracteres útiles → 1 fragmento(s)

=== 2) Texto de ~900 palabras: entero vs. recortado al tope ===
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (1605 > 128). Running this sequence through the model will result in indexing errors
archivo fuente:          01.-Manual-de-Depositos_V10.0_17jul2026.pdf
palabras del texto:      900
tokens del texto:        1605
tope del modelo:         128 (max_seq_length; 126 de contenido)
fracción que cabe:       7.9%
coseno(entero, cortado): 1.0000

# Parte 0C
# Salida:
AVISO ingesta: instructivo_escaneado.pdf: 0 caracteres útiles en 2 página(s). ¿Escaneado o fotografiado? NO se indexa: sin texto no hay nada que recuperar (OCR aparte).
ingesta: nimbus_gastos.md: 1 página(s), 294 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_remoto.md: 1 página(s), 296 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_vacaciones.md: 1 página(s), 363 caracteres útiles → 1 fragmento(s)
fragmentos en el índice: 3

Pregunta: ¿cuál es la política de mascotas?
  0.075  nimbus_gastos.md / nimbus_gastos-0000  '# Política de reembolso de gastos de NimbusSoft Los gastos d'
  0.039  nimbus_remoto.md / nimbus_remoto-0000  '# Política de trabajo remoto de NimbusSoft El trabajo remoto'
  -0.000  nimbus_vacaciones.md / nimbus_vacaciones-0000  '# Política de vacaciones de NimbusSoft Los empleados a tiemp'

Pregunta: ¿qué exige el instructivo de prácticas de campo sobre el seguro de accidentes?
  0.229  nimbus_remoto.md / nimbus_remoto-0000  '# Política de trabajo remoto de NimbusSoft El trabajo remoto'
  0.171  nimbus_gastos.md / nimbus_gastos-0000  '# Política de reembolso de gastos de NimbusSoft Los gastos d'
  0.099  nimbus_vacaciones.md / nimbus_vacaciones-0000  '# Política de vacaciones de NimbusSoft Los empleados a tiemp'

