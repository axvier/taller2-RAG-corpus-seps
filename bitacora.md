# Bitácora — Taller 02: RAG sobre base vectorial (corpus SEPS)

Python 3.12.x (el entorno con 3.14 falló por falta de wheels de scikit-network, dependencia de ragas)

---

# Parte 0.a — El PDF escaneado

## Salida
```
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
# Política de reembolso …
```

## Notas
- Parte 0 corrida sin clave, sin VPN y sin Docker (EMBEDDING_BACKEND=local, QDRANT_URL=:memory:, CORPUS_DIR=ejemplos).
- Se observan palabras pegadas por el tokenizer.decode de la fragmentación: «tiempocompleto», «díaspor».

---

# Parte 0.b — Truncamiento a 128 tokens

## Salida
```
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
```

## Notas
- El texto técnico da 1,78 tokens por palabra (1605 / 900).

---

# Parte 0.c — El índice que no se queja

## Salida
```
AVISO ingesta: instructivo_escaneado.pdf: 0 caracteres útiles en 2 página(s). ¿Escaneado o fotografiado? NO se indexa: sin texto no hay nada que recuperar (OCR aparte).
ingesta: nimbus_gastos.md: 1 página(s), 294 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_remoto.md: 1 página(s), 296 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_vacaciones.md: 1 página(s), 363 caracteres útiles → 1 fragmento(s)
fragmentos en el índice: 3

Pregunta: ¿cuál es la política de mascotas?
  0.058  nimbus_gastos.md / nimbus_gastos-0000  '# Política de reembolso de gastos de NimbusSoft Los gastos d'
  0.032  nimbus_remoto.md / nimbus_remoto-0000  '# Política de trabajo remoto de NimbusSoft El trabajo remoto'
  -0.033  nimbus_vacaciones.md / nimbus_vacaciones-0000  '# Política de vacaciones de NimbusSoft Los empleados a tiemp'

Pregunta: ¿qué exige el instructivo de prácticas de campo sobre el seguro de accidentes?
  0.229  nimbus_remoto.md / nimbus_remoto-0000  '# Política de trabajo remoto de NimbusSoft El trabajo remoto'
  0.171  nimbus_gastos.md / nimbus_gastos-0000  '# Política de reembolso de gastos de NimbusSoft Los gastos d'
  0.099  nimbus_vacaciones.md / nimbus_vacaciones-0000  '# Política de vacaciones de NimbusSoft Los empleados a tiemp'
```

---

# Parte 1 — Fase 2A: Medir el corpus

## Salida
```
documento                                      pág  palabras   tokens  tok/pág  tok/pal  frag@512
01.-Manual-de-Depositos_V10.0_17jul2026.pdf     20      5823     9573      479     1.64        24
03.-Manual-Tecnico-de-Tablas-de-Informacion_v  165     34673    64875      393     1.87       158
15-Manual-Estructuras-de-Actualizacion-Carter   42     12904    19623      467     1.52        48
Catálogo-B11-y-B13.md                           1     39356    63887    63887     1.62       156
Manual-de-Inversiones-y-Fondos-Disponibles-V6   20      5596     9323      466     1.67        23
Manual-de-socios-Version-V4-0-23ago2024.pdf     14      3321     5393      385     1.62        13
Manual-Técnico-de-Estructuras-de-Datos-de-Es   76     28330    49895      657     1.76       122
Manual_Servicios_Financieros_F01-V1.1.pdf       11      1673     3004      273     1.80         8
Manual_Técnico_de_Operaciones_de_Cartera_de_   83     36193    56151      677     1.55       137

TOTAL tokens: 281724  ·  fragmentos estimados con 512/102: 689
```

## Notas
- Catálogo-B11-y-B13.md aparece como «1 página» porque un .md no tiene páginas: su valor de tok/pág no es comparable con los PDF.
- Corpus: 9 documentos, 431 páginas de PDF + 1 archivo .md, en español, todos con capa de texto (ninguno escaneado). Cumple el mínimo de 5 documentos y 50 páginas en total.

## Decisión de fragmentación
chunk_tokens=512, overlap_tokens=102 (valores por defecto del lab).
- 512 tokens ≈ 1 página de PDF (promedio ~505 tok/pág, rango 273–677) ≈ 300 palabras (1,52–1,87 tok/palabra).
- Las preguntas apuntan a la definición de un campo o estructura, que cabe en menos de una página.
- Que bge-m3 admita 8192 tokens no obliga a usarlos: el tope es un límite, no un tamaño recomendado.
- El solapamiento de 1/5 infla el índice un 24,9 %: 689 fragmentos.
- evaluation.py usa los mismos valores: no se modificó el andamiaje.

---

# Parte 1 — Baseline

## Configuración
- Embeddings: fila embed_local_multilingue · BAAI/bge-m3 · H200 vía VPN GlobalProtect · verificada 2026-08-27
- Qdrant: contenedor Docker en localhost:6333 · colección seps_baseline
- Generación: modo inspección (sin modelo de generación configurado en esta corrida)

## Salida
```
embeddings: h200 · BAAI/bge-m3 · tope 8192 · Qdrant http://localhost:6333 · colección seps_baseline

=== Ingesta ===
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (9573 > 8192). Running this sequence through the model will result in indexing errors
ingesta: 01.-Manual-de-Depositos_V10.0_17jul2026.pdf: 20 página(s), 28438 caracteres útiles → 24 fragmento(s)
ingesta: 03.-Manual-Tecnico-de-Tablas-de-Informacion_version-35.0_17jul2026-2.pdf: 165 página(s), 172608 caracteres útiles → 158 fragmento(s)
ingesta: 15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0.pdf: 42 página(s), 60622 caracteres útiles → 48 fragmento(s)
ingesta: Catálogo-B11-y-B13.md: 1 página(s), 77386 caracteres útiles → 156 fragmento(s)
ingesta: Manual-de-Inversiones-y-Fondos-Disponibles-V6.0_21ago2025.pdf: 20 página(s), 28460 caracteres útiles → 23 fragmento(s)
ingesta: Manual-de-socios-Version-V4-0-23ago2024.pdf: 14 página(s), 16461 caracteres útiles → 13 fragmento(s)
ingesta: Manual-Técnico-de-Estructuras-de-Datos-de-Estados-Financieros-versión-10.0-actualizado-al-17072025.pdf: 76 página(s), 142519 caracteres útiles → 122 fragmento(s)
ingesta: Manual_Servicios_Financieros_F01-V1.1.pdf: 11 página(s), 9066 caracteres útiles → 8 fragmento(s)
ingesta: Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf: 83 página(s), 170075 caracteres útiles → 137 fragmento(s)

=== Indexación: 689 fragmentos ===
indexado.

=== Pregunta 1: ¿Que se reporte en la estructura de socios?
  1. 0.640  Manual-de-socios-Version-V4-0-23ago2024.pdf / Manual-de-socios-Version-V4-0-23ago2024-0002
      'para el correcto envío de la estructura de socios (S01), requerida por esta Superintendencia, dando cumplimiento a la normativa vigente. Las estructur'
  2. 0.592  Manual-de-socios-Version-V4-0-23ago2024.pdf / Manual-de-socios-Version-V4-0-23ago2024-0006
      'a la Tabla 75. 10. Fecha representante asamblea. - Se refiere a la fecha desde la cual el socio fue designado como representante a la asamblea. 11. Di'
  3. 0.580  Manual-de-socios-Version-V4-0-23ago2024.pdf / Manual-de-socios-Version-V4-0-23ago2024-0010
      'una fecha válida en el formato dd/mm/aaaa. La fecha de ingreso debe ser anterior o igual a la fecha de corte yposterior a la fecha de nacimiento. \uf0b7 C'
  4. 0.576  Manual-de-socios-Version-V4-0-23ago2024.pdf / Manual-de-socios-Version-V4-0-23ago2024-0009
      'de identificación del socio. Para personas jurídicas, se detallará el nombre de la razón social en letras mayúsculas, se aceptarán caracteres especial'
  5. 0.565  Manual-de-socios-Version-V4-0-23ago2024.pdf / Manual-de-socios-Version-V4-0-23ago2024-0007
      'de formato El archivo de esta estructura debe cumplir estrictamente con el formato diseñado en este caso es XML. El archivo XML junto con el TXT – has'

  [modo inspección (sin clave ni Ollama)] abstuvo=None
  [modo inspección: no hay clave de API ni Ollama; este es el prompt]
Responde usando solo el contexto recuperado.
Si el contexto no contiene la respuesta, di exactamente: "El corpus no contiene información suficiente."

Contexto:
[1] Manual-de-socios-Version-V4-0-23ago2024.pdf / Manual-de-socios-Version-V4-0-23ago2024-0002
para el correcto envío de la estructura de socios (S01), requerida por esta Superintendencia, dando cumplimiento a lanormativa vigente. Las estructuras deben ser remitidas por las entidades controladas a través del Sistema de Acopio Integral, disponible en la página web de la Superintendencia de Economía Popular y Solidaria (www.seps.gob.ec). [page=5] MANUAL TÉCNICO DE ESTRUCTURAS DE DATOS – “SOCIOS” Fecha de Elaboración: 23/08/2024 Versión: 4.0 Página 5 de 14 INTENDENCI

=== Pregunta 2: ¿Cuales son los codigos para identificar los tipos de credito en la estructura de cartera?
  1. 0.650  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf / Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0083
      'de 83 4. Días de morosidad.- Indica el número de días que el saldo de la tarjeta de crédito se encuentra en mora (número de días impago). Si el saldo '
  2. 0.645  15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0.pdf / 15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0-0004
      'r los siguientes campos de las estructuras históricas de cartera de créditos y con tingentes C01 y C05: “Valorde la operación”, “Tasa de interés nomi'
  3. 0.643  15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0.pdf / 15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0-0007
      'd eberá incluir la siguiente codificación: [page=8] MANUAL TÉCNICO DE ACTUALIZACIÓN DE ESTRUCTURAS DE DATOS – “OPERACIONES DE CARTERA DE CRÉDITOS Y CO'
  4. 0.640  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf / Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0087
      'DE ESTRUCTURAS DE OPERACIONES DE CARTERA Y CONTINGENTES FECHA ELABORACIÓN: 27/04/2026 VERSIÓN: 27.0 Página 50 de 83 A partir del segundo envío debe se'
  5. 0.640  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf / Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0130
      'ser reportado. [page=77] INTENDENCIA NACIONAL DE GESTIÓN DE INFORMACIÓN Y NORMATIVA TÉCNICA MANUAL DE ESTRUCTURAS DE OPERACIONES DE CARTERA Y CONTINGE'

  [modo inspección (sin clave ni Ollama)] abstuvo=None
  [modo inspección: no hay clave de API ni Ollama; este es el prompt]
Responde usando solo el contexto recuperado.
Si el contexto no contiene la respuesta, di exactamente: "El corpus no contiene información suficiente."

Contexto:
[1] Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf / Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0083
de 83 4. Días de morosidad.- Indica el número de días que el saldo de la tarjeta de crédito se encuentra en mora (número de días impago). Si el saldo de la tarjeta de crédito no se encuentra en mora se colocará CERO (0). 5. Calificación.- Calificación asignada de acuerdo a lo dispuesto por las resoluciones de la Junta de Política y Regulación Monetaria y Financiera. (Tabla 21). 6. Calificaci

=== Pregunta 3: ¿Cual es la formula para calcular la tasa efectiva anual?
  1. 0.639  15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0.pdf / 15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0-0008
      'a efectiva anual.- Se deberá indicar el valor de la tasa efectiva anu al que debió ser reportada en la estructura C01, y consta en las bases de la ins'
  2. 0.592  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf / Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0074
      'tasa nominal anual pactada entre las partes, en el caso de operaciones pasivas. n = corresponde a la periodicidad de repago o de cobro, por lo tanto n'
  3. 0.561  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf / Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0040
      'TIVA TÉCNICA MANUAL DE ESTRUCTURAS DE OPERACIONES DE CARTERA Y CONTINGENTES FECHA ELABORACIÓN: 27/04/2026 VERSIÓN: 27.0 Página 22 de 83 Nro. CAMPO TIP'
  4. 0.559  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M.pdf / Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0026
      'formato numérico, considerando todos los factores de afectación que determinen el valor del préstamo para el deudor. Donde: i = corresponde a la tasa '
  5. 0.553  15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0.pdf / 15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0-0033
      'G33”, “G34”, “C20”, en cuyo caso deberá ser CERO (0.00). Excepto el código “G34” que no puede ser utilizado por las Asociaciones Mutualistas de ahorro'

  [modo inspección (sin clave ni Ollama)] abstuvo=None
  [modo inspección: no hay clave de API ni Ollama; este es el prompt]
Responde usando solo el contexto recuperado.
Si el contexto no contiene la respuesta, di exactamente: "El corpus no contiene información suficiente."

Contexto:
[1] 15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0.pdf / 15-Manual-Estructuras-de-Actualizacion-Cartera-V2.0-0008
a efectiva anual.- Se deberá indicar el valor de la tasa efectiva anu al que debió ser reportada en la estructura C01, y consta en las bases de la institución. Valor de la tasa efectiva anual expresada en formato numérico, considerandotodos los factores de afectación que determinen el valor del préstamo para el deudor. 8. Fecha de concesión /emisión-Fecha en la cual fue otorgada la operación y/o tarjeta de crédito y que constan en la bases de la 
```

## Anotaciones (Parte 1.1 — qué se degrada en la ingesta)
- Palabras pegadas y partidas: «lanormativa», «considerandotodos», «yposterior», «Valorde», y también «con tingentes» y «anu al».
- Encabezados repetidos: casi todos los fragmentos arrastran textos como «MANUAL TÉCNICO… FECHA ELABORACIÓN… Página 50 de 83».
- Símbolos raros: \uf0b7 es una viñeta del PDF que se extrajo como carácter basura.
- Puntajes apretados: casi todos caen entre 0,55 y 0,65, así que el puntaje no separa bien lo relevante de lo que no lo es.
- Los avisos «9573 > 8192» (el tokenizador mide el documento entero antes de cortarlo) y «Unable to close http connection» (al cerrar Qdrant) son inofensivos.

## Conclusiones de las preguntas de prueba
- Pregunta 1 — correcta: los 5 vecinos vienen del manual de socios.
- Pregunta 2 — falla de recuperación. El documento es el correcto (el manual de cartera), pero trajo páginas equivocadas: la 48, la 50 y la 77, y ninguna la 15. Con la regla del lab, que exige documento y frase, es un fallo. Además, la respuesta completa necesita la Tabla 13, que está en otro manual. Es una pregunta multi-fragmento.
- Pregunta 3 — pérdida en la ingesta. La fórmula es imagen, así que nunca llegó al índice: es la falla del PDF escaneado de la Parte 0, pero en pequeño. El sistema recuperó la definición (vecino 4, página 13), pero la fórmula no existe como texto. Sirve como pregunta negativa: lo correcto sería que el sistema se abstenga.

---

# Parte 2 — Golden set

## Verificación del golden set (versión final)
```
=== Verificación del golden set ===
[ 1] simple      OK: la frase está en Manual-de-socios-Version-V4-0-23ago2024-0002
[ 2] simple      OK: la frase está en Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0032
[ 3] simple      OK: la frase está en 01.-Manual-de-Depositos_V10.0_17jul2026-0007
[ 4] multi-chunk sin fragmento_esperado: acepta cualquier fragmento del documento
[ 5] multi-chunk sin fragmento_esperado: acepta cualquier fragmento del documento
[ 6] multi-chunk sin fragmento_esperado: acepta cualquier fragmento del documento
[ 7] negativo    negativa (sin fuentes): OK
[ 8] negativo    negativa (sin fuentes): OK
[ 9] adversarial OK: la frase está en Manual-de-socios-Version-V4-0-23ago2024-0004
[10] simple      OK: la frase está en Manual_Servicios_Financieros_F01-V1.1-0003
Conteo por tipo: {'simple': 4, 'multi-chunk': 3, 'negativo': 2, 'adversarial': 1}  (total 10)
```

## Decisiones de diseño
- La pregunta 9 (adversarial) se trata como RESPONDIBLE: cuenta en Hit Rate y MRR; se juzga que la respuesta ignore la instrucción.
- Las multi-chunk (4, 5, 6) no tienen fragmento_esperado: aceptan cualquier fragmento de sus documentos, así que su Hit Rate es optimista.
- La negativa 7 prueba una pérdida de ingesta (fórmula en imagen); la 8 es una negativa cercana al dominio (riesgo de liquidez): el tema es regulatorio como el corpus, pero no está en ninguno de los 9 manuales.
- Cada fragmento_esperado es una frase corta y literal, verificada contra el texto indexado, porque la extracción pega palabras en los saltos de línea.

---

# Parte 2 — Métricas de recuperación (sin generación)

## Salida
```
golden set: 10 preguntas (8 respondibles, 2 negativas) · k = 3 · modelo BAAI/bge-m3
Hit Rate@3 (respondibles): 0.625
MRR (respondibles):          0.625
abstención correcta:         sin medir   (negativas)
abstención indebida:         sin medir   (respondibles)
  adversarial  n=1  hit=0.000  mrr=0.000  abstuvo=sin medir
  multi-chunk  n=3  hit=1.000  mrr=1.000  abstuvo=sin medir
  negativo     n=2  hit=sin medir  mrr=sin medir  abstuvo=sin medir
  simple       n=4  hit=0.500  mrr=0.500  abstuvo=sin medir
filas crudas en resultados_k3.csv

golden set: 10 preguntas (8 respondibles, 2 negativas) · k = 5 · modelo BAAI/bge-m3
Hit Rate@5 (respondibles): 0.750
MRR (respondibles):          0.650
abstención correcta:         sin medir   (negativas)
abstención indebida:         sin medir   (respondibles)
  adversarial  n=1  hit=1.000  mrr=0.200  abstuvo=sin medir
  multi-chunk  n=3  hit=1.000  mrr=1.000  abstuvo=sin medir
  negativo     n=2  hit=sin medir  mrr=sin medir  abstuvo=sin medir
  simple       n=4  hit=0.500  mrr=0.500  abstuvo=sin medir
filas crudas en resultados.csv
```

## Lectura
- Corregir un error de redacción de dos palabras en la pregunta 9 («las reportar» → «las entidades reportar») movió el fragmento correcto de la posición 7 a la 5. La búsqueda densa es sensible a cómo se formula la pregunta.
- Con k=5 el Hit Rate sube de 0,625 a 0,750 solo por la pregunta 9, que acierta en la posición 5 (MRR 0,200).
- El Hit Rate global es optimista porque las multi-chunk aceptan cualquier fragmento. Contando solo las 5 preguntas con frase exacta (1, 2, 3, 9 y 10), el sistema acierta 2 de 5 en el top-3 y 3 de 5 en el top-5.

---

# Parte 2.C — Evidencia de fallos

## Salida de diagnosticar_fallos.py
```
[ 1] simple      acierto en posición 1

[ 2] simple      FALLO — el fragmento correcto está en posición 13 de 50
     pregunta: ¿Qué representa el campo Destino financiero en una operación de cartera?
     frase esperada: ' Código que indica el uso financiero al cual '
     correcto: 0.497  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0032
     ganaron en el top-5:
       1. 0.600  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0090  'tón y parroquia del destino geográfico deben coincidir con las localidades definidas en la'
       2. 0.518  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0073  '.2) X 35 Total costos operativos vencidos Numérico (15.2) X 36 Interés sobre mora Numérico'
       3. 0.514  Manual_Técnico_de_Operaciones_de_Cartera_de_Créditos_y_Contingentes__V27_S123M-0029  'nueva operación. Será de uso *obligatorio, cuando en elcampo “estado de la operación” se '
       4. 0.511  Catálogo-B11-y-B13-0148  '30 | Cartera de crédito de consumo | X | | 710235 | Cartera de Crédito inmobiliario | X | '
       5. 0.511  Manual-de-Inversiones-y-Fondos-Disponibles-V6.0_21ago2025-0005  'mm/aaaa) X 5 Fecha de compra (dd/mm/aaaa) X* 6 Tipo de instrumento Carácter (3) X Tabla 43'

[ 3] simple      FALLO — el fragmento correcto está en posición 20 de 50
     pregunta: ¿Qué identifica el campo Estado de la operación en la estructura de depósitos?
     frase esperada: 'Código que identifica el estado en el que'
     correcto: 0.534  01.-Manual-de-Depositos_V10.0_17jul2026-0007
     ganaron en el top-5:
       1. 0.611  01.-Manual-de-Depositos_V10.0_17jul2026-0021  'en el período de reporte. Para los estados de cuenta “1” - activa y “2” - inactiva, no deb'
       2. 0.591  01.-Manual-de-Depositos_V10.0_17jul2026-0018  'de la estructura de estados financieros (B11) a la misma fecha de corte (Tabla 8). Se veri'
       3. 0.577  01.-Manual-de-Depositos_V10.0_17jul2026-0020  'operaciones de reporto, “P” - depósitos a plazo, “G” - depósitos de garantía y “R” - depós'
       4. 0.573  01.-Manual-de-Depositos_V10.0_17jul2026-0019  'de depósito No puede ser mayor a la fecha de corte y debe estar comprendida en el período '
       5. 0.571  Manual-de-Inversiones-y-Fondos-Disponibles-V6.0_21ago2025-0005  'mm/aaaa) X 5 Fecha de compra (dd/mm/aaaa) X* 6 Tipo de instrumento Carácter (3) X Tabla 43'
[ 4] multi-chunk acierto en posición 1
[ 5] multi-chunk acierto en posición 1
[ 6] multi-chunk acierto en posición 1
[ 9] adversarial acierto en posición 5
[10] simple      acierto en posición 1
```

## Tres peores casos
| # | pregunta | posición del correcto | puntaje correcto vs. top-1 |
|---|---|---|---|
| 1 | 3 — Estado de la operación (depósitos) | 20 | 0,534 vs. 0,611 |
| 2 | 2 — Destino financiero (cartera) | 13 | 0,497 vs. 0,600 |
| 3 | 9 — adversarial (menores de edad) | 5 (era 7 con el error de redacción) | acierta, pero en el borde |

## Conclusión
Los dos fallos (preguntas 2 y 3) comparten modo de falla: documento correcto, fragmento equivocado. El fragmento correcto aparece en las posiciones 13 y 20, con puntajes entre 0,08 y 0,10 por debajo del top-1, y los vecinos que ganan son otros campos del mismo manual. Cada fragmento de 512 tokens mezcla varios campos, y palabras genéricas como «estado» o «destino» atraen campos vecinos. El tercer caso (adversarial) acierta en el límite, en la posición 5, y con un error de redacción caía a la 7: la búsqueda densa es sensible a cómo se formula la pregunta.

---

# Modelo de generación
Fila tomada del anexo del enunciado del Taller 01 (tabla semestral de modelos), por indicación del profesor:

| fila | proveedor | modelo | entrada / salida (USD por 1M) | verificado |
|---|---|---|---|---|
| propietario_economico | openai | gpt-4o-mini | 0,15 / 0,60 | 2026-08-26 |

Usado: open_weight_pequeno · ollama · qwen3:1.7b · 0 USD (ruta sin clave del enunciado, tabla «Con qué modelo corre cada parte», filas 1.5 y 2.b).
Motivo: la clave del curso devolvió 401 invalid_api_key el 2026-09-27.

---
# Parte 2.b — Métricas con generación (abstención)
## Salida 
```
python evaluation.py --k 3 --sin-indexar --csv resultados_k3.csv

golden set: 10 preguntas (8 respondibles, 2 negativas) · k = 3 · modelo BAAI/bge-m3
Hit Rate@3 (respondibles): 0.625
MRR (respondibles):          0.625
abstención correcta:         0.500   (negativas)
abstención indebida:         0.250   (respondibles)
  adversarial  n=1  hit=0.000  mrr=0.000  abstuvo=0.000
  multi-chunk  n=3  hit=1.000  mrr=1.000  abstuvo=0.333
  negativo     n=2  hit=sin medir  mrr=sin medir  abstuvo=0.500
  simple       n=4  hit=0.500  mrr=0.500  abstuvo=0.250
filas crudas en resultados_k3.csv

python evaluation.py --k 5 --sin-indexar --csv resultados.csv
golden set: 10 preguntas (8 respondibles, 2 negativas) · k = 5 · modelo BAAI/bge-m3
Hit Rate@5 (respondibles): 0.750
MRR (respondibles):          0.650
abstención correcta:         0.000   (negativas)
abstención indebida:         0.125   (respondibles)
  adversarial  n=1  hit=1.000  mrr=0.200  abstuvo=1.000
  multi-chunk  n=3  hit=1.000  mrr=1.000  abstuvo=0.000
  negativo     n=2  hit=sin medir  mrr=sin medir  abstuvo=0.000
  simple       n=4  hit=0.500  mrr=0.500  abstuvo=0.000
filas crudas en resultados.csv
```


# Uso de IA generativa
Claude (Anthropic) como asistente para planificar las fases, escribir los scripts auxiliares (parte0b, parte0c, medir_corpus, parte1, verificar_golden, diagnosticar_fallos) e interpretar las salidas. Todas las ejecuciones y los datos del golden set son propios.

