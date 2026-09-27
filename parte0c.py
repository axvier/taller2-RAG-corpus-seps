"""Parte 0.c — El índice que no se queja (MiniLM, Qdrant en memoria).

No modifica el andamiaje. Indexa ejemplos/ y muestra los 3 vecinos de dos preguntas:
una cuya respuesta no está y otra cuya respuesta está en el PDF escaneado no indexado.
"""
import os
from pathlib import Path

os.environ["EMBEDDING_BACKEND"] = "local"
os.environ["QDRANT_URL"] = ":memory:"

from rag_pipeline import RagPipeline  # noqa: E402

p = RagPipeline()
chunks = p.ingest(Path("ejemplos"))
p.index(chunks)
print(f"fragmentos en el índice: {len(chunks)}\n")

preguntas = [
    "¿cuál es la política de mascotas?",
    "¿qué exige el instructivo de prácticas de campo sobre el seguro de accidentes?",
]
for q in preguntas:
    print(f"Pregunta: {q}")
    for h in p.retrieve(q, top_k=3):
        print(f"  {h['score']:.3f}  {h['document']} / {h['chunk_id']}  {h['text'][:60]!r}")
    print()