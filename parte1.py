"""Fase 2B — Parte 1: baseline sobre corpus/ con bge-m3 (H200) y Qdrant en contenedor.

No modifica el andamiaje: usa RagPipeline con sus valores por defecto
(chunk_tokens=512, overlap_tokens=102). Indexa UNA vez y muestra, para cada pregunta
de prueba, el top-5 con documento y puntaje, y la respuesta generada.
Uso:  python parte1.py
"""
from pathlib import Path

from rag_pipeline import EMBEDDING_BACKEND, EMBEDDING_MODEL, QDRANT_URL, RagPipeline

PREGUNTAS = [
    "¿Que se reporte en la estructura de socios?",
    "¿Cuales son los codigos para identificar los tipos de credito en la estructura de cartera?",
    "¿Cual es la formula para calcular la tasa efectiva anual?",
]

p = RagPipeline()
print(f"embeddings: {EMBEDDING_BACKEND} · {EMBEDDING_MODEL} · tope {p.max_seq_tokens} "
      f"· Qdrant {QDRANT_URL} · colección {p.collection}\n")

print("=== Ingesta ===")
chunks = p.ingest(Path("corpus"))
print(f"\n=== Indexación: {len(chunks)} fragmentos ===")
p.index(chunks)
print("indexado.\n")

for i, q in enumerate(PREGUNTAS, start=1):
    print(f"=== Pregunta {i}: {q}")
    salida = p.answer(q, top_k=5)
    for pos, h in enumerate(salida["contexts"], start=1):
        print(f"  {pos}. {h['score']:.3f}  {h['document']} / {h['chunk_id']}")
        print(f"      {h['text'][:150]!r}")
    print(f"\n  [{salida['generator']}] abstuvo={salida['abstained']}")
    print(f"  {salida['answer'][:800]}\n")