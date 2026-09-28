"""Fase 3 — Parte 2.3: evidencia de los fallos del baseline.

Para cada pregunta respondible que NO acertó en el top-5, busca en un top-50 en qué
posición aparece por primera vez el fragmento correcto, y muestra los 5 vecinos que
ganaron. No modifica el andamiaje ni reindexa: usa la colección ya existente en Qdrant.
Uso:  python diagnosticar_fallos.py
"""
import json
from pathlib import Path

from evaluation import acierta, es_respondible
from rag_pipeline import RagPipeline

K, PROFUNDO = 5, 50
golden = json.loads(Path("golden_set.json").read_text(encoding="utf-8"))
p = RagPipeline()

for item in golden:
    if not es_respondible(item):
        continue
    hits = p.retrieve(item["pregunta"], top_k=PROFUNDO)
    pos = next((i for i, h in enumerate(hits, start=1) if acierta(h, item)), None)
    if pos is not None and pos <= K:
        print(f"[{item['id']:>2}] {item['tipo']:<11} acierto en posición {pos}")
        continue
    donde = f"posición {pos} de {PROFUNDO}" if pos else f"fuera del top-{PROFUNDO}"
    print(f"\n[{item['id']:>2}] {item['tipo']:<11} FALLO — el fragmento correcto está en {donde}")
    print(f"     pregunta: {item['pregunta']}")
    print(f"     frase esperada: {item.get('fragmento_esperado')!r}")
    if pos:
        h = hits[pos - 1]
        print(f"     correcto: {h['score']:.3f}  {h['chunk_id']}")
    print("     ganaron en el top-5:")
    for i, h in enumerate(hits[:K], start=1):
        print(f"       {i}. {h['score']:.3f}  {h['chunk_id']}  {h['text'][:90]!r}")