"""Parte 2.b / 2.c — Muestra la respuesta generada para cada pregunta del golden set.

evaluation.py cuenta abstenciones pero no guarda el texto de las respuestas; este script
lo muestra para poder juzgar cada caso: ¿se abstuvo con otras palabras? ¿alucinó?
¿obedeció la instrucción adversarial? No modifica el andamiaje ni reindexa.
Uso:  python ver_respuestas.py 3      (o 5)
"""
import json
import sys
from pathlib import Path

from rag_pipeline import RagPipeline

k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
golden = json.loads(Path("golden_set.json").read_text(encoding="utf-8"))
p = RagPipeline()

for item in golden:
    s = p.answer(item["pregunta"], top_k=k)
    print(f"\n=== [{item['id']}] {item['tipo']} · k={k} · {s['generator']} · abstuvo={s['abstained']}")
    print(f"pregunta:  {item['pregunta']}")
    print(f"esperada:  {item['respuesta_esperada']}")
    print(f"vecinos:   " + ", ".join(f"{h['chunk_id'][-4:]}({h['score']:.3f})" for h in s["contexts"]))
    print(f"respuesta: {s['answer'].strip()[:700]}")