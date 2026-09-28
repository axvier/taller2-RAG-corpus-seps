"""Parte 3 — Opción C: reranking con cross-encoder sobre el baseline.

Recupera top-20 con bge-m3 (la misma colección seps_baseline, sin reindexar), reordena
con el cross-encoder de la fila rerank_local_multilingue y entrega top-k. Evalúa con el
mismo golden set y las mismas funciones de evaluation.py, y escribe el CSV crudo.
No modifica el andamiaje: hereda de RagPipeline y solo sobrescribe retrieve(), así que
la generación (answer) también usa el contexto reordenado.
Uso:  python evaluar_reranker.py --k 3 --csv resultados_rerank_k3.csv
      python evaluar_reranker.py --k 5 --csv resultados_rerank_k5.csv
      (añade --sin-generar para medir solo recuperación)
"""
import argparse
from pathlib import Path

from sentence_transformers import CrossEncoder

import evaluation as ev
from rag_pipeline import EMBEDDING_MODEL, RagPipeline

RERANKER_ID = "rerank_local_multilingue"                      # fila de la tabla semestral
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # verificada 2026-09-19
CANDIDATOS = 20


class RagConReranker(RagPipeline):
    def __init__(self, n_candidatos: int = CANDIDATOS):
        super().__init__()
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.n_candidatos = n_candidatos
        self.tope_reranker = self.reranker.tokenizer.model_max_length
        self.pares_truncados = 0
        self.pares_totales = 0

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        candidatos = super().retrieve(question, top_k=max(self.n_candidatos, top_k))
        pares = [(question, h["text"]) for h in candidatos]
        # Falla silenciosa a vigilar (como la Parte 0.b): el cross-encoder también tiene
        # tope de tokens y trunca el par (pregunta + fragmento) sin avisar.
        for q, t in pares:
            n = len(self.reranker.tokenizer(q, t, truncation=False)["input_ids"])
            self.pares_totales += 1
            self.pares_truncados += n > self.tope_reranker
        puntajes = self.reranker.predict(pares)
        for h, s in zip(candidatos, puntajes):
            h["score_denso"] = h["score"]
            h["score"] = float(s)
        candidatos.sort(key=lambda h: h["score"], reverse=True)
        return candidatos[:top_k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("golden_set.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--csv", type=Path, default=Path("resultados_rerank.csv"))
    ap.add_argument("--sin-generar", action="store_true")
    args = ap.parse_args()

    p = RagConReranker()
    golden = ev.load_golden_set(args.golden)
    r = ev.evaluate_retrieval(golden, p, k=args.k, generar=not args.sin_generar)
    etiqueta = f"{EMBEDDING_MODEL} + {RERANKER_MODEL} (top-{CANDIDATOS} → top-{args.k})"
    ev.escribir_csv(r["rows"], args.csv, etiqueta)

    f = ev._fmt
    print(f"golden set: {r['n_preguntas']} preguntas ({r['n_respondibles']} respondibles, "
          f"{r['n_negativas']} negativas) · k = {r['k']} · {etiqueta}")
    print(f"reranker: fila {RERANKER_ID} · tope {p.tope_reranker} tokens · "
          f"pares truncados: {p.pares_truncados} de {p.pares_totales}")
    print(f"Hit Rate@{args.k} (respondibles): {f(r['hit_rate'])}")
    print(f"MRR (respondibles):          {f(r['mrr'])}")
    print(f"abstención correcta:         {f(r['abstencion_correcta'])}   (negativas)")
    print(f"abstención indebida:         {f(r['abstencion_indebida'])}   (respondibles)")
    for tipo, m in r["por_tipo"].items():
        print(f"  {tipo:<12} n={m['n']}  hit={f(m['hit_rate'])}  mrr={f(m['mrr'])}  abstuvo={f(m['abstuvo'])}")
    print("posición del primer acierto por pregunta:",
          {row["id"]: row["posicion"] for row in r["rows"] if row["respondible"]})
    print(f"filas crudas en {args.csv}")