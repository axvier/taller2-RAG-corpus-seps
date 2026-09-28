# Taller 02 — RAG sobre un corpus real (corpus SEPS)

MMIA 6013 IA Generativa y Agentes · USFQ

Este repositorio parte del andamiaje `Lab-02-RAG-VectorSearch` **sin modificarlo**
(`rag_pipeline.py`, `ingestion.py` y `evaluation.py` están tal como se entregaron; el
segundo commit del historial es el andamiaje original). Todo lo propio está en scripts
aparte que importan el pipeline.

## Entorno
- Python **3.12** (con 3.14 falla la instalación: `scikit-network`, dependencia de `ragas`, no publica wheels).
- Dependencias con versiones fijadas en `requirements.txt`.
- Configuración en `.env` (a partir de `.env.example`; excluido por `.gitignore`). Ninguna clave está en el repositorio.

```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

## Corpus
Opción C, corpus propio: 9 manuales técnicos públicos de estructuras de datos de la SEPS,
en español (431 páginas de PDF + 1 archivo .md), en `corpus/`.

## Modelos (filas de la tabla semestral)
| uso | fila | modelo | verificada |
|---|---|---|---|
| Embeddings, Parte 0 | embed_notebook_s2 | paraphrase-multilingual-MiniLM-L12-v2 | 2026-08-27 |
| Embeddings, Partes 1–3 | embed_local_multilingue | BAAI/bge-m3 (H200, VPN) | 2026-08-27 |
| Generación | open_weight_pequeno | qwen3:1.7b (Ollama) | — |
| Reranking, Parte 3 | rerank_local_multilingue | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 2026-09-19 |

## Cómo reproducir cada parte
**Parte 0** (sin VPN, sin Docker, sin clave):
```powershell
$env:EMBEDDING_BACKEND="local"; $env:QDRANT_URL=":memory:"; $env:CORPUS_DIR="ejemplos"
python rag_pipeline.py "¿cuántos días de vacaciones puedo transferir?"   # 0.a
python parte0b.py                                                          # 0.b
python parte0c.py                                                          # 0.c
```
Cerrar la ventana después, para que esas variables no afecten al resto.

**Parte 1** (VPN GlobalProtect + Qdrant en Docker):
```powershell
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
python medir_corpus.py     # tokens por documento → justificación de chunk_tokens=512
python parte1.py           # ingesta, indexación (colección seps_baseline) y top-5 de 3 preguntas
```

**Parte 2** (golden set, métricas y fallos; generación con Ollama):
```powershell
python verificar_golden.py                                   # cada fragmento_esperado existe literal
python evaluation.py --k 3 --sin-indexar --csv resultados_k3.csv
python evaluation.py --k 5 --sin-indexar --csv resultados.csv
python ver_respuestas.py 3
python ver_respuestas.py 5
python diagnosticar_fallos.py                                # posición del correcto en un top-50
```

**Parte 3** (Opción C, reranking):
```powershell
python evaluar_reranker.py --k 3 --csv resultados_rerank_k3.csv
python evaluar_reranker.py --k 5 --csv resultados_rerank_k5.csv
python evaluar_reranker.py --k 3 --sin-generar --csv resultados_rerank_k3_recuperacion.csv
python evaluar_reranker.py --k 5 --sin-generar --csv resultados_rerank_k5_recuperacion.csv
```

## Archivos de resultados
| archivo | contenido |
|---|---|
| `resultados_k3.csv`, `resultados.csv` | baseline, k=3 y k=5, con generación (abstención) |
| `resultados_rerank_k3.csv`, `resultados_rerank_k5.csv` | reranking, con generación |
| `resultados_rerank_k*_recuperacion.csv` | reranking, solo recuperación |
| `golden_set.json` | el verificador: 10 preguntas (4 simples, 3 multi-fragmento, 2 negativas, 1 adversarial) |
| `bitacora.md` | todas las salidas crudas y su interpretación |