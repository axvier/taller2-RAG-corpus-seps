"""Fase 3 — Verifica el golden_set.json ANTES de evaluar.

Comprueba que cada fragmento_esperado aparezca LITERAL (tras la misma normalización que
usa evaluation.py) en algún fragmento indexado de sus documentos_fuente. Si no aparece,
esa pregunta nunca podría acertar: el error sería de la anotación, no del recuperador.
No llama a los embeddings ni toca Qdrant: solo re-fragmenta corpus/.
Uso:  python verificar_golden.py
"""
import json
from pathlib import Path

from rag_pipeline import RagPipeline, _normalizar_texto

golden = json.loads(Path("golden_set.json").read_text(encoding="utf-8"))
p = RagPipeline()
chunks = p.ingest(Path("corpus"))
docs_indexados = {c.document for c in chunks}


print("\n=== Verificación del golden set ===")
tipos = {}
for item in golden:
    tipos[item["tipo"]] = tipos.get(item["tipo"], 0) + 1
    fuentes = item.get("documentos_fuente", [])
    frag = (item.get("fragmento_esperado") or "").strip()
    etiqueta = f"[{item['id']:>2}] {item['tipo']:<11}"
    if not fuentes:
        print(f"{etiqueta} negativa (sin fuentes): OK")
        continue
    faltan = [d for d in fuentes if d not in docs_indexados]
    if faltan:
        print(f"{etiqueta} ERROR: documento no indexado o mal escrito: {faltan}")
        continue
    if not frag:
        print(f"{etiqueta} sin fragmento_esperado: acepta cualquier fragmento del documento")
        continue
    donde = [c.id for c in chunks
             if c.document in fuentes and _normalizar_texto(frag) in _normalizar_texto(c.text)]
    if donde:
        print(f"{etiqueta} OK: la frase está en {', '.join(donde)}")
    else:
        print(f"{etiqueta} NO ENCONTRADA: la frase no aparece literal en ningún fragmento "
              f"(¿palabras pegadas por la extracción? usa una frase más corta)")
print(f"\nConteo por tipo: {tipos}  (total {len(golden)})")