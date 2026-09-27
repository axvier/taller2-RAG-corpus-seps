"""Parte 0.b — El fragmento que se corta a 128 tokens (MiniLM, EMBEDDING_BACKEND=local).

No modifica el andamiaje: solo llama a RagPipeline e ingestion tal como vienen.
Uso:  python parte0b.py                 (toma el primer PDF de corpus/)
      python parte0b.py corpus/X.pdf    (o el archivo que indiques)
"""
import os
import sys
from pathlib import Path

# Parte 0: sin clave, sin VPN, sin Docker. Se fija ANTES de importar el pipeline.
os.environ["EMBEDDING_BACKEND"] = "local"
os.environ["QDRANT_URL"] = ":memory:"

from ingestion import MARCADOR_PAGINA, normalize_text, read_document  # noqa: E402
from rag_pipeline import RagPipeline  # noqa: E402

p = RagPipeline()
tope = p.max_seq_tokens
tok = p.encoder.tokenizer

print("=== 1) Ingesta de ejemplos/ pidiendo chunk_tokens=900 ===")
p.ingest(Path("ejemplos"), chunk_tokens=900)

print("\n=== 2) Texto de ~900 palabras: entero vs. recortado al tope ===")
archivo = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("corpus").glob("*.pdf"))[0]
texto = normalize_text(MARCADOR_PAGINA.sub("", read_document(archivo).text))
texto = " ".join(texto.split()[:900])
ids = tok(texto, add_special_tokens=False, truncation=False)["input_ids"]
recortado = tok.decode(ids[: tope - 2])   # el tope incluye los 2 tokens especiales
v = p.encoder.encode([texto, recortado])  # vectores normalizados -> dot = coseno
coseno = float(v[0] @ v[1])

print(f"archivo fuente:          {archivo.name}")
print(f"palabras del texto:      {len(texto.split())}")
print(f"tokens del texto:        {len(ids)}")
print(f"tope del modelo:         {tope} (max_seq_length; {tope - 2} de contenido)")
print(f"fracción que cabe:       {(tope - 2) / len(ids):.1%}")
print(f"coseno(entero, cortado): {coseno:.4f}")