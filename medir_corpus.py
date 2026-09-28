"""Fase 2 — Mide el corpus en tokens de bge-m3 para justificar chunk_tokens.

No modifica el andamiaje. Solo descarga el tokenizador de BAAI/bge-m3 (no el modelo).
Uso:  python medir_corpus.py
"""
from pathlib import Path

import transformers
from transformers import AutoTokenizer

from ingestion import MARCADOR_PAGINA, normalize_text, read_document

transformers.logging.set_verbosity_error()   # silencia el aviso de "sequence length"
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
C = 512            # chunk_tokens por defecto del lab
V = C // 5         # overlap_tokens por defecto (102)

print(f"{'documento':<45} {'pág':>4} {'palabras':>9} {'tokens':>8} {'tok/pág':>8} {'tok/pal':>8} {'frag@512':>9}")
tot_tokens = tot_frag = 0
for f in sorted(Path("corpus").glob("*")):
    if f.suffix.lower() not in {".pdf", ".txt", ".md"}:
        continue
    d = read_document(f)
    texto = normalize_text(d.text)
    n = len(tok(texto, add_special_tokens=False, truncation=False)["input_ids"])
    palabras = len(MARCADOR_PAGINA.sub("", texto).split())
    frag = 1 if n <= C else 1 + -(-(n - C) // (C - V))   # techo de (n-C)/(C-V)
    tot_tokens += n
    tot_frag += frag
    print(f"{f.name[:45]:<45} {d.pages:>4} {palabras:>9} {n:>8} {n / d.pages:>8.0f} "
          f"{n / max(palabras, 1):>8.2f} {frag:>9}")
print(f"\nTOTAL tokens: {tot_tokens}  ·  fragmentos estimados con {C}/{V}: {tot_frag}")