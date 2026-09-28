from pathlib import Path
from pypdf import PdfReader

total_pag = 0
for f in sorted(Path("corpus").glob("*")):
    if f.suffix.lower() == ".pdf":
        r = PdfReader(f)
        pags = len(r.pages)
        chars = sum(len((p.extract_text() or "").strip()) for p in r.pages)
        total_pag += pags
        aviso = "  <-- ¿ESCANEADO?" if chars / max(pags, 1) < 100 else ""
        print(f"{f.name:60} {pags:4} pág  {chars:8} caracteres{aviso}")
    else:
        print(f"{f.name:60} (texto)  {len(f.read_text(encoding='utf-8'))} caracteres")
print(f"\nTOTAL páginas PDF: {total_pag}")