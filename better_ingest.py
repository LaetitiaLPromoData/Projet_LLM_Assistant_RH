# ingest.py — Ingestion MINIMALE (PDF/TXT) -> Chroma
# Usage:
#   python ingest.py --docs_dir CV --persist_dir chroma_db --force_reindex

import argparse, shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_docs(docs_dir: Path):
    docs = []
    for p in docs_dir.rglob("*"):
        if p.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(p))
                pages = loader.load()
                for i, d in enumerate(pages):
                    # Ajoute des métadonnées utiles pour l’éval et le debug
                    d.metadata.update({
                        "source": str(p),
                        "page": i,
                        "doc_id": p.stem,                     # ex: CV_PLEINET Estelle
                        "candidate": p.stem.lower(),          # simple : nom dans le filename
                    })
                docs.extend(pages)
            except Exception as e:
                print(f"[WARN] PDF illisible: {p} -> {e}")
        elif p.suffix.lower() in [".txt", ".md"]:
            try:
                t = TextLoader(str(p), encoding="utf-8").load()
                for d in t:
                    d.metadata.update({
                        "source": str(p),
                        "doc_id": p.stem,
                        "candidate": p.stem.lower(),
                    })
                docs.extend(t)
            except Exception as e:
                print(f"[WARN] TEXTE illisible: {p} -> {e}")
    return docs

def sanity_report(raw_docs):
    total_chars = sum(len(d.page_content or "") for d in raw_docs)
    empty = [d for d in raw_docs if not d.page_content or not d.page_content.strip()]
    print(f"[LOG] Documents (pages/items) chargés : {len(raw_docs)}")
    print(f"[LOG] Total caractères               : {total_chars}")
    if empty:
        print(f"[WARN] {len(empty)} items vides (PDF scannés ? OCR manquant ?)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", default="CV")
    ap.add_argument("--persist_dir", default="chroma_db")
    ap.add_argument("--force_reindex", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=120)
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir)

    if args.force_reindex and persist_dir.exists():
        print(f"[LOG] Suppression de l'ancien index: {persist_dir}")
        shutil.rmtree(persist_dir, ignore_errors=True)

    print(f"[LOG] Chargement des documents depuis: {docs_dir}")
    raw_docs = load_docs(docs_dir)
    sanity_report(raw_docs)

    print("[LOG] Découpage en segments…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[LOG] Segments créés: {len(chunks)}")

    # Garder les métadonnées importantes
    for c in chunks:
        meta = c.metadata or {}
        c.metadata = {
            "source": meta.get("source"),
            "page": meta.get("page"),
            "doc_id": meta.get("doc_id"),
            "candidate": meta.get("candidate"),
        }

    print("[LOG] Création des embeddings…")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"[LOG] Écriture Chroma -> {persist_dir}")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="cv_rag"
    )
    _ = vs.get()  # matérialise
    print("[OK] Index prêt.")

if __name__ == "__main__":
    main()
