# main.py — minimal retrieval + RAG (Gemini)

import os
from pathlib import Path

# --- LangChain de base (PDF -> chunks -> Chroma) ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- Gemini pour la génération ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ========== Utils ingestion / index ==========
def load_pdfs(pdf_dir: str):
    """Charge tous les PDF d'un dossier en Documents."""
    loader = PyPDFDirectoryLoader(pdf_dir)
    return loader.load()  # -> List[Document]

def split_docs(docs, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Découpe les documents en chunks pour l'indexation."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def build_or_load_db(chunks, persist_dir: str):
    """Crée la base Chroma si absente, sinon la recharge depuis le disque."""
    
    ## embeddings de type SentenceTransformer
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    p = Path(persist_dir)
    if p.exists() and any(p.iterdir()):
        return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    # sinon on crée à partir des chunks
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    db.persist()
    return db

def search(db, query: str, k: int = 3):
    """Recherche sémantique simple et affiche un aperçu."""
    results = db.similarity_search(query, k=k)
    for i, r in enumerate(results, 1):
        preview = r.page_content.replace("\n", " ")[:300]
        print(f"\n--- Résultat {i}/{k} ---")
        print(preview)
        print("META:", {k: r.metadata.get(k) for k in ("source", "page", "title", "author")})
    print("\n[LOG] Recherche terminée.")


# ========== test : test lescture des pdf split + db ==========
def test():
    # Test rapide
    PDF_DIR = "CV"
    PERSIST_DIR = "chroma_db_test"

    # Permet la lecture des documents
    docs = load_pdfs(PDF_DIR)
    # Découpage en chunks, les chunks = des morceaux de texte
    docs = docs[:1]  # pour le test, on ne prend que 1 CV
    chunks = split_docs(docs)
    # Construction de l’index (base Chroma) ou chargement s’il existe déjà
    db = build_or_load_db(chunks, PERSIST_DIR)
    # Recherche
    search(db, "langage", k=3)


# ========== Exemple avec Gemini  ==========
def ask_gemini(question: str):
    """
    Récupère du contexte via Chroma puis appelle Gemini pour rédiger une réponse courte.
    Nécessite GOOGLE_API_KEY dans l'environnement et les paquets:
      - langchain-google-genai
      - google-generativeai
    """
    PDF_DIR = "CV"
    PERSIST_DIR = "chroma_db_test"

    # 1) On vérifie que la base ne soit pas déjà remplie
    if not (Path(PERSIST_DIR).exists() and any(Path(PERSIST_DIR).iterdir())):
        docs = load_pdfs(PDF_DIR)
        chunks = split_docs(docs)
        db = build_or_load_db(chunks, PERSIST_DIR)
    else:
        db = build_or_load_db([], PERSIST_DIR)

    # 2) Récupère quelques passages pertinents
    retriever = db.as_retriever(search_kwargs={"k": 6})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # 3) Prompt minimal et appel Gemini (pas d'enrobage)
    prompt = ChatPromptTemplate.from_template(
        "Réponds uniquement avec le contenu du contexte. Si l'info manque, dis-le.\n\n"
        "Question:\n{q}\n\nContexte:\n{ctx}\n\nRéponse:"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    messages = prompt.format_messages(q=question, ctx=context)
    out = llm.invoke(messages)

    print("\n=== Réponse (Gemini) ===\n")
    print(out.content)


# ========== Mini CLI ==========
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print(
            "Usage :\n"
            "  uv run python main.py test\n"
            '  uv run python main.py ask-gemini "Qui maîtrise PyTorch ?"\n'
        )
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "test":
        test()
    elif cmd == "ask-gemini":
        question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Quels langages ce candidat maîtrise-t-il ?"
        ask_gemini(question)
    else:
        print(f"Commande inconnue: {cmd}")
