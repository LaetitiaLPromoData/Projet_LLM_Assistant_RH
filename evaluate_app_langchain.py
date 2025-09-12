# -*- coding: utf-8 -*-
"""
evaluate_app_langchain.py — Évalue la fonction handle_query en dehors de Streamlit.
Métriques : LLM-judge (QAEvalChain) + distance d'embeddings.
"""
from pathlib import Path
import os

from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSIST_DIR = str((Path(__file__).parent / "chroma_db").resolve())
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="cv_rag",
    )
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20}), embeddings


def candidate_key(meta: dict) -> str:
    # Regroupe par identifiant de candidat si dispo, sinon par source
    if not meta:
        return "?"
    return meta.get("candidate") or meta.get("doc_id") or meta.get("source", "?")


def handle_query(query: str, retriever, llm) -> str:
    docs = retriever.invoke(query)
    grouped = {}
    for d in docs:
        key = candidate_key(getattr(d, "metadata", {}))
        grouped.setdefault(key, []).append(d.page_content.replace("\n", " "))
    context = ""
    for person, snippets in grouped.items():
        joined = " ".join(snippets[:3])
        context += f"\n\n### CANDIDAT: {person}\n{joined}"
    user_prompt = f"""
Tu as des extraits provenant de plusieurs CV (séparés par ### CANDIDAT: nom).
Réponds à la question suivante en donnant un récapitulatif clair par candidat.

QUESTION:
{query}

EXTRAITS:
{context}
"""
    return llm.invoke(user_prompt).content

# ==== Jeu d'exemples (références)
examples = [
    {
        "query": "Jip est-elle forte en biochimie ?",
        "reference": "Oui, Jip est docteure en biochimie et spécialisée dans ce domaine."
    },
    {
        "query": "Que fait Jip ?",
        "reference": "Jip est chercheuse en biochimie en reconversion."
    },
    {
        "query": "Véronique est-elle adepte du leadership ?",
        "reference": "Véronique a encadré une équipe et mené des projets, preuve de leadership."
    },
    {
        "query": "Qui est Estelle Pleinet ?",
        "reference": "Estelle Pleinet est développeuse en reconversion data scientist avec un master en informatique."
    },
    
]

def main():
    # LLM pour génération de réponses (même que l'app)
    if not os.getenv("GOOGLE_API_KEY"):
        print("[ERR] GOOGLE_API_KEY manquant dans l'environnement.")
        print("      export GOOGLE_API_KEY=\"...\"")
        return

    gen_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    retriever, embeddings = build_retriever()

    # Juge LLM (on reste sur Gemini pour cohérence)
    judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    qa_eval = load_evaluator("qa", llm=judge_llm)

    print("# Évaluation LLM-judge (QAEvalChain)")
    for ex in examples:
        pred = handle_query(ex["query"], retriever, gen_llm)
        graded = qa_eval.evaluate_strings(
            prediction=pred,
            reference=ex["reference"],
            input=ex["query"],
        )
        print("Q:", ex["query"])
        print("Pred:", pred[:200], "...")
        print("Ref:", ex["reference"])
        print("Eval:", graded)
        print("-" * 60)

    print("# Évaluation Similarité d'Embeddings")
    emb_eval = load_evaluator("embedding_distance", embeddings=embeddings)
    for ex in examples:
        pred = handle_query(ex["query"], retriever, gen_llm)
        score = emb_eval.evaluate_strings(
            prediction=pred,
            reference=ex["reference"],
        )
        print(ex["query"], "->", score)

if __name__ == "__main__":
    main()
