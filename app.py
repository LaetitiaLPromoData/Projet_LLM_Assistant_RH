import os
from pathlib import Path
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

PERSIST_DIR = str((Path(__file__).parent / "chroma_db").resolve())  # chemin absolu pour éciter les bug 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vs = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name="cv_rag",
)
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
print("[DBG] count =", vs._collection.count())

st.title("Chatbot RAG (Gemini + Chroma)")

# LLM (Gemini) setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY non défini dans l'environnement. Veuillez l'ajouter avant d'utiliser l'application.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

# Historique
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
if prompt := st.chat_input("Pose ta question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG
    docs = retriever.invoke(prompt)
    
    # On regroupe les fichier par source
    grouped = {}
    for d in docs:
        src = d.metadata.get("source", "?")
        grouped.setdefault(src, []).append(d.page_content.replace("\n", " "))
        
    context = ""
    for src, snippets in grouped.items():
        joined = " ".join(snippets[:3])  # limiter pour éviter trop long
        context += f"\n\n### {src}\n{joined}"

    # Consigne spéciale à Gemini
    user_prompt = f"""
Tu as des extraits provenant de plusieurs CV (séparés par ### nom_du_fichier).
Réponds à la question suivante en donnant un récapitulatif clair **par candidat**.

QUESTION:
{prompt}

EXTRAITS:
{context}
"""

    answer = llm.invoke(user_prompt).content

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
