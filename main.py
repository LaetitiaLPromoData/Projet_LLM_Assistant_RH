

import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from unstructured.partition.pdf import partition_pdf
from docling.document_converter import DocumentConverter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate


# --- Loaders ---
def load_file_pypdf(folder_path="./CV"):
    all_documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_documents.extend(docs)
    return all_documents

def load_file_unstructured(folder_path="./CV"):
    all_documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        elements = partition_pdf(filename=file_path, languages=["fr"])
        text = "\n".join(el.text for el in elements if hasattr(el, "text") and el.text)
        all_documents.append(
            Document(page_content=text, metadata={"source": os.path.basename(file_path)})
        )
    return all_documents

def load_file_docling(folder_path="./CV"):
    all_documents = []
    converter = DocumentConverter()
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        result = converter.convert(file_path)
        if isinstance(result, tuple):  # compatibilité selon version
            result = result[0]
        text = result.document.export_to_text()
        all_documents.append(
            Document(page_content=text, metadata={"source": os.path.basename(file_path)})
        )
    return all_documents


# --- Split ---
def text_split(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(data)


# --- Vector store ---
def get_vector_store(choice, persist_directory="./chroma_langchain_db"):
    collections = {
        "PyPDF": ("collection_1", load_file_pypdf),
        "Unstructured": ("collection_2", load_file_unstructured),
        "Docling": ("collection_3", load_file_docling),
    }

    collection_name, loader_func = collections[choice]
    collection_path = os.path.join(persist_directory, collection_name)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if os.path.exists(collection_path):
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    docs = loader_func()
    splits = text_split(docs)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.add_documents(splits)
    return vector_store


# --- Répondre à une question avec RAG ---
def answer_user(question, vector_store, llm, k=3):
    results = vector_store.similarity_search(question, k=k)
    if not results:
        return {"llm_response": "Aucun CV pertinent trouvé.", "top_docs": []}

    context = ""
    for i, doc in enumerate(results, 1):
        context += f"[CV {i} - Source: {doc.metadata.get('source','inconnue')}]\n{doc.page_content}\n\n"

    prompt = f"""
Tu es un assistant RH expert. 
Question posée : "{question}"

Voici les CV les plus pertinents extraits de la base de données :
{context}

Tâches :
1. Identifier le CV le plus pertinent et donner le nom du candidat si possible.
2. Fournir une réponse synthétique à la question, en te basant sur les CV.
3. Inclure les sources utilisées dans ta réponse.

Réponds de manière structurée.
"""
    response = llm.predict(prompt)

    return {
        "llm_response": response,
        "top_docs": [{"metadata": doc.metadata, "content": doc.page_content[:800]} for doc in results]
    }


# --- Évaluer la pertinence ---
def evaluate_answer(question, answer):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    prompt_template = ChatPromptTemplate.from_messages([
         ("system", "Tu es un expert en recrutement. Analyse les CVs et évalue leur pertinence."),
         ("human",
          "Question posée : {question}\n\n"
          "Réponse générée : {answer}\n\n"
          "Évalue selon :\n"
          "0 = Pas pertinent\n"
          "1 = Peu pertinent\n"
          "2 = Pertinent\n"
          "3 = Très pertinent\n\n"
          "Format JSON :\n"
          "{{\"pertinence\": 2, \"commentaire\": \"Compétence partielle.\"}}")
    ])

    chain = prompt_template | llm
    evaluation = chain.invoke({"question": question, "answer": answer})
    return evaluation.content


# --- Interface Streamlit ---
def main():
    st.set_page_config(page_title="Assistant RH", layout="wide")
    st.title("Assistant RH - Recherche dans les CV")

    load_dotenv()

    # Choix du loader
    loader_choice = st.sidebar.radio(
        "Méthode de chargement des CV",
        ["Docling", "PyPDF", "Unstructured"],
        index=0
    )

    # Initialisation du vector store
    with st.spinner("Chargement des CV..."):
        vector_store = get_vector_store(loader_choice)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Zone de question
    question = st.text_area("Posez une question RH :")

    if st.button("Rechercher et répondre"):
        if question.strip():
            with st.spinner("Recherche et génération de la réponse..."):
                answer = answer_user(question, vector_store, llm)

            st.subheader("Réponse de l'assistant RH")
            st.write(answer["llm_response"])

            st.subheader("CV sources utilisés")
            for doc in answer["top_docs"]:
                with st.expander(f"Source : {doc['metadata'].get('source','inconnue')}"):
                    st.write(doc["content"])

            if st.button(" Évaluer la réponse"):
                with st.spinner("Évaluation en cours..."):
                    evaluation = evaluate_answer(question, answer["llm_response"])
                st.subheader("📊 Évaluation de la réponse")
                st.json(evaluation)
        else:
            st.warning("Veuillez poser une question.")


if __name__ == "__main__":
    main()


