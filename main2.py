import os
import glob
import getpass
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from unstructured.partition.pdf import partition_pdf
from docling.document_converter import DocumentConverter
import asyncio
from chromadb import PersistentClient
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
import json


# ------------------------
# Loaders (inchangés)
# ------------------------
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


# ------------------------
# Split
# ------------------------
def text_split(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(data)


# ------------------------
# Vector store
# ------------------------
def get_vector_store(choice, embeddings, persist_directory="./chroma_langchain_db"):
    collections = {
        "1": ("collection_1", load_file_pypdf),
        "2": ("collection_2", load_file_unstructured),
        "3": ("collection_3", load_file_docling),
    }

    if choice not in collections:
        choice = "3"  # Docling par défaut

    collection_name, loader_func = collections[choice]

    #Vérifie si la bd existe
    client = PersistentClient(path = persist_directory)
    db_collection = client.list_collections()
    exists = any(c.name==collection_name for c in db_collection)

    if exists:
        st.info(f"Collection {collection_name} existe, chargement...")
        vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    else:
        st.info(f"Création de la collection : {collection_name}")
        docs = loader_func()
        splits = docs
        vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
        vector_store.add_documents(splits)

    return vector_store


# ------------------------
# RAG - réponse à une question
# ------------------------
def answer_user(question, vector_store, llm, k=16):
    results = vector_store.similarity_search(question, k=k)
    if not results:
        return {"llm_response": "Aucun CV pertinent trouvé.", "top_docs": []}

    context = ""
    for i, doc in enumerate(results, 1):
        context += f"[CV {i} - Source: {doc.metadata.get('source','inconnue')}]\n{doc.page_content}\n\n"

    prompt = f"""
Rôle : Tu es un assistant RH expert en analyse de candidatures et en sélection de profils pertinents.

Contexte :
- Question posée : "{question}"
- Voici les CV les plus pertinents extraits de la base de données :
{context}

Objectif principal :
Analyser les CV fournis afin de répondre à la question de manière précise, argumentée et sourcée.

Tâches :
1. Identifier le CV le plus pertinent et préciser le nom du candidat si disponible.
2. Expliquer en quelques points pourquoi ce CV est le plus pertinent (expérience, compétences, formation, etc.).
3. Fournir une réponse synthétique et structurée à la question posée, en t’appuyant uniquement sur les CV.
4. Indiquer les sources utilisées (références aux CV ou passages spécifiques).

Contraintes :
- Ta réponse doit être claire, concise et professionnelle.
- Tu ne dois pas inventer d’informations non présentes dans les CV.
- Structure ta réponse en sections numérotées ou avec des titres.

Format attendu :
1. Candidat le plus pertinent
2. Justification
3. Réponse synthétique
4. Sources
"""

    response = llm.predict(prompt)
    return {"llm_response": response, "top_docs": results}

def get_llm_and_embeddings():
    # Ensure there's a running event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    return llm, embeddings

"""Évaluer la pertinence"""

def evaluate_answer(question, answer):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

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
          "Format de sortie attendu :\n"
          "pertinence: 2\n\n"
          "commentaire: Compétence partielle.")
    ])

    chain = prompt_template | llm
    evaluation = chain.invoke({"question": question, "answer": answer})
    st.write(evaluation.content)

def evaluate2(question, reponse, info):

    if hasattr(reponse, "content"):  
        reponse = reponse.content
    elif isinstance(reponse, list):
        reponse = "\n\n".join([str(r) for r in reponse])
    elif not isinstance(reponse, str):
        reponse = str(reponse)

    if hasattr(info, "page_content"):
        info = info.page_content
    elif isinstance(info, list):
        info = "\n\n".join([str(i) for i in info])
    elif not isinstance(info, str):
        info = str(info)

    test_case = LLMTestCase(
        input = question,
        actual_output= reponse,
        retrieval_context=[info]
    )

    # Initialiser les métriques
    api_key = os.getenv("GOOGLE_API_KEY")
    model = GeminiModel(model_name="gemini-2.5-flash-lite", api_key=api_key)

    answer_relevancy = AnswerRelevancyMetric(model=model)

    # Mesurer chaque métrique
    answer_relevancy.measure(test_case)
    st.subheader("Évaluation DeepEval :")
    st.write(f"Precision Score: {answer_relevancy.score}")
    st.write(f"Reason: {answer_relevancy.reason}") 

# helper pour extraire le texte si la réponse LLM n'est pas strictement une string
def _to_text(llm_response):
    # si c'est un objet LangChain avec .content
    if hasattr(llm_response, "content"):
        return llm_response.content
    # si c'est une chaîne déjà
    if isinstance(llm_response, str):
        return llm_response
    # fallback
    return str(llm_response)    

# ------------------------
# Interface Streamlit
# ------------------------
def app():
    load_dotenv()
    st.title("Assistant RH avec LLM pour analyse CV")
    st.write("Analyse automatique des CV et réponses aux questions RH (RAG + LLM).")

    loader_choice = st.selectbox(
        "Choisissez la méthode de chargement des CV :",
        ["PyPDF", "Unstructured", "Docling"],
        index=2  # Docling par défaut 
    )

    mapping = {"PyPDF": "1", "Unstructured": "2", "Docling": "3"}
    # choice = mapping.get(loader_choice, "3")
    choice = mapping[loader_choice]

    llm, embeddings = get_llm_and_embeddings()

    # Construire ou charger la base vectorielle
    st.info("Chargement/initialisation de la base vectorielle en cours...")
    vector_store = get_vector_store(choice, embeddings=embeddings)
    st.success("Base vectorielle prête")

        # initialise session_state (une seule fois)
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None
    if "last_docs" not in st.session_state:
        st.session_state["last_docs"] = None
    if "last_evaluation" not in st.session_state:
        st.session_state["last_evaluation"] = None

    st.header("Pose ta question RH")

    # on affiche la question précédente par défaut (pratique)
    question_input = st.text_input("Question :", value=st.session_state["last_question"])

    # col1, col2 = st.columns(2)

    # with col1:
    if st.button("Obtenir réponse"):
        if not question_input.strip():
            st.warning("Veuillez saisir une question.")
        else:
            with st.spinner("Recherche et génération en cours..."):
                # utilise ta fonction answer_user existante
                answer = answer_user(question_input, vector_store, llm)  # doit renvoyer dict {"llm_response", "top_docs"}
                # extraire texte propre de la réponse LLM
                resp_text = _to_text(answer["llm_response"])
                # sauvegarder dans la session
                st.session_state["last_question"] = question_input
                st.session_state["last_answer"] = resp_text
                st.session_state["last_docs"] = answer.get("top_docs", [])
                st.session_state["last_evaluation"] = None  # reset ancienne éval
            st.subheader("Réponse LLM")
            st.write(resp_text)

    # with col2:
    if st.button("Évaluer la pertinence"):
        # on s'assure qu'on a bien une réponse générée avant d'évaluer
        if not st.session_state["last_answer"]:
            st.warning("Génère d'abord une réponse (clique sur 'Obtenir réponse') avant d'évaluer.")
        else:
            with st.spinner("Évaluation en cours..."):
                # appelle ta fonction evaluate_answer qui renvoie une string (format attendu)
                evaluate_answer(st.session_state["last_question"], st.session_state["last_answer"])
                evaluate2(st.session_state["last_question"], st.session_state["last_answer"], st.session_state["last_docs"])
                
   
          



if __name__ == "__main__":
    app()