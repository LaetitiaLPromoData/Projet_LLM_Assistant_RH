from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from docling.document_converter import DocumentConverter
from chromadb import PersistentClient

from unstructured.partition.pdf import partition_pdf
from collections import defaultdict

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

import streamlit as st
import asyncio

# --------------------- Fonctions utilitaires ---------------------

def load_file(option=1):
    pdf_folder = Path("CV")
    all_docs = []

    for pdf_file in pdf_folder.glob("*.pdf"):
        if option == 1:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
        elif option == 2:
            raw_docs = partition_pdf(str(pdf_file), languages=["fr"])
            pages = defaultdict(list)
            for d in raw_docs:
                page_num = getattr(d, "page_number", 1)
                pages[page_num].append(str(d.text))
            docs = [
                Document(
                    page_content="\n".join(texts),
                    metadata={"filename": pdf_file.name, "page_number": page_num}
                )
                for page_num, texts in sorted(pages.items())
            ]
        elif option == 3:
            converter = DocumentConverter()
            docs_raw = converter.convert(pdf_file)
            docs_text = docs_raw.document.export_to_text()
            docs = [Document(page_content=docs_text, metadata={"filename": pdf_file.name})]
        else:
            raise ValueError("Invalid option for extraction: choose 1, 2, or 3")

        all_docs.extend(docs)
        st.info(f"{pdf_file.name} -> {len(docs)} pages chargées")

    st.success(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(data)
    st.info(f"{len(all_splits)} chunks générés")
    return all_splits


def create_vector_store(data, embeddings, persist_directory=None, collection_name="collection"):

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    vector_store.add_documents(documents=data)
    return vector_store


def answer_to_user(question, CV, llm):
    eval_prompt = ChatPromptTemplate.from_messages([
("system",
     "Tu es un assistant expert en analyse de CV et en réponse à des questions spécifiques. "
     "Ton rôle est d'utiliser uniquement les informations contenues dans les CV fournis pour répondre à la question de l'utilisateur."),
    ("human",
     "L'utilisateur pose cette question : {question}\n\n"
     "CVs récupérés : {CV}\n\n"
     "Tâches :\n"
     "1. Analyse la question pour déterminer si l'utilisateur cherche :\n"
     "   - Une seule personne → sélectionne uniquement le CV le plus pertinent.\n"
     "   - Plusieurs personnes → sélectionne exactement le nombre demandé de CVs les plus pertinents.\n"
     "2. IMPORTANT : Dans tous les cas, ta réponse doit contenir UNIQUEMENT les CVs retenus.\n"
     "   - Ne mentionne JAMAIS les CVs non retenus, directement ou indirectement (pas de phrases comme 'aucun autre CV', 'les autres candidats', 'les autres CV').\n"
     "   - Si l'utilisateur ne demande pas de comparaison, il est INTERDIT d'évoquer les CV non sélectionnés.\n"
     "3. Ne pas inventer d’informations.\n"
     "4. Si aucun CV n’est pertinent, indique-le clairement.\n\n"
     "Format de sortie attendu :\n"
     "Réponse : [réponse concise à la question, basée uniquement sur les CV retenus]\n"
     "Commentaire : [éléments précis issus uniquement des CV retenus]")
# ("system",
#      "Tu es un assistant expert en analyse de CV et en réponse à des questions spécifiques. "
#      "Ton rôle est d'utiliser uniquement les informations contenues dans les CV fournis pour répondre à la question de l'utilisateur."),
#     ("human",
#      "L'utilisateur pose cette question : {question}\n\n"
#      "CVs récupérés : {CV}\n\n"
#      "Tâches :\n"
#      "1. Parmi les CV fournis (il peut y en avoir un ou plusieurs), sélectionne uniquement le CV le plus pertinent pour répondre à la question.\n"
#      "2. Formule ta réponse en utilisant uniquement ce CV sélectionné.\n"
#      "3. Ne fais aucune comparaison entre les CVs.\n"
#      "4. Ne mentionne en aucun cas les autres CVs, directement ou indirectement (ex : 'aucun autre CV ne…', 'les autres candidats…').\n"
#      "5. Ne pas inventer d’informations.\n"
#      "6. Si le CV choisi ne contient pas d’information pertinente, précise-le explicitement.\n\n"
#      "Format de sortie :\n"
#      "Réponse : [réponse concise à la question]\n"
#      "Justification : [éléments du CV utilisés pour répondre]")
])
    chain = eval_prompt | llm
    answer = chain.invoke({"question": question, "CV": CV})
    return answer.content

def evaluate(question, reponse, info, llm):
    eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un évaluateur expert en qualité de réponses générées par des modèles de langage. "
     "Ton rôle est d'analyser la pertinence des réponses d'un LLM par rapport à une question posée."),
    ("human",
     "L'utilisateur a posé la question : {question}\n\n"
     "La base de donnée à sorties les informations {info} \n\n"
     "Le LLM a fourni la réponse : {reponse}\n\n"
     "Tâches :\n"
     "1. Évalue la réponse en fonction de :\n"
     "   - Exactitude : la réponse est-elle correcte ?\n"
     "   - Pertinence : la réponse répond-elle directement à la question ?\n"
     "   - Complétude : la réponse est-elle suffisamment détaillée et complète ?\n"
     "2. Utilise l'échelle suivante pour la pertinence :\n"
     "   0 = Hors sujet\n"
     "   1 = Partiellement pertinent\n"
     "   2 = Pertinent\n"
     "   3 = Très pertinent\n"
     "3. Justifie toujours ton évaluation avec des éléments précis de la réponse.\n\n"
     "Format de sortie attendu :\n"
     "Pertinence : [0|1|2|3] \n\n"
     "Commentaire : [explication détaillée de l'évaluation]")
])
    # Création de la requête
    chain = eval_prompt | llm
    evaluation = chain.invoke({"question": question, "reponse": reponse, "info":info})
    st.subheader("Évaluation du LLM :")
    st.write(evaluation.content)

def evaluate2(question, reponse, info):
    # Sécurisation : on transforme en string si ce n'est pas déjà le cas
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
    answer_relevancy = AnswerRelevancyMetric(model="gpt-4o-mini")

    # Mesurer chaque métrique
    answer_relevancy.measure(test_case)
    st.subheader("Évaluation DeepEval :")
    st.write(f"Precision Score: {answer_relevancy.score}")
    st.write(f"Reason: {answer_relevancy.reason}")


def get_llm_and_embeddings():
    # Ensure there's a running event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    return llm, embeddings


# --------------------- Interface Streamlit ---------------------

def app():
    st.title("Assistant LLM pour CV")
    load_dotenv()
    # Choix du loader
    loader_choice = st.selectbox("Choisir le loader :", ["PyPDF", "Unstructured", "Docling"])
    loader_map = {"PyPDF": 1, "Unstructured": 2, "Docling": 3}
    collection_map = {"PyPDF": "CV_collection_PyPDF",
                      "Unstructured": "CV_collection_Unstructured",
                      "Docling": "CV_collection_DocArray"}
    option = loader_map[loader_choice]
    collection_name = collection_map[loader_choice]

    # Entrée utilisateur pour la question
    question = st.text_input("Pose ta question :", "")

    if st.button("Poser la question") and question:
        db_path = "./chroma_langchain_db"
        llm, embeddings = get_llm_and_embeddings()
        # Vérifier si la collection existe
        client = PersistentClient(path=db_path)
        collections = client.list_collections()
        exists = any(c.name == collection_name for c in collections)

        if exists:
            st.info(f"Collection '{collection_name}' existante, chargement ...")
            vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=db_path,
                embedding_function=embeddings
            )
        else:
            st.info(f"Création de la collection '{collection_name}' ...")
            all_docs = load_file(option=option)
            all_splits = all_docs
            vector_store = create_vector_store(
                all_splits,
                embeddings,
                persist_directory=db_path,
                collection_name=collection_name
            )

        nb_CV=16

        # Recherche
        info_pertinent = vector_store.similarity_search(question, k=nb_CV)
        if not info_pertinent:
            st.warning("Aucun résultat pertinent trouvé.")
            return

        # Générer la réponse du LLM
        reponse = answer_to_user(question, info_pertinent[:nb_CV],llm)
        st.subheader("Réponse du LLM :")
        st.write(reponse)

        # Évaluer la réponse
        evaluate(question, reponse, info_pertinent[:nb_CV],llm)
        evaluate2(question, reponse, info_pertinent[:nb_CV])


if __name__ == "__main__":
    app()

