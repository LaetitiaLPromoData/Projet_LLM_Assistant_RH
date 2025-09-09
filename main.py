import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document
import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Charge les variables définies dans .env
def load_data_ocr(folder_path, lang="fra"):
    docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    for file_name in pdf_files:
        file_path = os.path.join(folder_path, file_name)
        pages = convert_from_path(file_path)  # PDF → liste d’images

        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page, lang=lang)  # OCR
            if text.strip():  # éviter les pages vides
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "filename": file_name,
                            "page": i + 1
                        }
                    )
                )
    return docs

def load_data(folder_path):

    from langchain_community.document_loaders import UnstructuredPDFLoader


    pdf_files = glob.glob(f"{folder_path}/*.pdf")

    docs = []

    # Charger chaque PDF et ajouter le nom de fichier comme métadonnée
    for file_path in pdf_files:
        
        loader = PyPDFLoader(file_path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["filename"] = os.path.basename(file_path)  # ajoute juste le nom du fichier
        docs.extend(loaded_docs)
    return docs

def split_docs(docs):
    # Split en chunks
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    '''
    all_splits = docs
    return all_splits


def get_vectorstore(docs=None, persist_dir="chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if docs:  
        # Si on fournit des docs → on recrée la DB et on la sauvegarde
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()  # très important : écrit sur disque
    else:
        # Sinon, on recharge une DB déjà existante
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    return vectordb

def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Chat LLM Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # tu peux tester aussi "gemini-1.5-pro" ou "gemini-1.5-flash"
        temperature=0,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # possible aussi: "map_reduce" ou "refine"
        return_source_documents=True
    )
    return qa_chain

  
def main():
    load_dotenv()

    persist_dir = "chroma_db"

    if not os.path.exists(persist_dir):
        print("➡️ Création de la base vectorielle...")
        all_docs = load_data_ocr("CV")   # ou load_data("CV")
        all_splits = split_docs(all_docs)
        vectordb = get_vectorstore(docs=all_splits, persist_dir=persist_dir)
    else:
        print("✅ Rechargement de la base vectorielle existante")
        vectordb = get_vectorstore(docs=None, persist_dir=persist_dir)

    # Construire la QA chain
    qa = build_qa_chain(vectordb)

    # Exemple de requête
    query = input("Pose ta question sur les CV: ")
    result = qa.invoke(query)

    print("✅ Réponse générée:\n", result["result"])
    print("\n📖 Sources utilisées:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['filename']} (page {doc.metadata['page']})")

if __name__ == "__main__":

    main()
