# search.py — Script de recherche sémantique minimal utilisant Chroma et HuggingFace
# Ce script permet de rechercher des documents similaires à une requête donnée dans un index local Chroma.
# On peut choisir entre deux types de recherche : "similarity" (basé sur la similarité) ou "mmr" (Maximal Marginal Relevance pour plus de diversité).

import argparse  # Permet de gérer les arguments passés en ligne de commande
from pathlib import Path  # Permet de manipuler les chemins de fichiers et dossiers

# Importation des modules nécessaires pour Chroma et les embeddings
from langchain_chroma import Chroma  # Chroma est utilisé pour gérer l'index des embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Permet de générer des embeddings avec un modèle HuggingFace

def main():
    # Création d'un parser pour gérer les arguments en ligne de commande
    ap = argparse.ArgumentParser(description="Recherche sémantique sur un index local Chroma (version étudiante).")
    ap.add_argument("--persist_dir", default="chroma_db", help="Répertoire où l'index est sauvegardé.")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Modèle HuggingFace pour les embeddings.")
    ap.add_argument("--query", required=True, help="La requête ou question à rechercher.")
    ap.add_argument("--k", type=int, default=4, help="Nombre de résultats à afficher.")
    ap.add_argument("--search_type", choices=["similarity", "mmr"], default="similarity", help="Type de recherche : 'similarity' ou 'mmr'.")
    ap.add_argument("--fetch_k", type=int, default=20, help="Nombre de candidats pour MMR (utilisé uniquement avec --search_type mmr).")
    args = ap.parse_args()  # Analyse les arguments fournis par l'utilisateur

    # Vérification que le répertoire de l'index existe
    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise SystemExit(f"[ERR] Index introuvable : {persist_dir.resolve()} — exécutez d'abord ingest.py.")

    print(f"[LOG] Chargement de l'index depuis : {persist_dir}")
    # Chargement des embeddings avec le modèle spécifié
    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    # Chargement de l'index Chroma
    vs = Chroma(embedding_function=embeddings, persist_directory=str(persist_dir))

    # Si l'utilisateur choisit le mode "similarity"
    if args.search_type == "similarity":
        # Recherche des documents les plus similaires à la requête
        results = vs.similarity_search_with_score(args.query, k=args.k)
        print(f"\n=== Top {args.k} (similarité) ===\n")
        for i, (doc, score) in enumerate(results, 1):
            # Extraction des métadonnées et contenu du document
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            preview = (doc.page_content or "").strip().replace("\n", " ")
            if len(preview) > 200:  # Limite la longueur de l'aperçu
                preview = preview[:200] + "…"
            # Affichage des résultats avec le score
            print(f"{i:>2}. score={score:.4f} | {src}#p{page}\n    {preview}\n")
    else:
        # Si l'utilisateur choisit le mode "mmr" (Maximal Marginal Relevance)
        retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": args.k, "fetch_k": args.fetch_k, "lambda_mult": 0.5})
        docs = retriever.invoke(args.query)
        print(f"\n=== Top {args.k} (MMR diversifié) ===\n")
        for i, doc in enumerate(docs, 1):
            # Extraction des métadonnées et contenu du document
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            preview = (doc.page_content or "").strip().replace("\n", " ")
            if len(preview) > 200:  # Limite la longueur de l'aperçu
                preview = preview[:200] + "…"
            # Affichage des résultats
            print(f"{i:>2}. {src}#p{page}\n    {preview}\n")

# Point d'entrée du script
if __name__ == "__main__":
    main()
