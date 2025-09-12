# ingest.py — Ingestion minimale de style étudiant pour la recherche sémantique
# Étapes : charger -> découper -> embarquer -> stocker
# Exécution : python ingest.py --docs_dir CV --persist_dir chroma_db --force_reindex

# Importation des bibliothèques nécessaires
import argparse  # Pour gérer les arguments de ligne de commande
import shutil  # Pour manipuler des fichiers et des dossiers
from pathlib import Path  # Pour gérer les chemins de fichiers de manière portable

# --- Chargeurs / Diviseur ---
# Importation des outils pour charger et diviser les documents
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Stockage vectoriel / Embeddings (packages modernes) ---
# Importation des outils pour créer des embeddings et stocker les vecteurs
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Fonction pour charger les documents depuis un dossier donné
def charger_documents(dossier_docs: Path):
    docs = []  # Liste pour stocker tous les documents chargés
    for p in dossier_docs.rglob("*"):  # Parcours récursif des fichiers dans le dossier
        if p.is_dir():  # Ignorer les sous-dossiers
            continue
        try:
            # Si le fichier est un PDF, utiliser PyPDFLoader
            if p.suffix.lower() == ".pdf":
                chargeur = PyPDFLoader(str(p))
                docs.extend(chargeur.load())  # Charger toutes les pages du PDF
            # Si le fichier est un texte ou Markdown, utiliser TextLoader
            elif p.suffix.lower() in {".txt", ".md"}:
                chargeur = TextLoader(str(p), encoding="utf-8")
                docs.extend(chargeur.load())  # Charger le contenu du fichier
        except Exception as e:
            # En cas d'erreur, afficher un avertissement et continuer
            print(f"[AVERT] Ignorer {p.name} : {e}")
    return docs  # Retourner la liste des documents chargés

# Fonction principale exécutée lorsque le script est lancé
def main():
    # Définition des arguments de ligne de commande
    ap = argparse.ArgumentParser(description="Ingestion de documents dans un index local Chroma (version étudiant).")
    ap.add_argument("--docs_dir", default="docs", help="Dossier contenant vos documents personnels (pdf/txt/md).")
    ap.add_argument("--persist_dir", default="chroma_db", help="Où stocker l'index vectoriel.")
    ap.add_argument("--chunk_size", type=int, default=800, help="Caractères par segment (ajustable).")
    ap.add_argument("--chunk_overlap", type=int, default=150, help="Chevauchement entre segments (ajustable).")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Modèle d'embedding HF.")
    ap.add_argument("--force_reindex", action="store_true", help="Supprimer l'ancien index avant d'en écrire un nouveau.")
    args = ap.parse_args()  # Analyse des arguments

    # Conversion des chemins en objets Path
    dossier_docs = Path(args.docs_dir)
    dossier_persist = Path(args.persist_dir)

    # Vérification que le dossier des documents existe
    if not dossier_docs.exists():
        raise SystemExit(f"[ERR] docs_dir introuvable : {dossier_docs.resolve()}")

    # Si l'option force_reindex est activée, supprimer l'ancien index
    if args.force_reindex and dossier_persist.exists():
        print(f"[LOG] Suppression de l'ancien index : {dossier_persist}")
        shutil.rmtree(dossier_persist)

    # Étape 1 : Charger les documents
    print("[LOG] Chargement des documents…")
    docs_bruts = charger_documents(dossier_docs)
    print(f"[LOG] {len(docs_bruts)} documents bruts chargés.")

    # Si aucun document n'a été trouvé, arrêter le programme
    if not docs_bruts:
        raise SystemExit("[ERR] Aucun document trouvé. Ajoutez des fichiers PDF ou TXT dans le dossier.")

    # Étape 2 : Découper les documents en segments
    print("[LOG] Découpage…")
    diviseur = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,  # Taille de chaque segment
        chunk_overlap=args.chunk_overlap,  # Chevauchement entre segments : utile pour le contexte
        separators=["\n\n", "\n", " ", ""],  # Séparateurs utilisés pour diviser le texte
    )
    segments = diviseur.split_documents(docs_bruts)  # Découper les documents
    print(f"[LOG] {len(segments)} segments générés (chunk_size={args.chunk_size}, overlap={args.chunk_overlap}).")

    # Étape 3 : Générer les embeddings pour les segments
    print(f"[LOG] Embedding avec : {args.model}")
    embeddings = HuggingFaceEmbeddings(model_name=args.model)  # Charger le modèle d'embedding

    # Étape 4 : Stocker les embeddings dans une base de données vectorielle
    print(f"[LOG] Écriture de l'index Chroma dans : {dossier_persist}")
    vs = Chroma.from_documents(
        documents=segments,  # Les segments à indexer
        embedding=embeddings,  # Le modèle d'embedding utilisé
        persist_directory=str(dossier_persist),  # Dossier où stocker l'index
    )
    # Forcer la matérialisation de l'index
    _ = vs.get()
    print("[OK] Ingestion terminée. Vous pouvez maintenant exécuter search.py.")

# Point d'entrée du script
if __name__ == "__main__":
    main()
