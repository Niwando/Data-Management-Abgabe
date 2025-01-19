import os
import time
import json
import argparse
import textwrap
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
from trulens.providers.openai import OpenAI

from src.utils.connect_db import connect_to_db
from src.setup.pgvector_extentsion import enable_pgvector_extension
from src.setup.create_table import create_track_table
from src.evaluation.similarity.print_similarity import print_similarity
from src.setup.data_to_pgvector import insert_into_pgvector
from src.evaluation.trulens.evaluation import evaluate_trulens
from src.evaluation.similarity.metrics import similarity
from src.data.meta_results import generate_meta_results
from src.rag import RAG

from dotenv import load_dotenv

load_dotenv()

def setup():
    """
    Führt das Setup durch: aktiviert die Extension, erstellt die Tabelle und lädt die Daten.
    Mit Ladebalken für Fortschritt.
    """
    conn = connect_to_db()
    steps = [
        enable_pgvector_extension,
        create_track_table,
        insert_into_pgvector
    ]

    with tqdm(total=len(steps), desc="Setup-Prozess", unit="Schritt") as pbar:
        for func in steps:
            time.sleep(0.1)  # Simuliert eine kleine Verzögerung
            func(conn)  # Führt die Funktion aus
            pbar.update(1)  # Aktualisiert den Ladebalken

    conn.close()
    print("Setup erfolgreich abgeschlossen.")

def evaluation_from_data(data_size, similarity_search_type, top_k, model_name="llama-3.3-70b-versatile", eval_model="gpt-3.5-turbo"):

    data = pd.read_csv("src/data/preprocessed/data.csv")
    if data_size != "full":
        data = data.sample(int(data_size), random_state=1).reset_index(drop=True)

    vector_columns = [
        "Danceability", "Energy", "Key", "Loudness", "Mode", "Speechiness",
        "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", 
        "Duration_ms", "Time_Signature"
    ]

    data['Vector'] = data[vector_columns].apply(lambda row: row.tolist(), axis=1)

    ground_truth = []
    for _, row in data.iterrows():
        ground_truth.append({
            "query": row["Vector"], 
            "expected_response": row["Label"],
        })

    rag = RAG(model_name=model_name, limit=top_k, metric=similarity_search_type)
    provider = OpenAI(model_engine=eval_model)

    results = {}
    with tqdm(total=len(data), desc="Evaluation", unit="Step") as pbar:
        for index in range(len(data)):
            sub_df = data.iloc[[index]].reset_index(drop=True)
            input_vector = sub_df[vector_columns].iloc[0].tolist()
            
            record_id = sub_df["ID"].iloc[0]
            label = sub_df["Label"].iloc[0]

            similar_tracks = rag.retrieve_only(query=input_vector)

            similarity_results = similarity(similar_tracks=similar_tracks, input_vector=input_vector)

            evaluation = evaluate_trulens(provider=provider, input_vector=input_vector, rag=rag, ground_truth=label)
            evaluation["context"] = {}
            
            for result in similarity_results:
                rank = result["Rang"]
                evaluation["context"][rank] = {
                    "id": result["ID"],
                    "cosine similarity": result["Cosine Similarity"],
                    "euclidean distance": result["Euclidean Distance"],
                    "manhattan distance": result["Manhattan Distance"],
                    "mahalanobis distance": result["Mahalanobis Distance"],
                    "pearson correlation": result["Pearson Correlation"]
                }

            # Speichern der Evaluation mit der ID als Schlüssel
            results[record_id] = evaluation
            pbar.update(1)

    overall_results = {
        "meta results": generate_meta_results(results),
        "results": results
    }

    directory = "src/data/results"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    # Specify the file name
    file_name = "src/data/results/overall_results.json"

    # Save as JSON
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(overall_results, json_file, indent=4)

    print(f"Overall results have been saved to {file_name}.")

    

def evaluation_from_user(stage, input, similarity_search_type, top_k, model_name="llama-3.3-70b-versatile", eval_model="gpt-3.5-turbo"):
    input_vector = json.loads(input)

    rag = RAG(model_name=model_name, limit=top_k, metric=similarity_search_type)

    if stage == "retrieval" or stage == "all":
        # Ähnlichkeitssuche durchführen
        similar_tracks = rag.retrieve_only(query=input_vector)

        similarity_results = similarity(similar_tracks=similar_tracks, input_vector=input_vector)

        print_similarity(similar_tracks=similar_tracks, metrics_results=similarity_results)

    if stage == "generation" or stage == "all":

        rag = RAG(model_name=model_name, limit=top_k, metric=similarity_search_type)
        provider = OpenAI(model_engine=eval_model)
        
        evaluation = evaluate_trulens(provider=provider, input_vector=input_vector, rag=rag, ground_truth=None)
        # Textwrapping für die langen Begründungen
        def wrap_text(text, width=50):
            return "\n".join(textwrap.wrap(text, width))

        # Tabelle vorbereiten
        headers = [
            "Response",
            "Groundedness",
            "Relevance Score",
            "Relevance Reason",
            "Context Relevance Score",
            "Context Relevance Reason",
        ]

        rows = [
            [
                evaluation["response"],
                round(evaluation["groundedness"], 2),
                evaluation["relevance"]["score"],
                wrap_text(evaluation["relevance"]["reasons"]["reason"]),
                evaluation["context_relevance"]["score"],
                wrap_text(evaluation["context_relevance"]["reasons"]["reason"]),
            ]
        ]

        # Tabelle erstellen und ausgeben
        table = tabulate(rows, headers=headers, tablefmt="grid")
        print(table)


def main():
    parser = argparse.ArgumentParser(description="Verwaltungsskript für Datenbankaktionen")
    subparsers = parser.add_subparsers(dest='type', required=True, help="Verfügbare Aktionen")

    # Subparser für 'setup'
    setup_parser = subparsers.add_parser('setup', help="Setup ausführen")

    # Subparser für 'eval-data'
    eval_data_parser = subparsers.add_parser('eval-data', help="Datenevaluation durchführen")
    eval_data_parser.add_argument('--data-size', type=str, required=True,
                                  help="Datengröße, gib entweder eine bestimmte größe an oder 'full' für den ganzen Datensatz.")
    eval_data_parser.add_argument('--similarity-search', type=str, required=True, choices=['cosine', 'euclidean', 'inner-product'],
                                  help="Similarity-Search Art: 'cosine', 'euclidean' oder 'inner-product'.")
    eval_data_parser.add_argument('--top-k', type=str, required=True,
                                  help="Die Anzahl der zurückgegebenen Dokumente.")
    eval_data_parser.add_argument('--model', type=str, required=True, choices=["gpt-4o","gpt-4o-mini","o1","o1-mini","gpt-3.5-turbo",
                                                                "gemma2-9b-it","llama-3.1-70b-versatile","llama-3.1-8b-instant",
                                                                "llama-3.2-11b-vision-preview","llama-3.2-1b-preview",
                                                                "llama-3.2-3b-preview","llama-3.2-90b-vision-preview",
                                                                "llama-3.3-70b-specdec","llama-3.3-70b-versatile",
                                                                "llama-guard-3-8b","llama3-70b-8192","llama3-8b-8192",
                                                                "mixtral-8x7b-32768"],
                                  help="Modellname. Nur erforderlich für 'all'.")
    eval_data_parser.add_argument('--eval-model', type=str, required=True, choices=["gpt-4o","gpt-4o-mini","o1","o1-mini","gpt-3.5-turbo"],
                                  help="Modellname für das Evaluierungsmodell. Nur erforderlich für 'all'.")

    eval_user_parser = subparsers.add_parser('eval-user', help="Benutzerevaluation durchführen")
    eval_user_parser.add_argument('--stage', type=str, required=True, choices=['retrieval', 'generation', 'all'],
                                  help="Evaluationsstufe: 'retrieval', 'generation' oder 'all'.")
    eval_user_parser.add_argument('--similarity-search', type=str, required=True, choices=['cosine', 'euclidean', 'inner-product'],
                                  help="Similarity-Search Art: 'cosine', 'euclidean' oder 'inner-product'.")
    eval_user_parser.add_argument('--top-k', type=int, required=True,
                                  help="Die Anzahl der zurückgegebenen Dokumente.")
    eval_user_parser.add_argument('--model', type=str, choices=["gpt-4o","gpt-4o-mini","o1","o1-mini","gpt-3.5-turbo",
                                                                "gemma2-9b-it","llama-3.1-70b-versatile","llama-3.1-8b-instant",
                                                                "llama-3.2-11b-vision-preview","llama-3.2-1b-preview",
                                                                "llama-3.2-3b-preview","llama-3.2-90b-vision-preview",
                                                                "llama-3.3-70b-specdec","llama-3.3-70b-versatile",
                                                                "llama-guard-3-8b","llama3-70b-8192","llama3-8b-8192",
                                                                "mixtral-8x7b-32768"],
                                  help="Modellname. Nur erforderlich für 'generation' und 'all'.")
    eval_user_parser.add_argument('--eval-model', type=str, choices=["gpt-4o","gpt-4o-mini","o1","o1-mini","gpt-3.5-turbo"],
                                  help="Modellname für das Evaluierungsmodell. Nur erforderlich für 'generation' und 'all'.")
    eval_user_parser.add_argument('--input', type=str, required=True,
                                  help="Eingaben für die Evaluation. Gebe folgende Werte zwischen 0 und 1 an: [Akustizität, Tanzbarkeit, Dauer, Energie, Instrumentalität, Tonart, Lebendigkeit, Lautstärke, Modus, Sprachanteil, Tempo, Taktart, Valenz].")

    # Argumente parsen
    args = parser.parse_args()

    # Aktion basierend auf 'type' ausführen
    if args.type == "setup":
        setup()
    elif args.type == "eval-data":
        evaluation_from_data(args.data_size, args.similarity_search, args.top_k, args.model, args.eval_model)
    elif args.type == "eval-user":
        if args.stage == "retrieval":
            evaluation_from_user(args.stage, args.input, args.similarity_search, args.top_k)
        elif args.stage == "generation" or args.stage == "all":
            evaluation_from_user(args.stage, args.input, args.similarity_search, args.top_k, args.model, args.eval_model)
    else:
        print(f"Unbekannter Typ: {args.type}. Erlaubte Typen: 'setup', 'eval-user', 'eval-data'.")

if __name__ == "__main__":
    main()