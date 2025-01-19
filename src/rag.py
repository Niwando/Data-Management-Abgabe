import os
import requests
from src.utils.connect_db import connect_to_db
from src.utils.formatting import query_to_vector
from trulens.apps.custom import instrument
from openai import OpenAI

class RAG:

    def __init__(self, model_name, limit, metric):
        self.model_name = model_name
        self.limit = limit
        self.metric = metric


    def retrieve_only(self, query) -> list:
        """
        Führt eine Similarity Search mit einer wählbaren Metrik durch und gibt die Ergebnisse als Dictionary zurück.

        :param input_vector: Der Vektor, nach dem gesucht wird (Liste von Zahlen).
        :param limit: Die maximale Anzahl von Ergebnissen.
        :param metric: Die Metrik für die Similarity Search ('cosine', 'euclidean', 'inner_product').
        :return: Ein Dictionary mit den Ergebnissen.
        """
        conn = connect_to_db()
        cursor = conn.cursor()

        # Wähle den passenden Operator basierend auf der Metrik
        if self.metric == "cosine":
            operator = "<=>"
        elif self.metric == "euclidean":
            operator = "<->"
        elif self.metric == "inner_product":
            operator = "<#>"
        else:
            raise ValueError("Ungültige Metrik. Wähle zwischen 'cosine', 'euclidean' oder 'inner_product'.")

        # Eingabevektor in ein String-Format umwandeln
        vector_str = ','.join(map(str, query))

        # SQL-Abfrage für Similarity Search
        query = f"""
        SELECT id, name, label, embedding
        FROM track
        WHERE embedding {operator} '[{vector_str}]' IS NOT NULL
        AND embedding != '[{vector_str}]'
        ORDER BY embedding {operator} '[{vector_str}]' LIMIT %s;
        """

        # Query ausführen
        cursor.execute(query, (self.limit,))
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Ergebnisse in ein Dictionary umwandeln
        results_dict = [
            {
                "id": row[0],
                "name": row[1],
                "label": row[2],
                "embedding": row[3]
            }
            for row in results
        ]

        return results_dict
    
    @instrument
    def retrieve(self, query) -> list:

        vector = query_to_vector(query)

        # Abrufen der Ergebnisse
        retrieval = self.retrieve_only(vector)

        # Initialisierung für dynamische Spaltenbreiten
        column_widths = {
            "Rank": len("Rank"),
            "Name": len("Name"),
            "Danceability": len("Danceability"),
            "Energy": len("Energy"),
            "Key": len("Key"),
            "Loudness": len("Loudness"),
            "Mode": len("Mode"),
            "Speechiness": len("Speechiness"),
            "Acousticness": len("Acousticness"),
            "Instrumentalness": len("Instrumentalness"),
            "Liveness": len("Liveness"),
            "Valence": len("Valence"),
            "Tempo": len("Tempo"),
            "Duration_ms": len("Duration_ms"),
            "Time_Signature": len("Time_Signature"),
            "Label": len("Label")
        }

        # Aktualisiere die Spaltenbreiten basierend auf den Werten
        for idx, result in enumerate(retrieval):
            column_widths["Rank"] = max(column_widths["Rank"], len(str(idx + 1)))
            column_widths["Name"] = max(column_widths["Name"], len(result["name"]))
            column_widths["Label"] = max(column_widths["Label"], len(result["label"]))

            embedding_values = result["embedding"]
            if isinstance(embedding_values, str):
                embedding_values = eval(embedding_values)  # Konvertiere den String zu einer Liste

            for key, value in zip(["Danceability", "Energy", "Key", "Loudness", "Mode", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Time_Signature"], embedding_values):
                column_widths[key] = max(column_widths[key], len(f"{round(value, 2)}"))

        # Ergebnisse für die vollständige Tabelle
        header = " | ".join(f"{key:<{column_widths[key]}}" for key in column_widths.keys())
        separator = " | ".join("-" * column_widths[key] for key in column_widths.keys())

        markdown_table = f"These are the {len(retrieval)} most similar songs to the request:\n"
        markdown_table += f"| {header} |\n"
        markdown_table += f"| {separator} |\n"

        for idx, result in enumerate(retrieval):
            embedding_values = result["embedding"]
            if isinstance(embedding_values, str):
                embedding_values = eval(embedding_values)  # Konvertiere den String zu einer Liste

            # Runde die Werte auf 2 Nachkommastellen und erstelle eine Zeile
            rounded_values = [round(value, 2) for value in embedding_values]
            row = [
                f"{idx + 1:<{column_widths['Rank']}}",
                f"{result['name']:<{column_widths['Name']}}",
                *[f"{rounded_values[i]:<{column_widths[col]}}" for i, col in enumerate(["Danceability", "Energy", "Key", "Loudness", "Mode", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Time_Signature"])],
                f"{result['label']:<{column_widths['Label']}}",
            ]
            markdown_table += f"| {' | '.join(row)} |\n"

        # Einzelne Tabellen ohne Rank, angepasst an die berechneten Spaltenbreiten
        individual_tables = []
        for idx, result in enumerate(retrieval):
            embedding_values = result["embedding"]
            if isinstance(embedding_values, str):
                embedding_values = eval(embedding_values)  # Konvertiere den String zu einer Liste

            # Runde die Werte auf 2 Nachkommastellen
            rounded_values = [round(value, 2) for value in embedding_values]
            
            # Tabellenheader und Trennlinie mit dynamischen Spaltenbreiten
            individual_table = f"This is one of the most similar songs to the request:\n"
            header = " | ".join(f"{key:<{column_widths[key]}}" for key in column_widths.keys() if key != "Rank")
            separator = " | ".join("-" * column_widths[key] for key in column_widths.keys() if key != "Rank")
            individual_table += f"| {header} |\n"
            individual_table += f"| {separator} |\n"

            # Tabellenzeile erstellen
            row = [
                f"{result['name']:<{column_widths['Name']}}",
                *[f"{rounded_values[i]:<{column_widths[col]}}" for i, col in enumerate(["Danceability", "Energy", "Key", "Loudness", "Mode", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Time_Signature"])],
                f"{result['label']:<{column_widths['Label']}}",
            ]
            individual_table += f"| {' | '.join(row)} |\n"

            individual_tables.append(individual_table)

        return individual_tables + [markdown_table]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        if len(context_str) == 0:
            return "Sorry, I couldn't find an answer to your question."
        
        if self.model_name in ["gemma2-9b-it","llama-3.1-70b-versatile","llama-3.1-8b-instant",
                                "llama-3.2-11b-vision-preview","llama-3.2-1b-preview",
                                "llama-3.2-3b-preview","llama-3.2-90b-vision-preview",
                                "llama-3.3-70b-specdec","llama-3.3-70b-versatile",
                                "llama-guard-3-8b","llama3-70b-8192","llama3-8b-8192",
                                "mixtral-8x7b-32768"]:
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.api_key = os.getenv("GROQ_API_KEY")

            if not self.api_key:
                raise ValueError("API-Key ist nicht gesetzt. Bitte .env-Datei prüfen.")

            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Imagine you are an expert in music, and your goal is to help others decide whether they would like the suggested song based on their preferences for songs they already know and whether they like them or not.\n"
                        f"We have provided context information below:\n"
                        f"{context_str}\n"
                        f"Based on this information, respond to the following question and provide a suggestion on whether the person would probably like or dislike the song:\n"
                        f"{query}\n"
                        f"Label:", 
                    }
                ]
            }

            # Anfrage senden
            response = requests.post(self.api_url, headers=self.headers, json=data)
            result = response.json()
            completion = result["choices"][0]["message"]["content"]
        
        elif self.model_name in ["gpt-4o","gpt-4o-mini","o1","o1-mini","gpt-3.5-turbo"]:
            oai_client = OpenAI()
            completion = (
                oai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    messages=[
                        {
                            "role": "user", 
                            "content": f"Imagine you are an expert in music, and your goal is to help others decide whether they would like the suggested song based on their preferences for songs they already know and whether they like them or not.\n"
                            f"We have provided context information below:\n"
                            f"{context_str}\n"
                            f"Based on this information, respond to the following question and provide a suggestion on whether the person would probably like or dislike the song:\n"
                            f"{query}\n"
                            f"Label:", 
                        }
                    ],
                )
                .choices[0]
                .message.content
            )

        if completion:
            return completion
        else:
            return "Did not find an answer."
    
    @instrument
    def query(self, query) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(
            query=query, context_str=context_str
        )
        return completion