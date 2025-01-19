from src.utils.connect_db import connect_to_db

def retrieval(input_vector, limit=10, metric="cosine", include_identical=True):
    """
    Führt eine Similarity Search mit einer wählbaren Metrik durch und gibt die Ergebnisse als Dictionary zurück.

    :param input_vector: Der Vektor, nach dem gesucht wird (Liste von Zahlen).
    :param limit: Die maximale Anzahl von Ergebnissen.
    :param metric: Die Metrik für die Similarity Search ('cosine', 'euclidean', 'inner_product').
    :param include_identical: Boolean, ob Lieder mit identischen Embeddings berücksichtigt werden sollen.
    :return: Ein Dictionary mit den Ergebnissen.
    """
    conn = connect_to_db()
    cursor = conn.cursor()

    # Wähle den passenden Operator basierend auf der Metrik
    if metric == "cosine":
        operator = "<=>"
    elif metric == "euclidean":
        operator = "<->"
    elif metric == "inner_product":
        operator = "<#>"
    else:
        raise ValueError("Ungültige Metrik. Wähle zwischen 'cosine', 'euclidean' oder 'inner_product'.")

    # Eingabevektor in ein String-Format umwandeln
    vector_str = ','.join(map(str, input_vector))

    # SQL-Abfrage für Similarity Search
    query = f"""
    SELECT id, name, label, embedding
    FROM track
    WHERE embedding {operator} '[{vector_str}]' IS NOT NULL
    """
    
    # Füge Bedingung hinzu, um identische Embeddings zu filtern
    if not include_identical:
        query += f" AND embedding != '[{vector_str}]'"

    query += f" ORDER BY embedding {operator} '[{vector_str}]' LIMIT %s;"

    # Query ausführen
    cursor.execute(query, (limit,))
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
