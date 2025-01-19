
def vector_to_query(vector):
    # Initialisierung für dynamische Spaltenbreiten
    column_widths = {
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
        "Time_Signature": len("Time_Signature")
    }

    # Werte auf 2 Nachkommastellen runden und als Strings formatieren
    rounded_values = [f"{round(value, 2)}" for value in vector]

    # Aktualisiere die Spaltenbreiten basierend auf den Werten im Eingabevektor
    for key, value in zip(column_widths.keys(), rounded_values):
        column_widths[key] = max(column_widths[key], len(value))

    # Tabelle erstellen
    query = "Evaluate whether I would like this song and respond with only the label 'like' or 'dislike', nothing else:\n"
    header = " | ".join(f"{key:<{column_widths[key]}}" for key in column_widths.keys())
    separator = " | ".join("-" * column_widths[key] for key in column_widths.keys())
    query += f"| {header} |\n"
    query += f"| {separator} |\n"

    # Tabellenzeile mit dynamischen Breiten füllen
    row = [
        f"{rounded_values[i]:<{column_widths[col]}}" for i, col in enumerate(column_widths.keys())
    ]
    query += f"| {' | '.join(row)} |\n"

    return query

def query_to_vector(query):
    # Zerlege die Tabelle in Zeilen
    lines = query.strip().split("\n")
    
    # Finde die Zeile, die den Vektor enthält (letzte Datenzeile)
    for line in lines:
        if line.startswith("| 0."):  # Erkennung einer Zahlenzeile
            # Entferne die Spaltenrahmen ("|") und bereinige Leerzeichen
            values = line.strip("|").split("|")
            # Konvertiere Werte in eine Liste von Floats
            query = [float(value.strip()) for value in values]
    
    return query