from psycopg2.extras import execute_values
import pandas as pd

# Daten in die Datenbank schreiben
def insert_into_pgvector(conn):
    df = pd.read_csv('src/data/data.csv')
    data = df.to_dict(orient="records")

    cursor = conn.cursor()

    # SQL-Befehl zum Einfügen der Daten
    insert_query = """
    INSERT INTO track (id, name, label, embedding)
    VALUES %s
    ON CONFLICT (id) DO NOTHING;
    """

    # Daten für den Insert vorbereiten
    values = []
    for record in data:
        embedding = [
            record["Danceability"],
            record["Energy"],
            record["Key"],
            record["Loudness"],
            record["Mode"],
            record["Speechiness"],
            record["Acousticness"],
            record["Instrumentalness"],
            record["Liveness"],
            record["Valence"],
            record["Tempo"],
            record["Duration_ms"],
            record["Time_Signature"]
        ]
        values.append((record["ID"], record["Name"], record["Label"], embedding))

    # Daten einfügen
    execute_values(cursor, insert_query, values)
    conn.commit()
    cursor.close()
