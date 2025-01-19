# Funktion zum Erstellen der Tabelle
def create_track_table(conn):
    cursor = conn.cursor()

    try:
        # Tabelle löschen, falls sie existiert, und neu erstellen
        cursor.execute("""
        DROP TABLE IF EXISTS track;
        CREATE TABLE track (
            id TEXT PRIMARY KEY,
            name TEXT,
            label TEXT,
            embedding VECTOR(13) -- 13 ist die Anzahl der Merkmale
        );
        """)
        conn.commit()
    except Exception as e:
        print(f"Fehler beim Erstellen der Tabelle: {e}")
    finally:
        cursor.close()