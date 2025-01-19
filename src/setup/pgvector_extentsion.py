# Funktion, um die Erweiterung zu aktivieren
def enable_pgvector_extension(conn):
    cursor = conn.cursor()

    try:
        # Erweiterung aktivieren
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception as e:
        print(f"Fehler beim Aktivieren der Erweiterung: {e}")
    finally:
        cursor.close()