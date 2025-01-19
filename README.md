# Evaluierung von RAG-Ergebnissen

## 🔍 Projektübersicht

Dieses Projekt befasst sich mit der Evaluierung von Retrieval-Augmented Generation (RAG) mithilfe verschiedener Evaluierungsmetriken. Dabei kommt insbesondere die Python-Bibliothek [TruLens](https://www.trulens.org) zum Einsatz. Die Evaluierung basiert auf Musikdaten, wobei Nutzer entweder einen vorhandenen Datensatz verwenden oder ihre eigenen Song-Eigenschaften testen können.

---

## 📂 Ordnerstruktur

Im Root-Verzeichnis befinden sich mehrere zentrale Bereiche: **data** enthält die Rohdaten `raw/`, vorverarbeitete Datensätze `preprocessed/` und Ergebnisse `results/`. **evaluation** umfasst Module zur Bewertung, wie Ähnlichkeitsmetriken (`similarity/`) und Trulens-Tools (`trulens/`). **setup** bietet Skripte zur Umgebungseinrichtung, während **utils** Hilfsfunktionen wie Datenbankverbindungen und Formatierungswerkzeuge bereitstellt. Die Hauptlogik für RAG befindet sich in `rag.py`, und alles ist über die Hauptdatei `run.py` verbunden.

Die Ordnerstruktur ist wie folgt aufgebaut:
```plaintext
.venv/                  
src/                    
├── data/               
│   ├── raw/            
│   │   ├── data.csv    
│   │   ├── dislike.json
│   │   ├── good.json
│   │   ├── no.py
│   │   ├── yes.py
│   ├── preprocessed/ 
│   │   ├── covariance_matrix.csv
│   │   ├── data.csv
│   ├── results/ 
│   │   ├── overall_results.json     
│   ├── data_preprocessing.py
│   ├── meta_results.py
├── evaluation/         
│   ├── similarity/     
│   │   ├── cosine.py
│   │   ├── euclidean.py
│   │   ├── mahalanobis.py
│   │   ├── manhattan.py
│   │   ├── metrics.py
│   │   ├── pearson.py
│   │   ├── print_similarity.py
│   ├── trulens/        
│       ├── evaluation.py
├── setup/              
│   ├── create_table.py
│   ├── data_to_pgvector.py
│   ├── pgvector_extension.py
├── utils/              
│   ├── connect_db.py
│   ├── formatting.py
│   ├── retrieval.py
├── rag.py
.env                               
README.md               
requirements.txt        
run.py                          
```
---

## 🎧 Datengrundlage

Die Daten stammen aus dem [Spotify Recommendation Dataset](https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation). Der Datensatz umfasst Informationen wie Danceability, Energy, Tempo und andere musikalische Merkmale.

### Zusammenfassung der Merkmale
- **Danceability**: Eignung eines Songs zum Tanzen.
- **Energy**: Wahrgenommene Intensität und Aktivität.
- **Key**: Musikalischer Schlüssel.
- **Loudness**: Lautstärke in Dezibel.
- **Mode**: Modus (Dur oder Moll).
- **Speechiness**: Anteil an gesprochenem Wort.
- **Instrumentalness**: Wahrscheinlichkeit, dass ein Song instrumental ist.
- **Valence**: Positivität der musikalischen Stimmung.

Weitere Informationen zu den Attributen finden Sie in der [Spotify-API-Dokumentation](https://developer.spotify.com/documentation/web-api/).

---

## 📊 Evaluierungsmetriken

### Retrieval-basierte Metriken
- **Cosine Similarity**: Misst die Ähnlichkeit zwischen zwei Vektoren.
- **Euclidean Distance**: Abstandsmaß in einem n-dimensionalen Raum.
- **Manhattan Distance**: Absolute Distanz zwischen Vektorkoordinaten.
- **Mahalanobis Distance**: Skaliertes Abstandsmaß.
- **Pearson Correlation**: Korrelation zwischen zwei Variablen.

### Generations-basierte Metriken
- **Correctness**: Korrektheit der generierten Antworten.
- **Groundedness**: Grad, zu dem generierte Antworten durch die Quelle gestützt werden.
- **Relevance**: Relevanz der Antwort (mit Begründungen).
- **Context Relevance**: Passgenauigkeit der Antwort im Kontext (mit Begründungen).

---

## 🚀 Erste Schritte

### Voraussetzungen
- **Pflicht**: `OPENAI_API_KEY`
- **Optional**: `GROQ_API_KEY` für zusätzliche Generierungsmodelle.

### Installation
1. **Virtuelle Umgebung aufsetzen**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Abhängigkeiten installieren**:
   ```bash
   pip install -r requirements.txt
   ```

3. **PostgreSQL und pgVector installieren** (für macOS):
    ```bash
    brew install postgresql@17
    brew install libpq
    echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc
    echo 'export LDFLAGS="-L/opt/homebrew/opt/libpq/lib"' >> ~/.zshrc
    echo 'export CPPFLAGS="-I/opt/homebrew/opt/libpq/include"' >> ~/.zshrc
    echo 'export PKG_CONFIG_PATH="/opt/homebrew/opt/libpq/lib/pkgconfig"' >> ~/.zshrc

    source ~/.zshrc
    brew services start postgresql@17
    ```

---

## 🔧 Datenaufbereitung
Die Daten wurden bereits vorverarbeitet. Falls eine erneute Verarbeitung erforderlich ist:
```bash
python src/data_preprocessing.py
```

---

## 🔢 Setup

1. **.env-Datei einrichten**:
   ```
   DB_NAME=<name>
   DB_USER=<user>
   DB_PASSWORD=<your_password>
   DB_HOST=<your_host>
   DB_PORT=<your_port>
   OPENAI_API_KEY=<your_api_key>
   GROQ_API_KEY=<optional>
   ```

2. **Vektordatenbank vorbereiten**:
   ```bash
   python run.py setup
   ```

---

## 🌐 Nutzung

### Evaluierung vorhandener Daten

Evaluieren Sie RAG basierend auf einem vorhandenen Datensatz:
```bash
python run.py eval-data --data-size <Datengröße> --similarity-search <Methode> --top-k <Anzahl> --model <Generierungsmodell> --eval-model <Evaluierungsmodell>
```
**Parameterbeschreibung**
- `--data-size <Datengröße>`: Gibt an, welche Menge der Daten für die Evaluierung verwendet werden soll.
- `--similarity-search <Suchmetrik>`: Wählen Sie die Methode zur Ähnlichkeitssuche.
- `--top-k <Anzahl der Ergebnisse>`: Legt fest, wie viele Top-Ergebnisse aus der Ähnlichkeitssuche zurückgegeben werden sollen.
- `--model <Generierungsmodell>`: Das Sprachmodell, das für die Generierung verwendet wird.
- `--eval-model <Evaluierungsmodell>`: Das Sprachmodell, das für die Evaluierung verwendet wird.

Beispiel:
```bash
python run.py eval-data --data-size 50 --similarity-search cosine --top-k 5 --model "gpt-3.5-turbo" --eval-model "gpt-3.5-turbo"
```

### Interaktive Tests

Erstellen Sie individuelle Anfragen und testen Sie RAG mit benutzerdefinierten Song-Eigenschaften:
```bash
python run.py eval-user --stage <Evaluierungsstufe> --similarity-search <Suchmetrik> --top-k <Anzahl der Ergebnisse> --model <Generierungsmodell> --eval-model <Evaluierungsmodell> --input "<Eigenschaftsvektor>"
```
**Parameterbeschreibung**
- `--stage <Evaluierungsstufe>`: Gibt die Evaluierungsstufe an.
- `--similarity-search <Suchmetrik>`: Wählen Sie die Methode zur Ähnlichkeitssuche.
- `--top-k <Anzahl der Ergebnisse>`: Legt fest, wie viele Top-Ergebnisse aus der Ähnlichkeitssuche zurückgegeben werden sollen.
- `--model <Generierungsmodell>`: Das Sprachmodell, das für die Generierung verwendet wird.
- `--eval-model <Evaluierungsmodell>`: Das Sprachmodell, das für die Evaluierung verwendet wird.
- `--input "<Eigenschaftsvektor>"`: Ein benutzerdefinierter Vektor, der die Eigenschaften des zu bewertenden Songs repräsentiert. 


Beispiel:
```bash
python run.py eval-user --stage all --similarity-search cosine --top-k 5 --model "gpt-3.5-turbo" --eval-model "gpt-3.5-turbo" --input "[0.7585784313725491,0.841988727858293,0.5454545454545454,0.9371070757670632,1,0.5255759468957438,0.0871328801560648,0.0,0.2848808134689115,0.813697470096327,0.9335252158678514,0.05847130672479715,0.75]"
```

---

## 🌐 Verfügbare Modelle

| Provider | Model                       | Generierungs-Modell | Evaluierungs-Modell |
|----------|-----------------------------|---------------------|---------------------|
| OpenAI   | gpt-4o                     | ✅                   | ✅                   |
| OpenAI   | gpt-4o-mini                | ✅                   | ✅                   |
| OpenAI   | o1                         | ✅                   | ✅                   |
| OpenAI   | o1-mini                    | ✅                   | ✅                   |
| OpenAI   | gpt-3.5-turbo              | ✅                   | ✅                   |
| Groq     | gemma2-9b-it               | ✅                   |                     |
| Groq     | llama-3.3-70b-versatile    | ✅                   |                     |
| Groq     | llama-3.1-70b-versatile    | ✅                   |                     |
| Groq     | llama-3.1-8b-instant       | ✅                   |                     |
| Groq     | llama-guard-3-8b           | ✅                   |                     |
| Groq     | llama3-70b-8192            | ✅                   |                     |
| Groq     | llama3-8b-8192             | ✅                   |                     |
| Groq     | mixtral-8x7b-32768         | ✅                   |                     |

---

## 🎧 Hintergrund
Dieses Projekt wurde entwickelt, um die Potenziale von RAG bei der Bewertung von Musikdaten zu erforschen. Durch die Verwendung sowohl von Retrieval- als auch Generationsmetriken liefert es wertvolle Erkenntnisse zur Optimierung von Empfehlungsalgorithmen.

---

Viel Spaß bei der Nutzung!