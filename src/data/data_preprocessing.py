import json
import pandas as pd
import re
import numpy as np

# Dateien einlesen
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_python_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Extrahiere den Dictionary-Teil mit einem Regex
        match = re.search(r"(yes_ids|no_ids)\s*=\s*(\{.*\})", content, re.DOTALL)
        if match:
            return eval(match.group(2))  # Evaluiere nur den Dictionary-Teil
        else:
            raise ValueError(f"Keine gültigen Daten in {file_path} gefunden")

# Daten aus Dateien laden
yes_tracks = load_python_dict('src/data/raw/yes.py')['items']
no_tracks = load_python_dict('src/data/raw/no.py')['items']
good_features = load_json_file('src/data/raw/good.json')['audio_features']
dislike_features = load_json_file('src/data/raw/dislike.json')['audio_features']

# Funktion zur Track-Zuordnung
def create_track_mapping(tracks, label):
    return {track['track']['id']: {"name": track['track']['name'], "label": label} for track in tracks}

yes_mapping = create_track_mapping(yes_tracks, "yes")
no_mapping = create_track_mapping(no_tracks, "no")

def create_feature_mapping(features, label):
    return {
        feature['id']: {
            "danceability": feature.get('danceability'),
            "energy": feature.get('energy'),
            "key": feature.get('key'),
            "loudness": feature.get('loudness'),
            "mode": feature.get('mode'),
            "speechiness": feature.get('speechiness'),
            "acousticness": feature.get('acousticness'),
            "instrumentalness": feature.get('instrumentalness'),
            "liveness": feature.get('liveness'),
            "valence": feature.get('valence'),
            "tempo": feature.get('tempo'),
            "duration_ms": feature.get('duration_ms'),
            "time_signature": feature.get('time_signature'),
            "label": label,
        }
        for feature in features
    }

good_mapping = create_feature_mapping(good_features, "good")
dislike_mapping = create_feature_mapping(dislike_features, "dislike")

# Merge aller Daten
merged_data = {}
for source in [yes_mapping, no_mapping, good_mapping, dislike_mapping]:
    for track_id, details in source.items():
        if track_id not in merged_data:
            merged_data[track_id] = details
        else:
            merged_data[track_id].update(details)

# Normalisierung der Werte
features_to_normalize = [
    "danceability", "energy", "key", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"
]

# Berechnung der Normalisierungsparameter
normalization_params = {
    "min": {feature: float('inf') for feature in features_to_normalize},
    "max": {feature: float('-inf') for feature in features_to_normalize},
}

# Min- und Max-Werte finden
for details in merged_data.values():
    for feature in features_to_normalize:
        value = details.get(feature)
        if value is not None:
            normalization_params["min"][feature] = min(normalization_params["min"][feature], value)
            normalization_params["max"][feature] = max(normalization_params["max"][feature], value)

# Normalisierung anwenden
def normalize(value, min_val, max_val):
    if value is None or min_val == max_val:
        return value
    return (value - min_val) / (max_val - min_val)

for details in merged_data.values():
    for feature in features_to_normalize:
        details[feature] = normalize(details.get(feature), normalization_params["min"][feature], normalization_params["max"][feature])

# Binärcodierung der Labels
def encode_label(label):
    return "like" if label == "good" else "dislike"

# Daten in ein DataFrame umwandeln
data = []
for track_id, details in merged_data.items():
    data.append({
        "ID": track_id,
        "Name": details.get("name", "Unknown"),
        "Danceability": details.get("danceability"),
        "Energy": details.get("energy"),
        "Key": details.get("key"),
        "Loudness": details.get("loudness"),
        "Mode": details.get("mode"),
        "Speechiness": details.get("speechiness"),
        "Acousticness": details.get("acousticness"),
        "Instrumentalness": details.get("instrumentalness"),
        "Liveness": details.get("liveness"),
        "Valence": details.get("valence"),
        "Tempo": details.get("tempo"),
        "Duration_ms": details.get("duration_ms"),
        "Time_Signature": details.get("time_signature"),
        "Label": encode_label(details.get("label")),
    })

# DataFrame erstellen und speichern
df = pd.DataFrame(data)
csv_file = 'src/data/preprocessed/data.csv'
df.to_csv(csv_file, index=False, encoding='utf-8')

print(f"Die Daten wurden erfolgreich in {csv_file} gespeichert.")

# Kovarianzmatrix berechnen
features = df[[
    "Danceability", "Energy", "Key", "Loudness", "Mode", "Speechiness",
    "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Time_Signature"
]]
cov_matrix = features.cov()

# Kovarianzmatrix speichern
cov_file = 'src/data/preprocessed/covariance_matrix.csv'
cov_matrix.to_csv(cov_file, index=True)
print(f"Kovarianzmatrix wurde in {cov_file} gespeichert.")
