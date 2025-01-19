import json
from src.evaluation.similarity.mahalanobis import mahalanobis_distance
from src.evaluation.similarity.pearson import pearson_correlation
from src.evaluation.similarity.manhattan import manhattan_distance
from src.evaluation.similarity.euclidean import euclidean_distance
from src.evaluation.similarity.cosine import cosine_similarity

def calculate_all_metrics(vec1, vec2):
    metrics = {}

    # Metriken berechnen
    metrics["Cosine Similarity"] = cosine_similarity(vec1, vec2)
    metrics["Euclidean Distance"] = euclidean_distance(vec1, vec2)
    metrics["Manhattan Distance"] = manhattan_distance(vec1, vec2)
    metrics["Mahalanobis Distance"] = mahalanobis_distance(vec1, vec2)
    metrics["Pearson Correlation"] = pearson_correlation(vec1, vec2)
    
    return metrics

def similarity(input_vector, similar_tracks):
    # Ergebnisse speichern und Metriken berechnen
    metrics_results = []  # Speichert die berechneten Metriken für jeden Track

    for idx, track in enumerate(similar_tracks):
        track_id = track["id"]
        track_name = track["name"]
        track_label = track["label"]  # Das Label aus similar_tracks
        track_features = json.loads(track["embedding"])  # Annahme: Embedding ist JSON-String
        
        # Metriken berechnen
        metrics = calculate_all_metrics(input_vector, track_features)
        
        # Ergebnisse in einer Liste speichern
        metrics_results.append({
            "Rang": idx + 1,
            "ID": track_id,
            "Name": track_name,
            "Label": track_label,  # Das Label hinzufügen
            "Embeddings": track_features,
            **metrics
        })
    return metrics_results