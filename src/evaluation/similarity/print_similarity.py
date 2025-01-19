from tabulate import tabulate

def print_similarity(similar_tracks, metrics_results):

    # 1. Top-k Ergebnisse mit Rang, Namen und Embeddings
    print("\nTop-k Ergebnisse (Rang, Name, Label und Embedding):")
    top_k_table = [
        {"Rang": idx + 1, "Name": track["name"], "Label": track["label"], "Embeddings": track["embedding"]}
        for idx, track in enumerate(similar_tracks)
    ] 
    print(tabulate(top_k_table, headers="keys", tablefmt="grid"))

    # 2. Metriken für jeden Track anzeigen
    print("\nMetriken für jeden Track:")
    metrics_table = [
        {"Name": result["Name"], "Label": result["Label"], **{k: v for k, v in result.items() if k not in ["Rang", "ID", "Name", "Label", "Embeddings"]}}
        for result in metrics_results
    ]
    print(tabulate(metrics_table, headers="keys", tablefmt="grid"))