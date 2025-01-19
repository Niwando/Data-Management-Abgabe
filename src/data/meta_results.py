def generate_meta_results(data):
    overall_meta_results = {
        "total_tracks": len(data),
        "average_correctness": 0,
        "average_groundedness": 0,
        "average_relevance_score": 0,
        "average_context_relevance_score": 0,
        "average_cosine_similarity": 0,
        "average_euclidean_distance": 0,
        "average_manhattan_distance": 0,
        "average_mahalanobis_distance": 0,
        "average_pearson_correlation": 0
    }
    
    # Summing up the metrics across all tracks
    for details in data.values():
        overall_meta_results["average_correctness"] += details["correctness"]
        overall_meta_results["average_groundedness"] += details["groundedness"]
        overall_meta_results["average_relevance_score"] += details["relevance"]["score"]
        overall_meta_results["average_context_relevance_score"] += details["context_relevance"]["score"]
        
        overall_meta_results["average_cosine_similarity"] += sum(
            [ctx["cosine similarity"] for ctx in details["context"].values()]
        ) / len(details["context"])
        
        overall_meta_results["average_euclidean_distance"] += sum(
            [ctx["euclidean distance"] for ctx in details["context"].values()]
        ) / len(details["context"])
        
        overall_meta_results["average_manhattan_distance"] += sum(
            [ctx["manhattan distance"] for ctx in details["context"].values()]
        ) / len(details["context"])
        
        overall_meta_results["average_mahalanobis_distance"] += sum(
            [ctx["mahalanobis distance"] for ctx in details["context"].values()]
        ) / len(details["context"])
        
        overall_meta_results["average_pearson_correlation"] += sum(
            [ctx["pearson correlation"] for ctx in details["context"].values()]
        ) / len(details["context"])
    
    # Calculate averages
    total_tracks = overall_meta_results["total_tracks"]
    if total_tracks > 0:
        overall_meta_results["average_correctness"] /= total_tracks
        overall_meta_results["average_groundedness"] /= total_tracks
        overall_meta_results["average_relevance_score"] /= total_tracks
        overall_meta_results["average_context_relevance_score"] /= total_tracks
        overall_meta_results["average_cosine_similarity"] /= total_tracks
        overall_meta_results["average_euclidean_distance"] /= total_tracks
        overall_meta_results["average_manhattan_distance"] /= total_tracks
        overall_meta_results["average_mahalanobis_distance"] /= total_tracks
        overall_meta_results["average_pearson_correlation"] /= total_tracks
    
    return overall_meta_results