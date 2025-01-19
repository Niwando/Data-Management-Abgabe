import numpy as np

from src.utils.formatting import vector_to_query

def evaluate_trulens(provider, input_vector, rag, ground_truth):

    query = vector_to_query(input_vector)

    context = rag.retrieve(query=query)

    response = rag.generate_completion(query=query, context_str=context)

    groundedness = []
    for text in context:
        groundedness.append(provider.groundedness_measure_with_cot_reasons(query, text)[0])
    
    relevance = provider.relevance_with_cot_reasons(query, response)
    context_relevance = provider.context_relevance_with_cot_reasons(query, context)

    return {
        "label": ground_truth,
        "response" : response,
        "correctness" : 1 if response==ground_truth else 0,
        "groundedness": np.mean(groundedness),
        "relevance": {
            "score": relevance[0],
            "reasons": relevance[1]
        },
        "context_relevance": {
            "score": context_relevance[0],
            "reasons": context_relevance[1]
        }
    }