import numpy as np

def pearson_correlation(vec1, vec2):
    mean_a = np.mean(vec1)
    mean_b = np.mean(vec2)
    numerator = np.sum((vec1 - mean_a) * (vec2 - mean_b))
    denominator = np.sqrt(np.sum((vec1 - mean_a) ** 2)) * np.sqrt(np.sum((vec2 - mean_b) ** 2))
    return numerator / denominator if denominator != 0 else 0