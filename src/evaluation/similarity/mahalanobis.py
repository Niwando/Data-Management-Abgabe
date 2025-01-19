import numpy as np
import pandas as pd

def mahalanobis_distance(vec1, vec2):
    cov_matrix_df = pd.read_csv("src/data/preprocessed/covariance_matrix.csv", index_col=0)

    diff = np.array(vec1) - np.array(vec2)
    inv_cov_matrix = np.linalg.inv(cov_matrix_df.values)
    distance = np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))
    return distance