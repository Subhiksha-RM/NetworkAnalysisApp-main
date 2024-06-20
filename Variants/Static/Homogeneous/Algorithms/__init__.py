import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from utils import convert_to_graph

def aggregate_scores(edge_data):
    weights = {
        "coreness": 0.2,
        "degree": 0.2,
        "eigenvector": 0.2,
        "katz": 0.2,
        "laplacian": 0.2        
    }
    
    # Calculate centrality scores
    coreness_scores = coreness_centrality(edge_data)
    degree_scores = degree_centrality(edge_data)
    eigenvector_scores = eigenvector_centrality(edge_data)
    katz_scores = katz_centrality(edge_data)
    laplacian_scores = laplacian_centrality(edge_data)

    # Convert scores to a common format (dict of lists) for normalization
    nodes = list(coreness_scores.keys())
    scores_matrix = {node: [] for node in nodes}
    
    for node in nodes:
        scores_matrix[node].append(coreness_scores[node])
        scores_matrix[node].append(degree_scores[node])
        scores_matrix[node].append(eigenvector_scores[node])
        scores_matrix[node].append(katz_scores[node])
        scores_matrix[node].append(laplacian_scores[node])

    # Normalize scores individually using MinMaxScaler
    scaler = MinMaxScaler()
    for i, alg in enumerate(weights):
        algorithm_scores = [scores_matrix[node][i] for node in nodes]
        # Reshape algorithm_scores into a 2D array for MinMaxScaler
        algorithm_scores_reshaped = np.array(algorithm_scores).reshape(-1, 1)
        normalized_scores = scaler.fit_transform(algorithm_scores_reshaped).flatten()
        # Update scores_matrix with normalized scores
        for j, node in enumerate(nodes):
            scores_matrix[node][i] = normalized_scores[j]

    # Aggregate normalized scores with weights
    aggregated_scores = {}
    for node in nodes:
        weighted_sum = sum(weights[alg] * scores_matrix[node][i] for i, alg in enumerate(weights))
        aggregated_scores[node] = weighted_sum

    return aggregated_scores

def coreness_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.core_number(G)

def degree_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.degree_centrality(G)

def eigenvector_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.eigenvector_centrality(G)

def katz_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.katz_centrality(G)

def laplacian_centrality(edge_data):
    G = convert_to_graph(edge_data)
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    centrality = np.sum(eigenvectors**2, axis=1)
    return dict(zip(G.nodes(), centrality))
