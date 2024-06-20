import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import netcenlib as ncl
import netcenlib.algorithms as ncl_algorithms
import pandas as pd

from utils import convert_to_graph

def aggregate_scores(edge_data):
    weights = {
        "coreness": 1/14,
        "degree": 1/14,
        "eigenvector": 1/14,
        "katz": 1/14,
        "laplacian": 1/14,
        "betweeness": 1/14,
        "closeness": 1/14,
        "pagerank": 1/14,
        "local_clustering": 1/14,
        "percolation": 1/14,
        "clusterrank": 1/14,
        "max_neighborhood": 1/14,
        "semi_local": 1/14,
        "load": 1/14,
            }
    
    # Calculate centrality scores
    coreness_scores = coreness_centrality(edge_data)
    degree_scores = degree_centrality(edge_data)
    eigenvector_scores = eigenvector_centrality(edge_data)
    katz_scores = katz_centrality(edge_data)
    laplacian_scores = laplacian_centrality(edge_data)
    betweeness_scores = betweeness_centrality(edge_data)
    closeness_scores = closeness_centrality(edge_data)
    local_clustering_scores = local_clustering_coeff_centrality(edge_data)
    percolation_scores = percolation_centrality(edge_data)
    clusterrank_scores = cluster_rank_centrality(edge_data)
    max_neighborhood_scores = max_neighborhood_centrality(edge_data)
    semi_local_scores = semi_local_centrality(edge_data)
    load_scores = load_centrality(edge_data)
    pagerank_scores = pagerank_centrality(edge_data)

    # Convert scores to a common format (dict of lists) for normalization
    nodes = list(coreness_scores.keys())
    scores_matrix = {node: [] for node in nodes}
    
    for node in nodes:
        scores_matrix[node].append(coreness_scores[node])
        scores_matrix[node].append(degree_scores[node])
        scores_matrix[node].append(eigenvector_scores[node])
        scores_matrix[node].append(katz_scores[node])
        scores_matrix[node].append(laplacian_scores[node])
        scores_matrix[node].append(betweeness_scores[node])
        scores_matrix[node].append(closeness_scores[node])
        scores_matrix[node].append(local_clustering_scores[node])
        scores_matrix[node].append(percolation_scores[node])
        scores_matrix[node].append(clusterrank_scores[node])
        scores_matrix[node].append(max_neighborhood_scores[node])
        scores_matrix[node].append(semi_local_scores[node])
        scores_matrix[node].append(load_scores[node])
        scores_matrix[node].append(pagerank_scores[node])


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

def betweeness_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.betweenness_centrality(G)

def percolation_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.percolation_centrality(G)

def pagerank_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.pagerank(G)

def closeness_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.closeness_centrality(G)

def local_clustering_coeff_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.clustering(G)

def cluster_rank_centrality(edge_data):
    # G = convert_to_graph(edge_data)
    G = nx.Graph()
    
    for node in edge_data["source"].unique():
        G.add_node(int(node), label=node)
        
    for _, row in edge_data.iterrows():
        if pd.notnull(row["value"]):
            G.add_edge(row["source"], row["target"], weight=row["value"])
    return ncl.cluster_rank_centrality(G)

def max_neighborhood_centrality(edge_data):
    #G = convert_to_graph(edge_data)
    G = nx.Graph()
    
    for node in edge_data["source"].unique():
        G.add_node(int(node), label=node)
        
    for _, row in edge_data.iterrows():
        if pd.notnull(row["value"]):
            G.add_edge(row["source"], row["target"], weight=row["value"])
    return ncl.mnc_centrality(G)

def semi_local_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return ncl_algorithms.semi_local_centrality(G)

def load_centrality(edge_data):
    G = convert_to_graph(edge_data)
    return nx.load_centrality(G)