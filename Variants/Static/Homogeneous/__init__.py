import streamlit as st
from uuid import uuid4

from visualize import visualize_edge_data, visualize_multiple_graphs, visualize_graphs_with_scores
from utils import convert_to_graph

from Variants.Static.Homogeneous.Algorithms import aggregate_scores, coreness_centrality, degree_centrality, eigenvector_centrality, katz_centrality, laplacian_centrality, betweeness_centrality, percolation_centrality, pagerank_centrality, closeness_centrality,local_clustering_coeff_centrality, cluster_rank_centrality,max_neighborhood_centrality,semi_local_centrality,load_centrality

def static_homogenous(node_data, edge_data):
    st.title("Static Homogeneous Graphs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Edge Data")
        
        st.dataframe(edge_data)
        
    with col2:
        st.subheader("Visualization")
        
        graphs = []
        for f in edge_data["feature"].unique():
            f = edge_data[edge_data["feature"] == f]
            g = convert_to_graph(f)
            graphs.append(g)
            
        visualize_multiple_graphs(graphs=graphs, labels=edge_data['feature'].unique())
        
    st.header("Results")
    
    is_compare = st.checkbox("Compare results", key='compare_results')
    
    if is_compare:
        col1, col2 = st.columns(2)
            
        with col1:
            process_flow(edge_data, "left")
            
        with col2:
            process_flow(edge_data, "right")
    
    else:
        process_flow(edge_data, "single")
        
    
def process_flow(edge_data, key_prefix):
    algorithm_options = ["Aggregate Score", "Coreness Centrality", "Eigenvector Centrality", "Laplacian Centrality", "Katz Centrality", "Degree Centrality", 
                         "Betweenness Centrality", "Percolation Centrality", "PageRank Centrality", "Closeness Centrality", "Local Clustering Coefficient", 
                         "Cluster Rank Centrality", "Maximum Neighborhood Component", "Semi Local Centrality", "Load Centrality"]
    selected_algorithm = st.selectbox("Algorithm", options=algorithm_options, key=f'{key_prefix}_algorithm')
    input_features = st.text_input("Comma separated features (Empty for all features)", key=f'{key_prefix}_features')
    
    if input_features != "":
        features = [int(f.strip()) for f in input_features.split(',')]
    else:
        features = edge_data["feature"].unique()
    
    filtered = edge_data[edge_data["feature"].isin(features)]
        
    graphs = []
    scores = None
    
    if selected_algorithm == algorithm_options[0]:
        scores = aggregate_scores(filtered)
        
    if selected_algorithm == algorithm_options[1]:
        scores = coreness_centrality(filtered)
        st.write("Coreness Centrality: Tightly connected group of nodes in a n/w with max edges, Connection: Highlights the major core, Supply Chain: Strong interconnection with other major nodes")
    elif selected_algorithm == algorithm_options[2]:
        scores = eigenvector_centrality(filtered)
    elif selected_algorithm == algorithm_options[3]:
        scores = laplacian_centrality(filtered)
        st.write("Laplacian Centrality: Measures a node’s importance by deleting the node in the n/w, Connection: How much do a node deletion affect the n/w?, Supply Chain: Disruption / Deletion – affects whole n/w")
    elif selected_algorithm == algorithm_options[4]:
        scores = katz_centrality(filtered)
        st.write("Katz Centrality: Influence of node on its neighbour & their neighbours, Connection: How does this node influence other indirectly?, Supply Chain: Manufacturer – connected to a supplier – if it influences the neighbour of the supplier")
    elif selected_algorithm == algorithm_options[5]:
        scores = degree_centrality(filtered)
        st.write("Degree Centrality: No. of Edges Connecting a Node, Connection: How significant is its connection?, Supply Chain: Flow of Goods")
    elif selected_algorithm == algorithm_options[6]:
        scores = betweeness_centrality(filtered)
    elif selected_algorithm == algorithm_options[7]:
        scores = percolation_centrality(filtered)
        st.write("Percolation Centrality: Node that causes ripple effect throughout the n/w, Connection: To identify if it connects any major nodes?, Supply Chain: Disruption in Source node – affects other nodes")
    elif selected_algorithm == algorithm_options[8]:
        scores = pagerank_centrality(filtered)
    elif selected_algorithm == algorithm_options[9]:
        scores = closeness_centrality(filtered)
    elif selected_algorithm == algorithm_options[10]:
        scores = local_clustering_coeff_centrality(filtered)
    elif selected_algorithm == algorithm_options[11]:
        scores = cluster_rank_centrality(filtered)
        st.write("ClusterRank Centrality: Influence of a node on its immediate neighbours and how well the neighbours are clustered together, Connection: How well is the node connected and clustered?,Supply Chain: Identify Key suppliers/customers, Risk assessment, optimization of allocation, improve resilience")
    elif selected_algorithm == algorithm_options[12]:
        scores = max_neighborhood_centrality(filtered)
        st.write("Max Neighborhood Centrality: Measures a node’s largest neighbourhood without the node, Connection: How central is an entity in the n/w?, Supply Chain: Supplier supplying more than 1 materials, Distributor handling max no. of product distribution.")
    elif selected_algorithm == algorithm_options[13]:
        scores = semi_local_centrality(filtered)
        st.write("Semi Local Centrality: Degree of a node, degree of its neighbours (degree- no.of edges), Connection: What is the influence of the node on others?, Supply Chain: Disruption – how it affects the neighbours and the n/w")
    elif selected_algorithm == algorithm_options[14]:
        scores = load_centrality(filtered)
        
    for f in features:
        d = edge_data[edge_data["feature"] == f]
        g = convert_to_graph(d)
        
        graphs.append(g)
        
    if scores and graphs:
        st.subheader("Scores")
        st.write(scores)
        visualize_graphs_with_scores(graphs=graphs, node_scores=scores, labels=features)