import streamlit as st
from uuid import uuid4

from visualize import visualize_edge_data, visualize_multiple_graphs, visualize_graphs_with_scores
from utils import convert_to_graph

from Variants.Static.Homogeneous.Algorithms import aggregate_scores, coreness_centrality, degree_centrality, eigenvector_centrality, katz_centrality, laplacian_centrality

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
    algorithm_options = ["Aggregate Score", "Coreness Centrality", "Eigenvector Centrality", "Laplacian Centrality", "Katz Centrality", "Degree Centrality"]
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
    elif selected_algorithm == algorithm_options[2]:
        scores = eigenvector_centrality(filtered)
    elif selected_algorithm == algorithm_options[3]:
        scores = laplacian_centrality(filtered)
    elif selected_algorithm == algorithm_options[4]:
        scores = katz_centrality(filtered)
    elif selected_algorithm == algorithm_options[5]:
        scores = degree_centrality(filtered)
        
    for f in features:
        d = edge_data[edge_data["feature"] == f]
        g = convert_to_graph(d)
        
        graphs.append(g)
        
    if scores and graphs:
        st.subheader("Scores")
        st.write(scores)
        visualize_graphs_with_scores(graphs=graphs, node_scores=scores, labels=features)