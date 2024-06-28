import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components

import pathpyG as pp

from utils import convert_to_graph

def visualize_edge_data(edge_data):
    
    G = convert_to_graph(edge_data)
    plt.figure()
    
    pos = nx.circular_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20, edge_color='grey', width=2)

    nx.draw_networkx_labels(G, pos)
    
    plt.axis('off')

    st.pyplot(plt)

def visualize_multiple_graphs(graphs, labels):

    plt.figure()

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for graph, label, color in zip(graphs, labels, color_cycle):
        pos = nx.circular_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos, edge_color=color, label=label)
        nx.draw_networkx_labels(graph, pos)
    plt.axis('off')
    plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in zip(labels, color_cycle)],
               loc='best', title="Features")
    st.pyplot(plt)
    
def visualize_graphs_with_scores(graphs, node_scores, labels, height=None, width=None):
    scores = {int(k): float(v) for k, v in node_scores.items()}
    
    scores_array = np.array(list(scores.values()))
    min_score, max_score = scores_array.min(), scores_array.max()
    
    if min_score == max_score:
        norm_scores = {node: 0.5 for node in scores.keys()}
    else:
        norm_scores = {node: (score - min_score) / (max_score - min_score) for node, score in scores.items()}
    
    node_sizes = {node: 200 + 900 * norm for node, norm in norm_scores.items()}
    node_opacities = {node: 0.4 + 0.6 * norm for node, norm in norm_scores.items()}
    
    plt.figure()
    if height and width:
        plt.figure(figsize=(width, height))
    
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for graph, label, color in zip(graphs, labels, color_cycle):
        pos = nx.circular_layout(graph) 
        
        default_size = 200
        default_opacity = 0.4
        
        sizes = [node_sizes.get(node, default_size) for node in graph.nodes()]
        alphas = [node_opacities.get(node, default_opacity) for node in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=sizes, alpha=alphas)
        nx.draw_networkx_edges(graph, pos, edge_color=color, label=label, arrowstyle='->', arrowsize=20, width=2)
        nx.draw_networkx_labels(graph, pos)
    
    plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in zip(labels, color_cycle)],
               loc='best', title="Features")
    
    plt.axis('off')
    st.pyplot(plt)


def visualize_dynamic_graph(edge_data):
    edge_list = []
    
    
    feature = st.selectbox("Edge Feature", options=edge_data['feature'].unique())
    
    edge_data_filtered = edge_data[edge_data['feature'] == feature]

    # Iterate over the DataFrame
    for index, row in edge_data_filtered.iterrows():
        source = str(row['source'])
        target = str(row['target'])
        timestamp = int(row['timestamp'])
        value = row['value']

        # Add an edge to the edge list if value is 1
        if value == 1:
            edge_list.append([source, target, timestamp])

     # Convert the first two elements of each sublist to strings
    new_data = [[x[0], x[1], x[2]] for x in edge_list]

    t = pp.TemporalGraph.from_edge_list(new_data)
    
    # Get node labels and assign colors based on node IDs
    node_labels = t.mapping.node_ids
    if isinstance(node_labels, np.ndarray):
        node_labels = node_labels.tolist()
    
    # Assign colors to nodes based on their IDs
    num_nodes = len(node_labels)
    color_map = {node_labels[i]: f'#{i*120%256:02x}{(i*80)%256:02x}{(i*40)%256:02x}' for i in range(num_nodes)}  # Example color assignment
    
    pp.plot(t, node_labels=node_labels, node_color=color_map, start=-1, end=99, delta=500, filename='test_plot.html')
    
    # Read the HTML file
    with open('test_plot.html', 'r', encoding='utf-8') as HtmlFile:
        source_code = HtmlFile.read()

    # Display the HTML content
    return components.html(source_code, height=500)
