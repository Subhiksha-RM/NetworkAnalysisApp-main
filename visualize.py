import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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
