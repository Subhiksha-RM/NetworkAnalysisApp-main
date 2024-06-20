import pandas as pd

import networkx as nx

def convert_to_graph(edge_data):
    G = nx.DiGraph()
    
    for node in edge_data["source"].unique():
        G.add_node(int(node), label=node)
        
    for _, row in edge_data.iterrows():
        if pd.notnull(row["value"]):
            G.add_edge(row["source"], row["target"], weight=row["value"])
            
    return G