import streamlit as st
import pandas as pd

def get_input():
    
    input_choices = ["CSV"]
    
    input_selected = st.selectbox("Method of input", options=input_choices)
    
    if input_selected == input_choices[0]:
        node_file = st.file_uploader("Node data", type='csv')
        edge_file = st.file_uploader("Edge data", type='csv')
        
        if node_file and edge_file:
            node_data = pd.read_csv(node_file, index_col=0)
            edge_data = pd.read_csv(edge_file, index_col=0)
            
            return node_data, edge_data
    
    return None, None    