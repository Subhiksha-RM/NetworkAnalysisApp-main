import streamlit as st

from input import get_input

from Variants.Static.Homogeneous import static_homogenous
from Variants.Dynamic.Homogenous import dynamic_homogenous

node_data = None
edge_data = None

st.set_page_config(layout="wide")

with st.sidebar:
    st.header("Network Analysis")

    node_data, edge_data = get_input()
    # variant_choices = ["Static Graph", "Dynamic Graph"]
    
    # variant_selected = st.selectbox("Variant", options=variant_choices)
    
if node_data is not None and edge_data is not None:
    if 'timestamp' in edge_data.columns:
        dynamic_homogenous(node_data=node_data, edge_data=edge_data)
    else:        
        static_homogenous(node_data, edge_data)