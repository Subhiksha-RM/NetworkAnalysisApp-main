import streamlit as st

from input import get_input

from Variants.Static.Homogeneous import static_homogenous

node_data = None
edge_data = None

st.set_page_config(layout="wide")

with st.sidebar:
    st.header("Network Analysis")

    node_data, edge_data = get_input()
    variant_choices = ["Static Homogeneous"]
    
    variant_selected = st.selectbox("Variant", options=variant_choices)
    
if node_data is not None and edge_data is not None:
    if variant_selected == variant_choices[0]:
        static_homogenous(node_data, edge_data)