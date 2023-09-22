import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from intro import show_intro
from rct import show_rct
from ols import show_ols
from panel_data import show_panel_data
from did import show_did
from iv import show_iv
from rdd import show_rdd
from fuzzy_rdd import show_fuzzy_rdd

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        options=[
            "Introduction", 
            "Randomized Control Trials", 
            "Ordinary Least Squares (OLS)",
            "Panel Data: Fixed Effects", 
            "Difference-in-Differences (DiD)", 
            "Instrumental Variables (IV)", 
            "Regression-Discontinuity Designs (RDDs)", 
            "Fuzzy RDDs"
        ],
        menu_title=None,
        default_index=0)
    

# Display content based on selected page
if selected == "Introduction":
    show_intro()
elif selected == "Randomized Control Trials":
    show_rct()
elif selected == "Ordinary Least Squares (OLS)":
    show_ols()
elif selected == "Panel Data: Fixed Effects":
    show_panel_data()
elif selected == "Difference-in-Differences (DiD)":
    show_did()
elif selected == "Instrumental Variables (IV)":
    show_iv()
elif selected == "Regression-Discontinuity Designs (RDDs)":
    show_rdd()
elif selected == "Fuzzy RDDs":
    show_fuzzy_rdd()