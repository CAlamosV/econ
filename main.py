import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from intro import show_intro
from rct import show_rct
from ols import show_ols
from nonlinear_regression import show_nonlinear_regression
from panel_data import show_panel_data
from did import show_did
from iv import show_iv
from rdd import show_rdd
from fuzzy_rdd import show_fuzzy_rdd

page_mapping = {
    "Introduction": {'func': show_intro, 'code': 'intro'},
    "RCTs, Hypothesis Testing, and Potential Outcomes": {'func': show_rct, 'code': 'rct'},
    "Ordinary Least Squares (OLS) and Standard Errors": {'func': show_ols, 'code': 'ols'},
    "Nonlinear Regression and Omitted Variable Bias": {'func': show_nonlinear_regression, 'code': 'nonlinear_regression'},
    "Panel Data: Fixed Effects": {'func': show_panel_data, 'code': 'panel'},
    "Difference-in-Differences (DiD)": {'func': show_did, 'code': 'did'},
    "Instrumental Variables (IV)": {'func': show_iv, 'code': 'iv'},
    "Regression-Discontinuity Designs (RDDs)": {'func': show_rdd, 'code': 'rdd'},
    "Fuzzy RDDs": {'func': show_fuzzy_rdd, 'code': 'fuzzy_rdd'}
}
query_params = st.query_params
query_page = query_params.get("page", ["intro"])[0]
selected_title = [title for title, data in page_mapping.items() if data['code'] == query_page][0]


# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        options=[
            "Introduction", 
            "RCTs, Hypothesis Testing, and Potential Outcomes", 
            "Ordinary Least Squares (OLS) and Standard Errors",
            "Nonlinear Regression and Omitted Variable Bias",
            "Panel Data: Fixed Effects", 
            "Difference-in-Differences (DiD)", 
            "Instrumental Variables (IV)", 
            "Regression-Discontinuity Designs (RDDs)", 
            "Fuzzy RDDs"
        ],
        menu_title=None,
        default_index=0,
    )
    

# Display content based on selected page
if selected == "Introduction":
    show_intro()
elif selected == "RCTs, Hypothesis Testing, and Potential Outcomes":
    show_rct()
elif selected == "Ordinary Least Squares (OLS) and Standard Errors":
    show_ols()
elif selected == "Nonlinear Regression and Omitted Variable Bias":
    show_nonlinear_regression()
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