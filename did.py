import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# # Set configuration for plotting
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('font', size=11)
plt.rcParams['text.usetex'] = False


# Title and Introduction
def create_index():
    st.markdown('### Table of Contents')
    st.markdown('''
    1. [Introduction to Difference-in-Differences (DiD)](#introduction-to-difference-in-differences-did)
    2. [Potential Outcomes and Parallel Trends](#potential-outcomes-and-parallel-trends)
    3. [Application: The Effect of School Construction on Education and Wages in Indonesia](#application-the-effect-of-school-construction-on-education-and-wages-in-indonesia)
    ''')

def show_did():
    st.title("Difference-in-Differences (DiD)")
    create_index()
    st.markdown("""
    ### Introduction to Difference-in-Differences (DiD)
    Difference-in-Differences (DiD) designs are a method of using natural experiments to estimate causal effects using observational data.
    The underlying idea is to compare the changes in outcomes over time between a treatment group and a control group, similar to an RCT but without randomization.
    
    In its simplest form, all a DiD model requires is a treatment group, a control group, and two time periods (before and after treatment).
    Specifically, we first compute the average change in the outcome variable for the treatment group and the control group, and then take the difference between these two changes.
    This difference represents the causal effect of the treatment.

    We can use this method to estimate the impact of school construction on years of schooling and wages in Indonesia, as in Duflo, 2001.
    In this case, treatment group are children in areas with high levels of school construction, while the control group are older children and children in areas with low levels of school construction.    
    
    ### Potential Outcomes and Parallel Trends
    By now we may be wondering when we can use DiD designs, and how we can be sure that the control and treatment groups are comparable.
    As it turns out, we can formalize the notion of a control group being comparable to a treatment group using the potential outcomes framework.
    In particular, a control group is considered valid if the potential outcomes for the control group are the same as the potential outcomes for the treatment group in the absence of treatment.
    This is known as the parallel trends assumption.

    Let's now formalize this notion mathematically. Let $Y_{\\text {it }}$ represent the outcome of interest for individual $i$ at time $t$ , and let $D_{\\text {it }}$ be a binary indicator of treatment status. Specifically:
    - $Y_{i t}(1)$ denotes the potential outcome for individual $i$ at time $t$ if treated.
    - $Y_{i t}(0)$ denotes the potential outcome for individual $i$ at time $t$ if untreated.

    For the treated group $T$ and the untreated group $U$, we define the DiD estimator based on group means:

    $$
    \\begin{aligned}
    \\left(E\\left[Y_{T 1}(1)\\right]-E\\left[Y_{T 0}(0)\\right]\\right)-\\left(E\\left[Y_{U 1}(0)\\right]-E\\left[Y_{U 0}(0)\\right]\\right)
    \\end{aligned}
    $$

    We can re-arrange the above equation to obtain:
    $$
    \\begin{aligned}
    \\underbrace{\\left(E\\left[Y_{T 1}(1)\\right]-E\\left[Y_{T 0}(0)\\right]\\right)}_{\\text {ATT }}+\\underbrace{\\left(\\left(E\\left[Y_{T 1}(0)\\right]-E\\left[Y_{T 0}(0)\\right]\\right)-\\left(E\\left[Y_{U 1}(0)\\right]-E\\left[Y_{U 0}(0)\\right]\\right)\\right)}_{\\text {Non-parallel trends bias }}
    \\end{aligned}
    $$

    We've decomposed the DiD estimator into two components:
    1. ATT (Average Treatment Effect on the Treated):
    $$
    \\begin{aligned}
    \\underbrace{E\\left[Y_{T 1}(1)\\right]-E\\left[Y_{T 0}(0)\\right]}_{\\text {ATT }}
    \\end{aligned}
    $$
    This term captures the treatment effect for the treated group $T$ in the post-treatment period.

    2. Non-parallel Trends Bias:
    $$
    \\begin{aligned}
    \\underbrace{\\left(E\\left[Y_{T 1}(0)\\right]-E\\left[Y_{T 0}(0)\\right]\\right)-\\left(E\\left[Y_{U 1}(0)\\right]-E\\left[Y_{U 0}(0)\\right]\\right)}_{\\text {Non-parallel trends bias }}
    \\end{aligned}
    $$

    This term reflects any bias that arises if the untreated group $U$ and treated group $T$ do not follow parallel trends in the absence of treatment.
    Thus, for DiD design to correctly estimate the ATT, we require that:
    $$
    \\begin{aligned}
    E\\left[Y_{T 1}(0)\\right]-E\\left[Y_{T 0}(0)\\right]=E\\left[Y_{U 1}(0)\\right]-E\\left[Y_{U 0}(0)\\right]
    \\end{aligned}
    $$

    All this is saying is that the DiD estimator is unbiased if the average outcomes for the treated and untreated groups would have followed the same trend in the absence of treatment.

    ### Application: The Effect of School Construction on Education and Wages in Indonesia

    We can use a DiD design to estimate the impact of school construction on years of schooling and wages in Indonesia.
    The treatment and control groups are defined as follows:
    - **Treatment Group**: Children ages 2 to 6 in regions exposed to high levels of school construction.
    - **Control Group**: All children ages 12 to 17  and children ages 2 to 6 in regions with low levels of school construction.
    
    After some simple data cleaning, we obtain a dataset with the following variables:
    """)

    with st.expander("Show Code"):
        st.code("""
        df = pd.read_stata('data/indonesia_schooling.dta') # Load data
        df = df.rename(columns={ # Renaming and selecting columns
            'p504thn': 'birth_yr',
            'p509pro': 'province',
            'recp': 'school_construction',
            'lhwage': 'log_wage',
            'yeduc': 'years_of_education',
        })[['log_wage', 'years_of_education', 'school_construction', 'birth_yr', 'province', 'weight']]

        df = df[df['log_wage'].notna()] # Drop rows with missing log_wage

        df['age_in_1974'] = 74 - df['birth_yr'] # Calculate age in 1974

        df.sample(3, random_state=42) # Display 3 rows of the data
        """, language='python')
    df = pd.read_stata('data/indonesia_schooling.dta')
    df = df.rename(columns={ # Renaming and selecting columns
        'p504thn': 'birth_yr',
        'p509pro': 'province',
        'recp': 'school_construction',
        'lhwage': 'log_wage',
        'yeduc': 'years_of_education',
    })[['log_wage', 'years_of_education', 'school_construction', 'birth_yr', 'province', 'weight']]
    df = df[df['log_wage'].notna()] 
    df['age_in_1974'] = 74 - df['birth_yr'] # Calculate age in 1974

    st.dataframe(df.sample(3, random_state=42), hide_index=True, width=700) # Display 3 rows of the data

    # Generate age group indicators
    df['old'] = ((df['age_in_1974'] <= 17) & (df['age_in_1974'] >= 12)).astype(int)
    df['young'] = ((df['age_in_1974'] >= 2) & (df['age_in_1974'] <= 6)).astype(int)

    # Generate interaction term for high_inpres and young
    df['school_construction_x_young'] = df['school_construction'] * df['young']
    df = df[(df['young'] == 1) | (df['old'] == 1)]

    st.markdown("""
    We then generate an interaction term between the school construction variable and the young indicator.
    We also restrict out data to exclude children between the ages of 7 and 11, since they would only partially be affected by the school construction program.
    """)
    
    with st.expander("Show Code"):
        st.code("""
        # Generate a variable for indicating old and young age groups
        df['old'] = ((df['age_in_1974'] <= 17) & (df['age_in_1974'] >= 12)).astype(int)
        df['young'] = ((df['age_in_1974'] >= 2) & (df['age_in_1974'] <= 6)).astype(int)

        # Generate interaction term for school_construction and young
        df['school_construction_x_young'] = df['school_construction'] * df['young']

        # Filter data to include only young and old age groups
        df = df[(df['young'] == 1) | (df['old'] == 1)]
        """, language='python')
    
    st.markdown("""
    We can now estimate the following OLS regression model:
    $$
    \\begin{aligned}
    Y_{i} = & \ \\beta_{0} + \\beta_{1} \\text{SchoolConstruction}_{p(i)} + \\beta_{2} \\text{SchoolConstruction}_{p(i)} \\times \\text{Young}_{i} \\\\
    & + \\delta_{p(i)} + \\gamma_{c(i)} + \\varepsilon_{i}
    \\end{aligned}
    $$

    where:
    - $Y_{i}$ is the outcome variable (years of education or log wage) for individual $i$.
    - $\\text{SchoolConstruction}_{p(i)}$ is the school construction variable for province $p(i)$.
    - $\\text{Young}_{i}$ is an indicator variable for young children.
    - $\\delta_{p(i)}$ are province fixed effects.
    - $\\gamma_{c(i)}$ are cohort fixed effects.
    - $\\varepsilon_{i}$ is the error term.

    $\\beta_{2}$ captures the effect of school construction on wages for young children.

    Notice that in this instance we don't include a coefficient for the young indicator variable, since cohort fixed effects are already included in the model.
    If we were to include a coefficient for the young indicator, we have multicollinearity issues.

    After estimating the model, we obtain the following results$^1$:
    """)

    # Note that results are pre-computed to save loading time.
    # Feel free to run code in the dropdowns to verify results.
    df_results = {
    "Outcome": ["Years of Education", "Log Wage"],
    "Coefficient on Interaction Term": [0.180, 0.014],
    "Standard Error": [0.086, 0.005],
    "P-value": [0.037, 0.004]
    }

    with st.expander("Show Code"):
        st.code("""
    # Estimate the model for years of education and log wage
    education_formula = "years_of_education ~ school_construction + school_construction_x_young + C(province) + ch71*C(birth_yr)"
    log_wage_formula = "log_wage ~ school_construction + school_construction_x_young + C(province) + ch71*C(birth_yr)"

    # Run the weighted least squares regression using the "weight" column
    education_results = smf.wls(formula=education_formula, data=df, weights=df['weight'], hasconst=True).fit()
    log_wage_results = smf.wls(formula=log_wage_formula, data=df, weights=df['weight'], hasconst=True).fit()
   
    # Function to extract and present estimates
    def get_estimates(result, term):
        coef = np.round(result.params[term], 3)
        std_err = np.round(result.bse[term], 3)
        p_value = np.round(result.pvalues[term], 3)
        return coef, std_err, p_value

    # Extract estimates for the interaction term (beta 2)
    term = 'school_construction_x_young'
    education_coef, education_std_err, education_p_value = get_estimates(education_results, term)
    log_wage_coef, log_wage_std_err, log_wage_p_value = get_estimates(log_wage_results, term)

    # Create a DataFrame with the results and display it
    df_results = pd.DataFrame({
        'Outcome': ['Years of Education', 'Log Wage'],
        'Coefficient on Interaction Term': [education_coef, log_wage_coef],
        'Standard Error': [education_std_err, log_wage_std_err],
        'P-value': [education_p_value, log_wage_p_value]
    })
    df_results
        """, language='python')
    
    st.dataframe(df_results, hide_index=True, width=500) # Display the results

    st.markdown("""
    Thus, we find that high levels of school construction in Indonesia led to an increase of 0.18 years of education and a 1.4 percent increase in wages for young children.

    We may also want to know what this implies about the returns to an additional year of schooling.
    To answer questions like these, we need to introduce the concept of Instrumental Variables (IVs).
    """)
    st.markdown(
    """
    <p style='font-size: 12px;'>1: Results may differ slightly from the original paper since I am ommitting some sample restrictions for simplicity.</p>
    """,
    unsafe_allow_html=True
)



    
