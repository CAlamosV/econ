import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels.api as sm

from linearmodels import PanelOLS
from linearmodels.datasets import wage_panel
from linearmodels.panel.results import compare
import seaborn as sns

#config
sns.set_style("whitegrid")
sns.set_palette("husl")
# plt.rcParams["font.sans-serif"] = "Arial"
# plt.rcParams["mathtext.fontset"] = "cm"

def create_index():
    st.markdown('### Table of Contents')
    st.markdown('''
    1. [Motivation: The Effect of Marriage on Wages](#motivation-the-effect-of-marriage-on-wages)
    2. [Implementing a Fixed Effects Model](#implementing-a-fixed-effects-model)
    3. [Cluster-Robust Standard Errors](#cluster-robust-standard-errors)
    ''')

def show_panel_data():
    st.title('Panel Data: Fixed Effects')

    create_index()

    st.write('''
    ### Motivation: The Effect of Marriage on Wages

    Suppose we want to estimate the effect of getting married on wages.
    We cannot simply compare the wages of married and unmarried people, because married people are different from unmarried people in many ways.
    Even if we control for many obvious observable characteristics such as education, age, and race, there are still many unobservable characteristics that are correlated with both marriage and wages.
    In particular, we may be concerned that people who get married are more likely to have fixed traits such as conscientiousness, which may also be correlated with wages.
    Formally, we believe wages are determined by the following model:
    $$
    Y_{it} = \\beta_1  X_{it} + \\beta_2 u_i + \\varepsilon_{it}
    $$

    Where $Y_{it}$ represents the wage of individual $i$ at time $t$, $X_{it}$ are observable characteristics including marriage status and $u_i$ are time-invariant unobservable characteristics.
    (Note that the intercept is included in $u_i$ for simplicity).
    To control for these unobservables, we can a use fixed effects model.
    To use this type of model, we need to observe multiple time periods for each individual.
    All the fixed effects model amounts to is demeaning the data.

    To how this works, define $\\bar{Y}_i$ as the average of $Y_{it}$ for individual $i$ over all time periods.

    This implies that:
    $$
    \\begin{aligned}
    \\bar{Y}_{it} &= \\beta_1 \\bar{X}_{it} + \\beta_2 u_i 
    \end{aligned}
    $$
    Where $\\bar{X}_{it}$ is defined similarly to $\\bar{Y}_{it}$. (The error term has disappeared from the equation, since its mean is zero by contruction).
    We can then subtract this equation from the original equation to get:
    $$
    \\begin{aligned}
    Y_{it} - \\bar{Y}_{it} &= \\beta_1 (X_{it} - \\bar{X}_{it}) + \\varepsilon_{it} \\\\
    \\end{aligned}
    $$

    Thus, without ever having direct access to $u_i$, we have managed to control for it by simply using the fact that it's constant over time.

    Note that we can also estimate:
    $$
    Y_{it} = \\beta_1  X_{it} + \\alpha_i + \\varepsilon_{it}
    $$
    Where $\\alpha_i$ is an individual-specific intercept and will therefore capture time-invariant unobservable characteristics.
    Both of these approaches, demeaning and including individual-specific intercepts, are equivalent.

    ### Implementing a Fixed Effects Model
    We can use data from Vella and M. Verbeek (1998) to estimate the effect of marriage on wages.
    This data contains information on 595 individuals over 8 years (1980-1987).

    ''')
    with st.expander('Show code'):
        st.code('''
         df = wage_panel.load()
        df = df.rename(
            columns={
                "nr": "i",
                "year": "t",
                "exper": "experience",
                "expersq": "experience_squared",
                "married": "is_married",
                "union": "is_union_member",
                "lwage": "log_wage",
            }
        )[['i', 't', 'experience', 'experience_squared', 'is_married', 'is_union_member', 'log_wage']]

        # Demean the data
        for col in ['experience', 'experience_squared', 'is_married', 'is_union_member']:
            df[col + '_demeaned'] = df.groupby('i')[col].transform(lambda x: x - x.mean())

        # Estimate naive OLS
        endog = df['log_wage']
        exog = sm.add_constant(df[['experience', 'experience_squared', 'is_married', 'is_union_member']])
        sm.OLS(endog, exog).fit().summary().tables[1]

        # Estimate a Fixed Effects Model using OLS and demeaned data
        endog = df['log_wage']
        exog = df[['experience_demeaned', 'experience_squared_demeaned', 'is_married_demeaned', 'is_union_member_demeaned']]
        sm.OLS(endog, exog).fit().summary().tables[1]
        ''', language='python')


    df = wage_panel.load()
    df = df.rename(
        columns={
            "nr": "i",
            "year": "t",
            "exper": "experience",
            "expersq": "experience_squared",
            "married": "is_married",
            "union": "is_union_member",
            "lwage": "log_wage",
        }
    )[['i', 't', 'experience', 'experience_squared', 'is_married', 'is_union_member', 'log_wage']]

    # Demean the data
    for col in ['experience', 'experience_squared', 'is_married', 'is_union_member', 'log_wage']:
        df[col + '_demeaned'] = df.groupby('i')[col].transform(lambda x: x - x.mean())

    # Estimate naive OLS
    endog = df['log_wage']
    exog = sm.add_constant(df[['experience', 'experience_squared', 'is_married', 'is_union_member']])
    results = sm.OLS(endog, exog).fit().summary().tables[1]
    st.write('''
    We can first attempt a naive OLS regression without controlling for individual fixed effects.
    This is known as a "pooled" regression, since we are pooling all the data together.
    This yields the following results:
    ''')
    st.dataframe(results)
    st.write('''
    We can see the coefficient for is_married is roughly 0.16, which implies that getting married increases wages by 16%.
    This is statistically significant at the 1% level.

    Demeaning our data and estimating a fixed effects model yields the following results:
    ''')

    # Estimate a Fixed Effects Model using OLS and demeaned data
    endog = df['log_wage_demeaned']
    exog = df[['experience_demeaned', 'experience_squared_demeaned', 'is_married_demeaned', 'is_union_member_demeaned']]
    results = sm.OLS(endog, exog).fit().summary().tables[1]
    st.dataframe(results)

    st.write('''
    We can see that the coefficient on is_married is now a much smaller 0.045, and is no longer statistically significant.

    However, even this fixed effects model is misleading. 
    The reason is that we computed our standard errors assuming that the error term is independent across all observations.
    This is not true, since we have multiple observations for each individual.
    We can correct for this by using the cluster-robust standard errors.

    ### Cluster-Robust Standard Errors
    Let's derive an expression for $\\hat{\\beta}$ that will be easier to work with:
    $$
    \\begin{aligned}
    &\\frac{\\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\\\
    &= \\frac{\\sum_{i=1}^n (x_i - \\bar{x})y_i - \\bar{y}\\sum_{i=1}^n (x_i - \\bar{x})}{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\\\
    &= \\frac{\\sum_{i=1}^n (x_i - \\bar{x})y_i}{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\\\
    &= \\frac{\\sum_{i=1}^n (x_i - \\bar{x})(\\beta_0 + \\beta_1 x_i + \\varepsilon_i)}{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\\\
    &= \\frac{\\sum_{i=1}^n (x_i - \\bar{x})\\beta_0 + \\sum_{i=1}^n (x_i - \\bar{x})\\beta_1 x_i + \\sum_{i=1}^n (x_i - \\bar{x})\\varepsilon_i}{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\\\
    &= \\frac{\\sum_{i=1}^n (x_i - \\bar{x})\\beta_1 x_i + \\sum_{i=1}^n (x_i - \\bar{x})\\varepsilon_i}{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\\\
    &= \\beta_1 + \\frac{\\sum_{i=1}^n (x_i - \\bar{x})\\varepsilon_i}{\\sum_{i=1}^n (x_i - \\bar{x})^2} \\\\
    
    \\end{aligned}
    $$
    Where $\\bar{x}$ and $\\bar{y}$ are the sample means of $x$ and $y$ respectively.
    We now want to compute $\\text{Var}(\\hat{\\beta})$:
    $$
    \\begin{aligned}
    \\text{Var}(\\hat{\\beta}) &= \\text{Var}\\left(\\beta_1 + \\frac{\\sum_{i=1}^n (x_i - \\bar{x})\\varepsilon_i}{\\sum_{i=1}^n (x_i - \\bar{x})^2}\\right) \\\\
    &= \\text{Var}\\left(\\frac{\\sum_{i=1}^n (x_i - \\bar{x})\\varepsilon_i}{\\sum_{i=1}^n (x_i - \\bar{x})^2}\\right) \\\\
    &= \\frac{1}{\\left(\\sum_{i=1}^n (x_i - \\bar{x})^2\\right)^2}\\text{Var}\\left(\\sum_{i=1}^n (x_i - \\bar{x})\\varepsilon_i\\right) \\\\
    \\end{aligned}
    $$
    Where we used the fact that $\\beta_1$ is a constant and that $\\text{Var}(aX) = a^2\\text{Var}(X)$.
    Previously, we would have assumed that $\\varepsilon_i$ is independent across all observations, so we could take $\\sum_{i=1}^n (x_i - \\bar{x})$ out of the variance operator.
    This is no longer true, since we have multiple clusters.
    We simply have to grind through a little bit of algebra to get:
    $$
    \\begin{aligned}
    &= \\frac{1}{\\left(\\sum_{i=1}^n (x_i - \\bar{x})^2\\right)^2}\\sum_{i=1}^n\\sum_{j=1}^n \\text{Cov}\\left((x_i - \\bar{x})\\varepsilon_i, (x_j - \\bar{x})\\varepsilon_j\\right) \\\\
    &= \\frac{1}{\\left(\\sum_{i=1}^n (x_i - \\bar{x})^2\\right)^2}\\sum_{i=1}^n\\sum_{j=1}^n (x_i - \\bar{x})(x_j - \\bar{x})\\text{Cov}\\left(\\varepsilon_i, \\varepsilon_j\\right) \\\\
    \\end{aligned}
    $$
    Where we used the fact that $\\text{Cov}(aX, bY) = ab\\text{Cov}(X, Y)$ and that $\\text{var}(X + Y) = \\text{var}(X) + \\text{var}(Y) + 2\\text{Cov}(X, Y)$.
    In matrix form, this is:
    $$
    \\begin{aligned}
    (\\sum_{i=1}^n X^T_iX_i)^{-1}(\\sum_{i=1}^n X^T_i\\varepsilon_i\\varepsilon^T_iX_i)(\\sum_{i=1}^n X^T_iX_i)^{-1}
    \\end{aligned}
    $$

    Let's now re-run our fixed effects model, but this time using cluster-robust standard errors.
    We'll use the `linearmodels` package to do this and compare the results of naive OLS, fixed effects, and fixed effects with cluster-robust standard errors.
    ''')

    df = df.set_index(["i", "t"])

    # Define formulas for each model
    formula_no_fe = (
        "log_wage ~ 1 + experience + experience_squared + is_married + is_union_member"
    )
    formula = formula_no_fe + " + EntityEffects"

    # Compute all models
    model_no_fe = PanelOLS.from_formula(formula_no_fe, data=df).fit()
    model_no_cluster = PanelOLS.from_formula(formula, data=df).fit()
    model = PanelOLS.from_formula(formula, data=df).fit(
        cov_type="clustered", cluster_entity=True
    )

    # Compare results from all models
    results = compare(
        {
            "Pooled OLS": model_no_fe,
            "Fixed Effects without Clustering": model_no_cluster,
            "Fixed Effects": model,
        },
        precision = 'std_errors',
        stars=True,
    ).summary.tables[0].data

    # Format results table
    df_results = pd.DataFrame(results)
    df_results.columns = df_results.iloc[0]
    df_results = df_results[1:]
    df_results = df_results.iloc[11:21]

    with st.expander('Show Code'):
        st.code('''
        df = df.set_index(["i", "t"])

        # Define formulas for each model
        formula_no_fe = (
            "log_wage ~ 1 + experience + experience_squared + is_married + is_union_member"
        )
        formula = formula_no_fe + " + EntityEffects"

        # Compute all models
        model_no_fe = PanelOLS.from_formula(formula_no_fe, data=df).fit()
        model_no_cluster = PanelOLS.from_formula(formula, data=df).fit()
        model = PanelOLS.from_formula(formula, data=df).fit(
            cov_type="clustered", cluster_entity=True
        )

        # Compare results from all models
        results = compare(
            {
                "Pooled OLS": model_no_fe,
                "Fixed Effects without Clustering": model_no_cluster,
                "Fixed Effects": model,
            },
            precision = 'std_errors',
            stars=True,
        ).summary.tables[0].data

        # Format results table
        df_results = pd.DataFrame(results)
        df_results.columns = df_results.iloc[0]
        df_results = df_results[1:]
        df_results = df_results.iloc[11:21]
        ''')

    st.dataframe(df_results)

    st.write('''
    As we can see, the coefficient for `is_married` decreases after we add fixed effects, and the standard error increases as we cluster the standard errors.
    ''')








    





