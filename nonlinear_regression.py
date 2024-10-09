import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm

#config
sns.set_style("whitegrid")
sns.set_palette("husl")
# plt.rcParams["font.sans-serif"] = "Arial"
# plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['text.usetex'] = False

def create_index():
    st.markdown('### Table of Contents')
    st.markdown('''
    1. [Interaction Terms](#interaction-terms)
    2. [Quadratic Terms](#quadratic-terms)
    3. [Logarithmic Terms](#logarithmic-terms)
    4. [Omitted Variable Bias](#omitted-variable-bias)
    ''')

def show_nonlinear_regression():
    st.title("Nonlinear Regression")
    create_index()

    st.write(r'''
    ### Interaction Terms

    Building off of the previous section on OLS, let's now return to the question of whether the impact of years of work experience varies if you have a stereotypically black name.
    We can answer this question by adding what's referred to as an interaction term to our model.
    An interaction term is simply the product of two variables.
    In this case, we'll add the interaction between $\text{BlackName}$ and a new variable $\text{Experience}$ to our model. 
    We can write the new model as:

    $$
    y_i = \widehat{\beta}_0 + \widehat{\beta}_1 \text{BlackName}_i + \widehat{\beta}_2 \text{Experience}_i + \widehat{\beta}_3 \text{BlackName}_i \cdot \text{Experience}_i + \varepsilon_i
    $$

    Here, $\widehat{\beta}_3$ captures the difference in the return to experience between a person with a black-sounding name and a person with a white-sounding name.
    To see this, consider the case where BlackName = 1 and Experience = $x_1$. We then have:

    $$
    \begin{aligned}
    E[y_i|\text{Experience}=x_1] &= \widehat{\beta}_0 + \widehat{\beta}_1 \cdot 1 + \widehat{\beta}_2 \cdot x_1 + \widehat{\beta}_3 \cdot 1 \cdot x_1 \\
    &= \widehat{\beta}_0 + \widehat{\beta}_1 + \widehat{\beta}_2 \cdot x_1 + \widehat{\beta}_3 \cdot x_1
    \end{aligned}
    $$

    Now consider the case where BlackName = 1 and Experience = $x_2$. In this case, we have:

    $$
    \begin{aligned}
    E[y_i|\text{Experience}=x_2] &= \widehat{\beta}_0 + \widehat{\beta}_1 \cdot 1 + \widehat{\beta}_2 \cdot x_2 + \widehat{\beta}_3 \cdot 1 \cdot x_2 \\
    &= \widehat{\beta}_0 + \widehat{\beta}_1 + \widehat{\beta}_2 \cdot x_2 + \widehat{\beta}_3 \cdot x_2
    \end{aligned}
    $$

    Thus, the difference in expected value between two levels of experience \(x1\) and \(x2\) for a person with a black-sounding name is:

    $$
    \begin{aligned}
    E[y_i|\text{Experience}=x_1] - E[y_i|\text{Experience}=x_2] \\
    = (\widehat{\beta}_0 + \widehat{\beta}_1 + \widehat{\beta}_2 \cdot x_1 + \widehat{\beta}_3 \cdot x_1) - (\widehat{\beta}_0 + \widehat{\beta}_1 + \widehat{\beta}_2 \cdot x_2 + \widehat{\beta}_3 \cdot x_2) \\
    = \widehat{\beta}_2 \cdot (x1 - x2) + \widehat{\beta}_3 \cdot (x_1 - x_2) \\
    \end{aligned}
    $$

    For a person with a white-sounding name, the difference in expected value is simply 
    $$
    \widehat{\beta}_2 \cdot (x_1 - x_2)
    $$
    Thus, the difference in the return to having $(x_1 - x_2)$ years of experience between a person with a black-sounding name and a person with a white-sounding name is 
    $$
    \widehat{\beta}_2 \cdot (x_1 - x_2) + \widehat{\beta}_3 \cdot (x_1 - x_2) - \widehat{\beta}_2 \cdot (x_1 - x_2) \\
    = \widehat{\beta}_3 \cdot (x_1 - x_2)
    $$
    
    Let's now add the interaction term to our model and run the regression, this time using a package called statsmodels that makes things easier.
    ''')
    with st.expander("Show Code"):
        st.code('''
        import statsmodels.api as sm
        model = sm.formula.ols(
            formula="CallBack ~ BlackName + Experience + BlackName * Experience",
            data=df,
        ).fit()
        model.summary().tables[1]
        ''', language='python')
    df = pd.read_stata('data/bertrand_audit_data.dta')
    df = df[["call", "race", "yearsexp"]]
    df = df.rename(
        columns={
            "call": "CallBack", 
            "race": "BlackName",
            "yearsexp": "Experience"}
    )
    df["BlackName"] = (df["BlackName"] == "b").astype(int)
    model = sm.formula.ols(
        formula="CallBack ~ BlackName + Experience + BlackName * Experience",
        data=df,
    ).fit()
    results = model.summary().tables[1].as_html()
    results = pd.read_html(results, header=0, index_col=0)[0]
    st.dataframe(results)

    st.write(r'''
    As we can see, there is no statistically significant difference in the return to experience between a person with a black-sounding name and a person with a white-sounding name.
    
    ### Quadratic Terms
    We can also use OLS to fit nonlinear models. For instance, we may think that the effect of experience on callback rates is nonlinear.
    This could be because the marginal return to experience decreases as experience increases.
    We can capture this by adding a quadratic term to our model:
    $$
    y_i = \widehat{\beta}_0 + \widehat{\beta}_1 \text{Experience}_i + \widehat{\beta}_2 \text{Experience}_i^2 + \varepsilon_i
    $$

    ''')
    with st.expander("Show Code"):
        st.code('''
        import statsmodels.api as sm
        model = sm.formula.ols(
            formula="CallBack ~ Experience + np.power(Experience, 2)",
            data=df,
        ).fit()
        model.summary().tables[1]
        ''', language='python')
    model = sm.formula.ols(
            formula="CallBack ~  Experience + np.power(Experience, 2)",
            data=df,
        ).fit()
    results = model.summary().tables[1].as_html()
    results = pd.read_html(results, header=0, index_col=0)[0]
    st.dataframe(results)

    st.write(r'''
    The quadratic term is not statistically significant, so we have no evidence that the effect of experience on callback rates is nonlinear.

    ### Logarithmic Terms
    We can also use OLS to fit models that include logarithmic terms. 
    To illustrate this, it's more natural to consider a variable has a heavily skewed distribution, such as wages.
    We can use a dataset from Vella and M. Verbeek (1998) that contains individual-level data on wages and education.

    If we plot the distribution of wages, we can see that it is heavily skewed to the right:
    ''')
    with st.expander("Show Code"):
        st.code('''
        df = pd.read_csv("data/wage_education.csv")
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.histplot(
            data=df,
            x="Hourly Wage (USD)",
            stat="density",
            bins=30,
            ax=ax,
        )
        plt.show()
        
        ''', language='python')
    df = pd.read_csv("data/wage_education.csv")
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(
        data=df,
        x="Hourly Wage (USD)",
        stat="density",
        bins=30,
        ax=ax,
    )
    st.pyplot(fig)
    st.write(r'''
    Taking the log of the wages yields a distribution that is much closer to normal:
    ''')
    with st.expander("Show Code"):
        st.code('''
        df["Log Hourly Wage"] = np.log(df["Hourly Wage (USD)"])
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.histplot(
            data=df,
            x="Log Hourly Wage",
            stat="density",
            bins=30,
            ax=ax,
        )
        plt.show()
        ''', language='python')
    df["Log Hourly Wage"] = np.log(df["Hourly Wage (USD)"])
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(
        data=df,
        x="Log Hourly Wage",
        stat="density",
        bins=30,
        ax=ax,
    )
    st.pyplot(fig)
    st.write(r'''
    Note that there is nothing inherently wrong with having a skewed distribution and using OLS.
    It is simply that taking the log of the variable can make the model easier to interpret.
    In particular, log transformations allow for the interpretation of the coefficients as elasticities.
    Mathematically, this is because for small changes in $x$, we have:
    $$
    \begin{aligned}
    \ln (x+\Delta)-\ln (x)=\ln \left(\frac{x+\Delta}{x}\right)=\ln \left(1+\frac{\Delta}{x}\right) \approx \frac{\Delta}{x}
    \end{aligned}
    $$
    Where the last step follows from the Taylor series approximation $\ln (1+x) \approx x$ for small $x$:
    $$
    \ln (1+x)=x-\frac{x^{2}}{2}+\frac{x^{3}}{3}-\frac{x^{4}}{4}+\cdots
    $$
    Notice that if $x$ is small, then $x^2$, $x^3$, etc. are even smaller, so we can ignore them in practice.

    Thus, in a regression of the following form:

    $$
    \ln y_i = \widehat{\beta}_0 + \widehat{\beta}_1 x_i + \varepsilon_i
    $$
    We can interpret $\widehat{\beta}_1$ as the percentage change in $y$ for a one-unit change in $x$.
    Similarly, in a regression of the following form:
    $$
    y_i = \widehat{\beta}_0 + \widehat{\beta}_1 \ln x_i + \varepsilon_i
    $$
    We can interpret $\widehat{\beta}_1$ as the percentage change in $y$ for a one-percent change in $x$.

    Going back to our example, we can now run a regression of log wages on education:
    $$
    \ln y_i = \widehat{\beta}_0 + \widehat{\beta}_1 \text{Years of Schooling}_i + \varepsilon_i
    $$
    ''')

    with st.expander("Show Code"):
        st.code('''
        endog = df["Log Hourly Wage"]
        exog = sm.add_constant(df["Years of Schooling"])
        sm.OLS(endog, exog).fit().summary().tables[1]
        ''', language='python')
    endog = np.log(df["Hourly Wage (USD)"])
    exog = sm.add_constant(df["Years of Schooling"])
    results = sm.OLS(endog, exog).fit().summary().tables[1]
    st.dataframe(results)

    st.write(r'''
    As we can see, the coefficient for Years of Schooling is roughly 0.058, which means that a one-year increase in education is associated with a 5.8% increase in wages.
    
    ### Omitted Variable Bias
    As we can guess, this estimated effect likely does not capture the true causal effect of education on wages.
    This is because there are likely other factors that affect both education and wages.
    For instance, people who are more intelligent may be more likely to get more education and also earn higher wages.

    To see how this can bias our estimates, let's consider a simple example.
    Suppose that the true model is:
    $$
    y_i = \beta_0 + \beta_1 x_i + \beta_2 z_i + \varepsilon_i
    $$
    Where $z_i$ is a variable that is correlated with $x_i$ and affects $y_i$.
    In this case, $y_i$ is wage, $x_i$ is education, and $z_i$ is intelligence.
    
    If we estimate the following model:
    $$
    y_i = \widehat{\beta}_0 + \widehat{\beta}_1 x_i + \varepsilon_i
    $$
    Then we will get a biased estimate of $\widehat{\beta}_1$:
    $$
    \begin{aligned}
    \widehat{\beta}_1 &= \frac{\operatorname{Cov}\left(x_{i}, y_{i}\right)}{\operatorname{Var}\left(x_{i}\right)} \\
    &= \frac{\operatorname{Cov}\left(x_{i}, \beta_0 + \beta_1 x_i + \beta_2 z_i + \varepsilon_i\right)}{\operatorname{Var}\left(x_{i}\right)} \\
    &= \frac{\beta_1 \operatorname{Var}\left(x_{i}\right) + \operatorname{Cov}\left(x_{i}, \beta_0 + \beta_2 z_i + \varepsilon_i\right)}{\operatorname{Var}\left(x_{i}\right)} \\
    &= \beta_1 + \frac{\operatorname{Cov}\left(x_{i}, \beta_0 + \beta_2 z_i + \varepsilon_i\right)}{\operatorname{Var}\left(x_{i}\right)} \\
    &= \beta_1 + \frac{\beta_2 \operatorname{Cov}\left(x_{i}, z_i\right) + \operatorname{Cov}\left(x_{i}, \beta_0 + \varepsilon_i\right)}{\operatorname{Var}\left(x_{i}\right)} \\
    &= \beta_1 + \frac{\beta_2 \operatorname{Cov}\left(x_{i}, z_i\right)}{\operatorname{Var}\left(x_{i}\right)} + \frac{\operatorname{Cov}\left(x_{i}, \varepsilon_i\right)}{\operatorname{Var}\left(x_{i}\right)} \
    \end{aligned}
    $$
    Where the last step follows from the fact that $\operatorname{Cov}\left(x_{i}, \beta_0\right) = 0$ since $\beta_0$ is a constant and $\operatorname{Cov}\left(x_{i}, \varepsilon_i\right) = 0$ since $\varepsilon_i$ is uncorrelated with $x_i$.
    Thus, we can see that the bias in our estimate of $\widehat{\beta}_1$ is:
    $$
    \beta_2 \frac{\operatorname{Cov}\left(x_{i}, z_i\right)}{\operatorname{Var}\left(x_{i}\right)}
    $$
    This is known as omitted variable bias.
    We can see that the sign of the bias depends on the sign of $\beta_2$ and the sign of $\operatorname{Cov}\left(x_{i}, z_i\right)$.
    In our example, we expect that $\beta_2 > 0$ since more intelligent people earn higher wages and $\operatorname{Cov}\left(x_{i}, z_i\right) > 0$ since more intelligent people are more likely to get more education.
    Thus, we expect that the bias in our estimate of $\widehat{\beta}_1$ is positive, so we will overestimate the effect of education on wages.

    ''')

    