import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import norm

def create_index():
    st.markdown('### Table of Contents')
    st.markdown('''
    1. [Motivation: Quantifying Labor Market Discrimination](#motivation-quantifying-labor-market-discrimination)
    2. [What is OLS?](#what-is-ols)
    3. [Geometric Intuition](#geometric-intuition)
    4. [Standard Errors](#standard-errors)
    5. [Implementing OLS from Scratch](#implementing-ols-from-scratch)
    ''')


def show_ols():
    st.title("Ordinary Least Squares (OLS)")
    create_index()

    st.write(r'''
    ### Motivation: Quantifying Labor Market Discrimination
    [Bertrand and Mullainathan (2004)](https://www.aeaweb.org/articles?id=10.1257/0002828042002561) 
    conducted the following experiment to measure labor market discrimination.
    They sent out resumes to employers with randomly assigned names that were either stereotypically white or black.
    Importantly, they also randomly varied other characteristics of the resumes, such as education and experience.

    If we want to answer the question "Does having a stereotypically black name reduce the probability of getting a callback?",
    we can simply run the difference in means test outlined in the previous section. However, if we want to answer questions like
    "Does the quality of your resume matter more or less if you have a stereotypically black name?", or if we want to quantifity the
    impact of a continuous vairiable like years of education on callback rates, we need a more sophisticated approach.

    ### What is OLS?

    As it turns out, we can answer all of these questions using a single statistical technique: Ordinary Least Squares (OLS).
    All OLS amounts to is fitting a linear model to the data. In the context of the resume experiment, we can write the following linear model:
   
    $$
    y_i = \widehat{\beta}_0 + \widehat{\beta}_1 \text{BlackName}_i + \widehat{\beta}_2 \text{Experience}_i + \varepsilon_i
    $$
    Where $y_i = 1$ if person $i$ received a callback and $y_i = 0$ otherwise, 
    $\text{BlackName}_i = 1$ if the $i$th resume had a stereotypically black name and 0 otherwise, 
    $\text{Experience}_i$ is the number of years of experience on the $i$th resume, 
    and $\varepsilon$ is a random error term with mean 0. 
    We set the mean of $\varepsilon_i$ to 0 to ensure that the expected value of $y_i$ is equal to the expected value of the linear model.
    If we put no restriction on the mean of $\varepsilon$, we could simply let $\varepsilon_i = y_i - \widehat{\beta}_0 - \widehat{\beta}_1 \text{BlackName}_i - \widehat{\beta}_2 \text{Experience}_i$
    and our model would be completely unintresting since it would by definition always perfectly fit the data.

    We'll add more variables to this model later, but for now we'll use two variables to keep things simple.

    Now assume we have a dataset of $n$ individuals. We can write the linear model in matrix form as:
    $$
    y = X\widehat{\beta} + \varepsilon
    $$
    Where:

    $$y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}, \quad X = \begin{bmatrix} 1 & \text{BlackName}_1 & \text{Experience}_1 \\ 1 & \text{BlackName}_2 & \text{Experience}_2 \\ \vdots & \vdots & \vdots \\ 1 & \text{BlackName}_n & \text{Experience}_n \end{bmatrix}, \quad \widehat{\beta} = \begin{bmatrix} \widehat{\beta}_0 \\ \widehat{\beta}_1 \\ \widehat{\beta}_2 \end{bmatrix}, \quad \varepsilon = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}$$
    
    We we want $X\widehat{\beta}$ to be as close as possisble to $y$.
    Mathematically, we want to minimize the following objective function:
    $$
    \operatorname{argmin}_{\widehat{\beta}}||y - X\widehat{\beta}||_2^2
    $$
    We can solve this problem by simply taking the derivative of the objective function with respect to $\widehat{\beta}$ and setting it equal to 0:
    $$
    \begin{aligned}
    \frac{\partial}{\partial \widehat{\beta}} ||y - X\widehat{\beta}||_2^2 &= \frac{\partial}{\partial \widehat{\beta}} (y - X\widehat{\beta})^T(y - X\widehat{\beta}) \\
    &= \frac{\partial}{\partial \widehat{\beta}} (y^Ty - y^TX\widehat{\beta} - \widehat{\beta}^TX^Ty + \widehat{\beta}^TX^TX\widehat{\beta}) \\
    &= -2X^Ty + 2X^TX\widehat{\beta} \\
    &= 0
    \end{aligned}
    $$
    Solving for $\widehat{\beta}$ yields:
    $$
    \widehat{\beta} = (X^TX)^{-1}X^Ty
    $$
    Checking the second derivative, we can verify that this is indeed a minimum:
    $$
    \frac{\partial^2}{\partial \widehat{\beta}^2} ||y - X\widehat{\beta}||_2^2 = 2X^TX \succ 0
    $$
    Notice $X$ must have full column rank for $(X^TX)^{-1}$ to exist and for the second derivative to be positive definite.
    We call this condition the "no multicollinearity" assumption.

    The expression for $\widehat{\beta_j}$ is equivalent to the following:
    $$
    \widehat{\beta}_j = \frac{\sum_{i=1}^n x_iy_i}{\sum_{i=1}^n x_i^2} = \frac{\operatorname{Cov}(x_j, y)}{\operatorname{Var}(x_j)}
    $$
    where $x_j$ is the $j$th column of $X$ and $x_i$ is the value of $x_j$ for the $i$th observation.
    ''')

    st.write(r'''
    ### Geometric Intuition
    While the derivation above is mathematically rigorous, it is not very intuitive.
    We can instead look at OLS from a geometric perspective.
    Suppose $X$ is a 3 $\times$ 2 matrix,  $y$ is a 3 $\times$ 1 vector, and $\widehat{\beta}$ is a 2 $\times$ 1 vector.
    $X$ will span a plane in 3D space, and $X\widehat{\beta}$ will be a vector in this plane.
    Thus, we want to find the vector in the plane that is closest to $y$. This is equivalent to finding the orthogonal projection of $y$ onto the plane spanned by $X$.

    We can visualize this as follows, where the grey plane is spanned by $X$,  the light blue vector is $y$, the dark blue vector is $X\widehat{\beta}$, and the dashed grey vector is $y - X\widehat{\beta}$:
    ''')

    # Create sliders for the y vector
    size = 5.0
    # Make plane parallel to z = 0
    z = np.zeros((30, 30))
    y = np.linspace(-size, size, 30)
    x = np.linspace(-size, size, 30)
    y, x = np.meshgrid(y, x)

    # Create 3D Plot
    fig = go.Figure()

    # Add plane
    fig.add_trace(go.Surface(z=z, x=x, y=y, opacity=0.6, showscale=False))

    # Add y vector
    fig.add_trace(go.Scatter3d(x=[0, -2], y=[0, 2], z=[0, 2],
                            marker=dict(size=[0, 6]), line=dict(width=6), name='y', hoverinfo='skip'))

    # Add projection vectors
    fig.add_trace(go.Scatter3d(x=[0, -2], y=[0, 2], z=[0, 0],
                            marker=dict(size=[0, 6]), line=dict(width=6, color='blue'), name='Projection of y onto X', hoverinfo='skip'))

    # add y - X\widehat{\beta} vector
    fig.add_trace(go.Scatter3d(x=[-2, -2], y=[2, 2], z=[2, 0],
                            marker=dict(size=[6, 6]), line=dict(width=6, color='grey', dash='dash'), name='', hoverinfo='skip'))

    # Layout options remain the same
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[-size, 3], showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(nticks=4, range=[-3, size], showgrid=True, zeroline=False, showticklabels=False),
        zaxis=dict(nticks=4, range=[-3, size], showgrid=True, zeroline=False, showticklabels=False),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=.5, y=.5, z=.2)
        )),
        margin=dict(l=0, r=0, b=0, t=0),
        width=800,
        legend=dict(
        x=0,  # x coordinate (0 to 1, left to right)
        y=1,  # y coordinate (0 to 1, bottom to top)
        traceorder="normal",
        orientation="v",
        xanchor='left',
        yanchor='bottom'
        ))

    with st.expander("Show Code"):
        st.code('''
        fig = go.Figure()

        # Add plane
        fig.add_trace(go.Surface(z=z, x=x, y=y, opacity=0.6, showscale=False))

        # Add y vector
        fig.add_trace(go.Scatter3d(x=[0, -2], y=[0, 2], z=[0, 2],
                                marker=dict(size=[0, 6]), 
                                line=dict(width=6), 
                                name='y', 
                                hoverinfo='skip'))

        # Add projection vecto
        fig.add_trace(go.Scatter3d(x=[0, -2], y=[0, 2], z=[0, 0],
                                marker=dict(size=[0, 6]), 
                                line=dict(width=6, color='blue'), 
                                name='Projection of y onto X', 
                                hoverinfo='skip'))

        # add y - XBeta vector
        fig.add_trace(go.Scatter3d(x=[-2, -2], y=[2, 2], z=[2, 0],
                                marker=dict(size=[6, 6]), 
                                line=dict(width=6, color='grey', dash='dash'),
                                name='',
                                hoverinfo='skip'))

        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[-size, 3], zeroline=False, showticklabels=False),
                yaxis=dict(nticks=4, range=[-3, size], zeroline=False, showticklabels=False),
                zaxis=dict(nticks=4, range=[-3, size], zeroline=False, showticklabels=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=.5, y=.5, z=.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            width=800,
            legend=dict(
                x=0, 
                y=1,  
                traceorder="normal",
                orientation="v",
                xanchor='left',
                yanchor='bottom'
            )
        )
        ''', language='python')

    st.plotly_chart(fig)

    st.write(r'''
    We can use this visualization to derive the OLS estimator in a more intuitive manner.
    Notice that the vector $y - X\widehat{\beta}$ is orthogonal to the plane spanned by $X$.
    Thus, we can write:
    $$
    \begin{aligned}
    X^T(y - X\widehat{\beta}) &= 0 \\
    X^Ty - X^TX\widehat{\beta} &= 0 \\
    X^Ty &= X^TX\widehat{\beta} \\
    \widehat{\beta} &= (X^TX)^{-1}X^Ty
    \end{aligned}
    $$

    ### Standard Errors
    Now that we have an estimator for $\widehat{\beta}$, we need a way to quantify the uncertainty in our estimate.
    We can do this by calculating the variance of $\widehat{\beta}$.
    $$
    \begin{aligned}
    \operatorname{Var}(\widehat{\beta}) &= \operatorname{Var}((X^TX)^{-1}X^Ty) \\
    &= (X^TX)^{-1}X^T\operatorname{Var}(y)X(X^TX)^{-1} \\
    &= (X^TX)^{-1}X^T\sigma^2X(X^TX)^{-1} \\
    &= \sigma^2(X^TX)^{-1} \\
    &= \left[\begin{array}{cccc}\operatorname{Var}[\hat{\beta}_0] & \operatorname{Cov}[\hat{\beta}_0, \hat{\beta}_1] & \cdots & \operatorname{Cov}[\hat{\beta}_0, \hat{\beta}_p] \\ \operatorname{Cov}[\hat{\beta}_1, \hat{\beta}_0] & \operatorname{Var}[\hat{\beta}_1] & \cdots & \operatorname{Cov}[\hat{\beta}_1, \hat{\beta}_p] \\ \vdots & \vdots & \ddots & \vdots \\ \operatorname{Cov}[\hat{\beta}_p, \hat{\beta}_0] & \operatorname{Cov}[\hat{\beta}_p, \hat{\beta}_1] & \cdots & \operatorname{Var}[\hat{\beta}_p]\end{array}\right]
    \end{aligned}
    $$
    Where $\sigma^2$ is the variance of $\varepsilon$ (and of $y$).
    We only care about the diagonal elements of this matrix, since we are focusing on the variance of each individual coefficient.
    We can estimate $\sigma^2$ using the residuals from the regression:
    $$
    \widehat{\sigma}^2 = \frac{1}{n - k} \sum_{i=1}^n (y_i - \widehat{y}_i)^2 = \frac{1}{n - k} \varepsilon^T\varepsilon
    $$
    Where $k$ is the number of variables in the regression (including the intercept).
    We subtract $k$ from $n$ because we lose one degree of freedom for each variable we add to the regression.
    We can then calculate the standard error of $\widehat{\beta}$ as:
    $$
    \widehat{\operatorname{SE}}(\widehat{\beta}) = \sqrt{\operatorname{diag}(\widehat{\operatorname{Var}}(\widehat{\beta}))} = \sqrt{\widehat{\sigma}^2\operatorname{diag}(X^TX)^{-1}} = \sqrt{\frac{1}{n - k} \varepsilon^T\varepsilon \operatorname{diag}((X^TX)^{-1})}
    $$
    Typically, we indicate which coefficients are significantly different from 0 by conducting a t-test using standard errors, as outlined in the previous section.
    ''')

    df = pd.read_stata('data/bertrand_audit_data.dta')
    df = df[["call", "race", "yearsexp"]]
    df = df.rename(
        columns={
            "call": "CallBack", 
            "race": "BlackName",
            "yearsexp": "Experience"}
    )
    df["BlackName"] = (df["BlackName"] == "b").astype(int)

    st.write(r'''
    ### Implementing OLS from Scratch
    Let's now apply OLS to the resume experiment.
    We'll begin by loading and cleaning the data from Bertrand and Mullainathan (2004).
    ''')
    with st.expander("Show Code"):
        st.code('''
        df = pd.read_stata('data/bertrand_audit_data.dta')
        df = df[["call", "race", "yearsexp"]]
        df = df.rename(
            columns={
                "call": "CallBack", 
                "race": "BlackName",
                "yearsexp": "Experience"}
        )
        df["BlackName"] = (df["BlackName"] == "b").astype(int))
        df.head()
        ''', language='python')
    st.dataframe(df.head(3))

    # Define X and y
    X = df[["BlackName", "Experience"]].to_numpy()
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept
    y = df["CallBack"].to_numpy()
    # Compute beta hat
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    # Compute standard error of beta hat
    se_beta_hat = np.sqrt(
        np.diag(
            np.linalg.inv(X.T @ X)
            * ((y - X @ beta_hat).T @ (y - X @ beta_hat))
            / (X.shape[0] - X.shape[1])
        )
    )
    # Compute t-statistic
    t_stat = beta_hat / se_beta_hat
    # Compute p-value
    p_val = 2 * (1 - norm.cdf(abs(t_stat)))
    # Display results
    results = pd.DataFrame(
        {
            "Coefficient": beta_hat,
            "Standard Error": se_beta_hat,
            "p-value": p_val,
        },
        index=["Intercept", "BlackName", "Experience"],
    )

    st.write(r'''
    We can now run the regression and display the results:
    ''')

    with st.expander("Show Code"):
        st.code('''
        # Define X and y
        X = df[["BlackName", "Experience"]].to_numpy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept
        y = df["CallBack"].to_numpy()
        # Compute beta hat
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        # Compute standard error of beta hat
        se_beta_hat = np.sqrt(
            np.diag(
                np.linalg.inv(X.T @ X)
                * ((y - X @ beta_hat).T @ (y - X @ beta_hat))
                / (X.shape[0] - X.shape[1])
            )
        )
        # Compute t-statistic
        t_stat = beta_hat / se_beta_hat
        # Compute p-value
        p_val = 2 * (1 - norm.cdf(abs(t_stat)))
        # Display results
        results = pd.DataFrame(
            {
                "Coefficient": beta_hat,
                "Standard Error": se_beta_hat,
                "p-value": p_val,
            },
            index=["Intercept", "BlackName", "Experience"],
        )
        ''', language='python')


    st.dataframe(results)

    st.write(r'''
    Notice the coefficient for BlackName is -0.032.
    That is, if BlackName = 1, then $y$ is expected to decrease by 0.032.
    In other words, having a stereotypically black name reduces the probability of getting a callback by 3.2 percentage points.
    This is a statistically significant result, since the p-value is less than 0.05.

    We can also see that the coefficient for Experience is 0.003.
    That is, if Experience increases by 1 year, then $y$ is expected to increase by 0.003.
    In other words, an additional year of experience increases the probability of getting a callback by a statistically significant 0.3 percentage points.
    
    To answer our original question of "Does the quality of your resume matter more or less if you have a stereotypically black name?", we need to introduce the concept of nonlinear regression.
    ''')


    
    







