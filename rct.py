import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
# set latex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=11)

#config
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"

def create_index():
    st.markdown('### Table of Contents')
    st.markdown('''
    1. [Introduction and Motivation](#introduction-and-motivation)
    2. [Hypothesis Testing: The Difference in Means Test](#hypothesis-testing-the-difference-in-means-test)
    3. [The Potential Outcomes Framework](#the-potential-outcomes-framework)
    ''')

def show_rct():
    st.title("Randomized Control Trials (RCTs) and Hypothesis Testing")
    create_index()
    st.write("### Introduction and Motivation")
    st.write("""
    Economists often seek to understand the causal effect of one variable on another, such as the effect of a new policy on economic growth or the effect of online learning on student performance.
    However, establishing causality by simply looking at correlations (e.g., between education and earnings) is
    challenging due to the presence of what are known as confounding variables.

    To illustrate this, suppose that we want to evaluate the impact of moving a class online on student performance. 
    We can't simply offer two versions of a class, one online and one in-person, and then compare 
    the performance of the two groups of students. This is because the students who choose to 
    take the online class may be different from those who choose to take the in-person class. 
    For instance, students who take the online class may be more motivated and self-directed. 

    To account for this, we can randomize students into the two groups: online (the "treatment" group) and in-person (the "control" group).
    By randomly assigning subjects to treatment and control, RCTs create comparable groups that differ only in their exposure to the treatment.
    This means that any differences in outcomes between the two groups can be attributed to the treatment itself, rather than to pre-existing differences between the groups.
    """)

    st.write("""[Alpert et al. 2016](https://www.aeaweb.org/articles?id=10.1257/aer.p20161057) conduct exactly this analysis. Let's first clean their raw data:""")

    @st.cache_data
    def load_and_process_data():
        df = pd.read_stata("data/online_learning_rct.dta")
        df = (
            df[(df["enroll_count"] == 3) & (df["format_blended"] == 0)]
            .assign(format_ol=lambda x: x["format_ol"].astype(bool))
            .rename(
                columns={
                    "falsexam": "Final Exam Score",
                    "format_ol": "Online Format",
                }
            )[["Online Format", "Final Exam Score"]]
        )
        return df
    
    
    df = load_and_process_data()

    # Display the code to the viewer as a collapsable section
    with st.expander("Show Code"):
        st.code("""
        def load_and_process_data():
            # Load data from a Stata file located at "data/online_learning_rct.dta"
            df = pd.read_stata("data/online_learning_rct.dta")

            # Apply a series of transformations to the DataFrame:
            # 1. Filter rows where 'enroll_count' is 3 and 'format_blended' is 0
            # 2. Convert the "format_ol" column to boolean type
            # 3. Rename the columns "falsexam" to "Final Exam Score" and "format_ol" to "Online Format"
            # 4. Keep only the "Online Format" and "Final Exam Score" columns
            df = (
                df[(df["enroll_count"] == 3) & (df["format_blended"] == 0)]
                .assign(format_ol=lambda x: x["format_ol"].astype(bool))
                .rename(
                    columns={
                        "falsexam": "Final Exam Score",
                        "format_ol": "Online Format",
                    }
                )[["Online Format", "Final Exam Score"]]
            )
            return df
        """, language='python')

    st.write("We've now wrangled the data into the following format:")
    temp = df.copy()
    temp["Online Format"] = temp["Online Format"].astype(str)
    st.dataframe(temp.head(3), hide_index=True, width=350)

    st.write("We can then plot the distribution of grades for each group:")
    with st.expander("Show Code"):
        st.code("""
        fig, ax = plt.subplots(figsize=(7, 3)) # Set the size of the figure
        # Plot the distribution of grades for each group
        sns.histplot(
            data=df,
            x="Final Exam Score",
            hue="Online Format", 
            stat = "density",
            bins=30,
            palette="husl",
            ax=ax,
        )
        # Show the mean of each group with vertical lines
        plt.axvline(
            x=df[df["Online Format"]]["Final Exam Score"].mean(),
            color=sns.color_palette("husl")[3]
        )
        plt.axvline(
            x=df[~df["Online Format"]]["Final Exam Score"].mean(),
            color=sns.color_palette("husl")[0]
        )
        # Add a grid to the plot and remove the y-axis ticks and label
        plt.grid(axis="x")
        ax.set_yticks([])
        ax.set_ylabel('')
        st.pyplot(fig)"""
        , language='python')


    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(
        data=df,
        x="Final Exam Score",
        hue="Online Format", 
        stat = "density",
        bins=30,
        palette="husl",
        ax=ax,
    )
    plt.axvline(
        x=df[df["Online Format"]]["Final Exam Score"].mean(),
        color=sns.color_palette("husl")[3]
    )
    plt.axvline(
        x=df[~df["Online Format"]]["Final Exam Score"].mean(),
        color=sns.color_palette("husl")[0]
    )
    plt.grid(axis="x")
    ax.set_yticks([])
    ax.set_ylabel('')
    st.pyplot(fig)

    # Mean of each group
    mean_df = (
        temp.groupby("Online Format")
        .mean()
        .reset_index()
        .rename(columns={"Final Exam Score": "Mean Final Exam Score"})
    )
    st.write("We can also compute the numerical values for the means of each group:")
    with st.expander("Show Code"):
        st.code("""
        (df
            .groupby("Online Format")
            .mean()
            .reset_index()
            .rename(columns={"Final Exam Score": "Mean Final Exam Score"})
        )
        """, language='python')
    st.dataframe(mean_df, hide_index=True, width=350)
    # Convert Markdown to st.write and st.latex
    st.write("### Hypothesis Testing: The Difference in Means Test")
    st.write(r'''
    We can see that students in the online class are faring worse by 78.54 - 73.63 = 4.91 points. But how can we be sure this isn't a statistical fluke?  
    
    To answer this, we'll consider the following question: What is the probability that, given in-person and online classes have the same average grade, we would observe a difference in average grade as large or larger than the one we observed in our data? Let's formulate this mathematically.

    Let $X$ denote the difference in means between the two groups. We can write this as $X = \bar{X}_1 - \bar{X}_2$ where $\bar{X}_1$ and $\bar{X}_2$ are the sample means of the in-person and online classes respectively. 
    We can then write the probability we're interested in as $P(|X| > 4.91)$. We call this the p-value. It is the probability that, given the null hypothesis that the two groups have the same average grade (i.e. $\bar{X}=0$), we would observe a difference in average grade as large or larger than the one we observed in our data. 
    If this probability is small, then we can reject the null hypothesis and conclude that the two groups do not have the same average grade. Usually, the threshold for rejecting the null hypothesis is 0.05 by convention.

    Notice that  
    $P(|X| > 4.91) = P(X> 4.91) + P(X< -4.91) = 2P(X< -4.91) = 2\Phi\left(\frac{-4.91}{\sigma}\right)$

    To evaluate this expression, we need to determine $\sigma = \sqrt{\operatorname{Var}{X}}$. 

    Recall $X = \bar{X}_1 - \bar{X}_2$, which implies that $\sqrt{Var(X)} = \sqrt{Var(\bar{X}_1) + Var(\bar{X}_2)}$.

    We can derive the variance of $\bar{X_1}$ and $\bar{X_2}$ follows:  
    $$
    \begin{aligned} 
    \operatorname{Var}\left[\bar{X}_1\right] &= \operatorname{Var}\left[\frac{1}{n_1} \sum_{i=1}^{n_1} X_i\right] \\
    &= \frac{1}{n_1^2} \operatorname{Var}\left[\sum_{i=1}^{n_1} X_i\right] \\
    &= \frac{1}{n_1^2} \sum_{i=1}^{n_1} \operatorname{Var}\left[X_i\right] \\
    &= \frac{1}{n_1^2} \sum_{i=1}^{n_1} \sigma_1^2 \\
    &= \frac{1}{n_1^2} n_1 \sigma_1^2 \\
    &= \frac{\sigma_1^2}{n_1}
    \end{aligned}
    $$

    Thus, $\sigma = \sqrt{\operatorname{Var}{X}} = \sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}$

    Now, let's calculate the p-value:
    ''')

    online = df[df["Online Format"]]["Final Exam Score"]
    in_person = df[~df["Online Format"]]["Final Exam Score"]
    sigma1, mu1, n1 = online.std(), online.mean(), online.shape[0]
    sigma2, mu2, n2 = in_person.std(), in_person.mean(), in_person.shape[0]

    sd = np.sqrt(((sigma1**2) / n1) + ((sigma2**2) / n2))
    p_val = 2 * norm.cdf(-abs(mu1 - mu2) / sd)
    p_val = np.round(p_val, 4)

    with st.expander("Show Code"):
        st.code("""
            # Get a list of the exam scores for each group
            online = df[df["Online Format"]]["Final Exam Score"]
            in_person = df[~df["Online Format"]]["Final Exam Score"]

            # Calculate the standard deviation, mean, and number of students in each group
            sigma1, mu1, n1 = online.std(), online.mean(), online.shape[0]
            sigma2, mu2, n2 = in_person.std(), in_person.mean(), in_person.shape[0]

            # Calculate the standard deviation of X
            sd = np.sqrt(((sigma1**2) / n1) + ((sigma2**2) / n2))

            # Calculate and display the p-value
            p_val = 2 * norm.cdf(-abs(mu1 - mu2) / sd)
            print(f"p-value: {p_val:.4f}")
            """
            )
    st.write("p-value: ",  p_val)
    st.write("""
    We have shown that, if there were no difference between the two groups, 
    the probability of observing a difference at least as large as the one we observed is 0.0054.
    We can therefore reject the null hypothesis at the 5% level and conclude that the two groups do not have the same average grade.

    ### The Potential Outcomes Framework
    We previously stated that we need to randomize students into the two groups to ensure that the two groups are comparable and there's no selection bias.
    But what does this mean exactly? To move forward, we need to have a precise mathematical understanding of terms like "selection bias".
    While it may seem slightly pedantic at first, building a solid foundation of the potential outcomes framework will help us understand many of the identification strategies we'll cover later on.
    This particular introduction to the potential outcomes framework is based on [Walters (2021)](https://assets.aeaweb.org/asset-server/files/13709.pdf).

    Let $Y_i(1)$ denote individual $i$'s grade if they take the class online and $Y_i(0)$ denote their grade if they take the class in-person.
    The causal effect of taking the class online on person $i$ is then defined as 
    $$
    \\delta_i = Y_i(1) - Y_i(0).
    $$
    Let $D_i = 1$ if person $i$ takes the class online and $D_i = 0$ if they take the class in-person.
    We can then write
    $$
    \\begin{aligned}
    Y_i=Y_i(0)+\\left(Y_i(1)-Y_i(0)\\right) D_i
    \\end{aligned}
    $$
    Of course, we can never observe both $Y_i(1)$ and $Y_i(0)$ for the same person $i$. This is known as the fundamental problem of causal inference.

    In this example, we are interested in finding the average treatment effect (ATE), defined as
    $$
    \\begin{aligned}
    \\delta = E[Y_i(1) - Y_i(0)].
    \\end{aligned}
    $$

    Other parameters of interest include the effect of treatment on the treated (TOT), defined as
    $$
    \\begin{aligned}
    \\text{TOT} = E[Y_i(1) - Y_i(0) | D_i = 1]
    \\end{aligned}
    $$
    and the effect of treatment on the non-treated (TNT), defined as
    $$
    \\begin{aligned}
    \\text{TNT} = E[Y_i(1) - Y_i(0) | D_i = 0].
    \\end{aligned}
    $$

    Now, consider a comparison of the average grade of students who take the class online and those who take the class in-person.
    Assume we haven't randomized students into the two groups.

    Simply comparing the average grade of the two groups will yield the following:
    $$
    \\begin{aligned}
    E\\left[Y_i \\mid D_i=\\right. & 1]-E\\left[Y_i \\mid D_i=0\\right] \\\\
    &= E\\left[Y_i(1) \\mid D_i=1\\right]-E\\left[Y_i(0) \\mid D_i=0\\right] \\\\
    &= \\underbrace{E\\left[Y_i(1)-Y_i(0) \\mid D_i=1\\right]}_{\\text {TOT }} 
    +\\underbrace{E\\left[Y_i(0) \\mid D_i=1\\right]-E\\left[Y_i(0) \\mid D_i=0\\right]}_{\\text {Selection Bias}}
    \\end{aligned}
    $$

    Where the last line is derived by subtracting and adding $E[Y_i(0) | D_i = 1]$.

    However, if we run an RCT, we assign treatment randomly, so $D_i$ is independent of $Y_i(0)$ and $Y_i(1)$:
    $$
    \\begin{aligned}
    E\\left[Y_i \\mid D_i=1\\right]-E\\left[Y_i \\mid D_i=0\\right]=E\\left[Y_i(1) \\mid D_i=1\\right]-E\\left[Y_i(0) \\mid D_i=0\\right] \\\\
    =E\\left[Y_i(1)\\right]-E\\left[Y_i(0)\\right] \\\\
    =\\text{ATE}
    \\end{aligned}
    $$

    """)
