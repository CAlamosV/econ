import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

#config
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"



def show_rct():
    st.title("Randomized Control Trials (RCTs) and Hypothesis Testing")
    st.write("### Motivation: The Effect of Online Learning on Student Performance")
    st.write("""
    Suppose we want to evaluate the impact of moving a class online on student performance. 
    We can't simply offer two versions of a class, one online and one in-person, and then compare 
    the performance of the two groups of students. This is because the students who choose to 
    take the online class may be different from those who choose to take the in-person class. 
    For example, students who take the online class may be more motivated and self-directed. 
    To account for this, we can randomize students into the two groups. This is called a 
    randomized controlled trial, or RCT for short.
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

    st.write("We now have the following dataframe:")
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
    latext = r'''
    ## Latex example
    ### full equation 
    $$ 
    \Delta G = \Delta\sigma \frac{a}{b} 
    $$ 
    ### inline
    Assume $\frac{a}{b}=1$ and $\sigma=0$...  
    '''
    st.write(latext)
    st.write("We can see that students in the online class are faring worse by $78.54 - 73.63 = 4.91$ points. But how can we be sure this isn't a statistical fluke? To answer this, we'll consider the following question: What is the probability that, given in-person and online classes have the same average grade, we would observe a difference in average grade as large as the one we observed in our data? Let's formulate this mathematically:")
    st.latex(r"\text{Denote }X = \bar{X}_1 - \bar{X}_2 \text{, where } \bar{X}_1 \text{ and } \bar{X}_2 \text{ are the average grades of the in-person and online groups, respectively.}")
    st.latex(r"\text{We want to find }P(|X| > 4.91)")
    st.write("We call this last expression the p-value. It is the probability that, given the null hypothesis that the two groups have the same average grade, we would observe a difference in average grade as large as the one we observed in our data. If this probability is small, then we can reject the null hypothesis and conclude that the two groups do not have the same average grade. Usually, the threshold for rejecting the null hypothesis is 0.05 by convention. We can rewrite the p-value as:")
    st.latex(r"P(|X| > 4.91) = P(X> 4.91) + P(X< -4.91) = 2P(X< -4.91) = 2\Phi\left(\frac{-4.91}{\sigma}\right)")
    st.write("where $\Phi$ is the cumulative distribution function of the standard normal distribution. We can find $\sigma = \sqrt{\operatorname{Var}{X}}$ using:")
    st.latex(r"\sqrt{\operatorname{Var}(X)} = \sqrt{\operatorname{Var}(\bar{X}_1) + \operatorname{Var}(\bar{X}_2)}")
    st.write("We can derive the right hand side of the above expression as follows:")
    st.latex(r"""
    \begin{aligned}
    \text{Var}[\bar{X_1}] &= \text{Var}\left[\frac{1}{n_1} \sum_{i=1}^{n_1} X_i\right] \\
    &= \frac{1}{n_1^2} \text{Var}\left[\sum_{i=1}^{n_1} X_i\right] \\
    &= \frac{1}{n_1^2} \sum_{i=1}^{n_1} \text{Var}[X_i] \\
    &= \frac{1}{n_1^2} \sum_{i=1}^{n_1} \sigma_1^2 \\
    &= \frac{1}{n_1^2} n_1 \sigma_1^2 = \frac{\sigma_1^2}{n_1}
    \end{aligned}
    """)
    st.write("Thus, we have that $\sigma = \sqrt{\operatorname{Var}{X}} =$")
    st.latex(r"\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}")
    st.write("Where $\sigma_1$ and $\sigma_2$ are the standard deviation of exam scores of the two groups, and $n_1$ and $n_2$ are the number of students in each group. We can now compute the p-value:")

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
    """)
