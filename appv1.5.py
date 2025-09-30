"""
title: ASCVIT V1.5 [AUTOMATIC STATISTICAL CALCULATION, VISUALIZATION AND INTERPRETATION TOOL]
author: stefanpietrusky
author_url1: https://downchurch.studio/
author_url2: https://urlz.fr/uj1I [CAEDHET/HCDH Heidelberg University]
version: 1.0
"""


import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  
import plotly.graph_objects as go  
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import streamlit as st


# Function for displaying data types and statistics
def display_data_info(df):
    st.write("**Data description:**")
    st.write(df.describe())
    st.write(f"**Number of data points:** {len(df)}")
    
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    st.write("**Numerical variables:** ", ", ".join(numerical_columns))
    st.write("**Categorical variables:** ", ", ".join(categorical_columns))
    
    return numerical_columns, categorical_columns


# Function for performing descriptive statistics
def descriptive_statistics(df, numerical_columns):
    # Selection of the diagram type
    chart_type = st.selectbox("Select the diagram:", ["Histogram", "Boxplot", "Pairplot", "Correlation matrix"])
    
    # Show explanations based on the selection 
    if chart_type == "Histogram":
        st.markdown("""
        **Histogram:**
        A histogram shows the distribution of a numerical variable. It helps to 
        recognize how frequently certain values occur in the data and whether there are patterns, such as a normal distribution.
        """)
    elif chart_type == "Boxplot":
        st.markdown("""
        **Boxplot:**
        A boxplot shows the distribution of a numerical variable through its quartiles. 
        It helps to identify outliers and visualize the dispersion of the data.
        """)
    elif chart_type == "Pairplot":
        st.markdown("""
        **Pairplot:**
        A pairplot shows the relationships between different numerical variables through scatterplots.
        It helps to identify possible relationships between variables.
        """)
    elif chart_type == "Correlation matrix":
        st.markdown("""
        **Correlation matrix:**
        The correlation matrix shows the linear relationships between numerical variables.
        A positive correlation indicates that high values in one variable also correlate with high values in another.
        """)

    # Selection of variables
    if chart_type in ["Pairplot", "Correlation matrix"]:
        selected_vars = st.multiselect("Select variables:", numerical_columns, default=numerical_columns)
    else:
        selected_vars = [st.selectbox("Select a variable:", numerical_columns)]
    
    # Only show logarithmic scaling if it is not "correlation matrix"
    if chart_type != "Correlation matrix":
        apply_log_scale = st.checkbox("Apply logarithmic scaling?", value=False)
    else:
        apply_log_scale = False 
    
    # Button to start the analysis
    if st.button("Create diagram"):
        if chart_type == "Histogram":
            plot_histogram(df, selected_vars[0], apply_log_scale)
        elif chart_type == "Boxplot":
            plot_boxplot(df, selected_vars[0], apply_log_scale)
        elif chart_type == "Pairplot":
            plot_pairplot(df, selected_vars)
        elif chart_type == "Correlation matrix":
            plot_correlation_matrix(df, selected_vars)


# Function for displaying a histogram with LLM analysis
def plot_histogram(df, variable, apply_log_scale):
    # Cleaning the data (removing NaN values)
    cleaned_data = df[variable].dropna()

    # Calculating metrics
    mean_value = cleaned_data.mean()
    median_value = cleaned_data.median()
    std_value = cleaned_data.std()
    min_value = cleaned_data.min()
    max_value = cleaned_data.max()

    # Calculation of mean + and - standard deviation
    std_upper = mean_value + std_value
    std_lower = max(0, mean_value - std_value)  

    # Concentration range (mean ± 1 standard deviation)
    concentration_range = (mean_value - std_value, mean_value + std_value)

    # Description of the scatter
    if std_value < (max_value - min_value) / 6:
        scatter = "low"
    elif std_value < (max_value - min_value) / 3:
        scatter = "moderate"
    else:
        scatter = "high"

    # Description of the distribution
    if abs(mean_value - median_value) < 0.1 * std_value:
        distribution = "symmetrical"
    elif mean_value > median_value:
        distribution = "right-skewed"
    else:
        distribution = "left-skewed"

    # Creating the histogram
    fig, ax = plt.subplots()
    ax.hist(cleaned_data, bins=30, edgecolor='black', alpha=0.7)

    # Add vertical lines for mean, median and standard deviation
    ax.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
    ax.axvline(median_value, color='green', linestyle='-', label=f'Median: {median_value:.2f}')
    ax.axvline(std_upper, color='blue', linestyle=':', label=f'+1 Std: {std_upper:.2f}')
    ax.axvline(std_lower, color='blue', linestyle=':', label=f'-1 Std: {std_lower:.2f}')

    # Title and legend
    ax.set_title(f"Histogram of {variable}")
    ax.legend(title=f'Std-Deviation: {std_value:.2f}')

    # Log scaling if desired
    if apply_log_scale:
        ax.set_yscale('log')

    # Display of the graphic in Streamlit
    st.pyplot(fig)

    # Providing a standardized structure for LLM analysis
    context = (
        f"Here is an analysis of the distribution of the variable '{variable}':\n"
        f"- Mean: {mean_value:.2f}\n"
        f"- Median: {median_value:.2f}\n"
        f"- Standard deviation: {std_value:.2f}\n"
        f"- Minimum: {min_value:.2f}\n"
        f"- Maximum: {max_value:.2f}\n\n"
        f"The distribution of the data shows a {distribution} distribution.\n"
        f"The small difference between mean and median indicates a {distribution} distribution.\n"
        f"A strong concentration of data points is observed between {concentration_range[0]:.2f} and {concentration_range[1]:.2f}.\n"
        f"The scatter of the data is described as {scatter}, indicating a relatively tight distribution around the mean.\n\n"
        "Provide a standalone interpretation of this distribution without conversational language, focusing on symmetry, scatter, and potential deviations."
    )

    # Send request to the LLM
    response = query_llm_via_cli(context)
    st.write(f"**Histogram Interpretation:** {response}")


# Function for displaying a boxplot with LLM analysis
def plot_boxplot(df, variable, apply_log_scale):
    # Calculating metrics for the boxplot
    mean_value = df[variable].mean()
    median_value = df[variable].median()
    std_value = df[variable].std()
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1
    lower_whisker = max(df[variable].min(), q1 - 1.5 * iqr)
    upper_whisker = min(df[variable].max(), q3 + 1.5 * iqr)

    # Creating the boxplot
    fig = px.box(df, y=variable)
    fig.update_layout(title=f"Boxplot of {variable}")

    if apply_log_scale:
        fig.update_yaxes(type="log")  
    
    st.plotly_chart(fig)

    # Context for submission to the LLM
    context = (
        f"Here is an analysis of the distribution of the variable '{variable}' based on a boxplot:\n"
        f"- Mean: {mean_value:.2f}\n"
        f"- Median: {median_value:.2f}\n"
        f"- Standard deviation: {std_value:.2f}\n"
        f"- Lower quartile (Q1): {q1:.2f}\n"
        f"- Upper quartile (Q3): {q3:.2f}\n"
        f"- Interquartile range (IQR): {iqr:.2f}\n"
        f"- Potential outliers outside values from {lower_whisker:.2f} to {upper_whisker:.2f}.\n"
        "Interpret this distribution in a concise, non-conversational manner, identifying any patterns or outliers as shown in the boxplot."
    )

    # Send request to the LLM
    response = query_llm_via_cli(context)
    st.write(f"**Boxplot Interpretation:** {response}")


# Function to calculate regression data
def calculate_regression_stats(df, selected_vars):
    regression_results = []
    for var1 in selected_vars:
        for var2 in selected_vars:
            if var1 != var2:
                # Remove NaN values from both variables to ensure that the number of data points matches
                non_nan_data = df[[var1, var2]].dropna()

                X = non_nan_data[[var1]].values.reshape(-1, 1)
                y = non_nan_data[var2].values

                # Perform the regression analysis only if both variables contain data after removing NaNs
                if len(X) > 0 and len(y) > 0:
                    model = LinearRegression()
                    model.fit(X, y)
                    r_squared = model.score(X, y)
                    slope = model.coef_[0]

                    regression_results.append((var1, var2, slope, r_squared))

    return regression_results


# Function to display a pairplot with LLM analysis
def plot_pairplot(df, selected_vars):
    if len(selected_vars) > 1:
        st.write("**Pairplot with regression lines:**")
        pairplot_fig = sns.pairplot(df[selected_vars], kind='reg', diag_kind='kde', 
                                    plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'color': 'blue'}})
        st.pyplot(pairplot_fig.fig)

        # Calculate the correlation matrix
        corr_matrix = df[selected_vars].corr()

        # Calculate regression statistics
        regression_stats = calculate_regression_stats(df, selected_vars)

        # Create a neutral context for the analysis
        correlation_list = "\n".join(
            [f"The correlation between {var1} and {var2} is {corr_matrix.at[var1, var2]:.2f}."
             for var1 in corr_matrix.columns for var2 in corr_matrix.columns if var1 != var2]
        )

        regression_list = "\n".join(
            [f"The regression line for {var1} and {var2} has a slope of {slope:.2f} and an R² of {r_squared:.2f}."
             for var1, var2, slope, r_squared in regression_stats]
        )

        # Create a neutral context for the analysis
        context = (
            f"Here are the correlation and regression analyses between the selected variables:\n"
            f"{correlation_list}\n\n"
            f"{regression_list}\n\n"
            "Provide a clear statistical interpretation of the relationships between the selected variables, based on correlation and regression results. "
            "Do not use conversational language; focus on explaining any significant patterns observed in the data."
        )

        # Send request to the LLM
        response = query_llm_via_cli(context)
        st.write(f"**Pairplot Interpretation:** {response}")
    else:
        st.error("At least two variables must be selected for a pairplot.")


# Function to display the correlation matrix with LLM analysis
def plot_correlation_matrix(df, selected_vars):
    if len(selected_vars) > 1:
        corr_matrix = df[selected_vars].corr()

        # Display the correlation matrix
        fig, ax = plt.subplots(figsize=(12, 10))  # Increase the figure size for better readability
        sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 10}, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix", fontsize=16)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        st.pyplot(fig)

        # Identify significant correlations
        high_correlations = []
        for var1 in corr_matrix.columns:
            for var2 in corr_matrix.columns:
                if var1 != var2 and abs(corr_matrix.at[var1, var2]) >= 0.5:  # Only significant correlations
                    if (var2, var1) not in [(v1, v2) for v1, v2, _ in high_correlations]:  # Avoid duplicates
                        high_correlations.append((var1, var2, corr_matrix.at[var1, var2]))

        # Create context for the LLM
        if high_correlations:
            correlation_list = "\n".join([f"- {var1} and {var2} have a correlation value of {value:.2f}, "
                                        f"indicating a {'strong' if abs(value) > 0.7 else 'moderate' if abs(value) > 0.5 else 'weak'} correlation."
                                        for var1, var2, value in high_correlations])

            context = (
                f"Here is an analysis of the significant correlations between the selected variables in the correlation matrix:\n"
                f"{correlation_list}\n\n"
                "Provide an objective interpretation of the correlations, focusing on their strength and significance. "
                "Do not include conversational or exploratory language; describe only the statistical relationships and patterns observed."
            )

            # Send request to the LLM
            response = query_llm_via_cli(context)
            st.write(f"**Model Response:** {response}")
        else:
            st.write("**No significant correlations were found.**")
    else:
        st.write("**The correlation matrix cannot be displayed because fewer than two variables were selected.**")


# Function for hypothesis tests
def hypothesis_testing(df, numerical_columns, categorical_columns):
    test_type = st.selectbox("Choose the hypothesis test:", ["t-Test", "ANOVA", "Chi-square test"])
    
    if test_type == "t-Test":
        st.markdown("""
        **t-Test:**
        The t-test compares the means of two groups to determine whether they are significantly different.
        A low p-value indicates that the difference between the groups cannot be explained by chance.
        """)
        t_test(df, numerical_columns, categorical_columns)
        
    elif test_type == "ANOVA":
        st.markdown("""
        **ANOVA (Analysis of Variance):**
        ANOVA tests whether there are significant differences in the means of several groups.
        If the p-value is small, this indicates differences between the groups.
        """)
        anova_test(df, numerical_columns, categorical_columns)
        
    elif test_type == "Chi-square test":
        st.markdown("""
        **Chi-square test:**
        This test checks whether there is a statistically significant association between two categorical variables.
        """)
        chi_square_test(df, categorical_columns)


# Function for the t-Test
def t_test(df, numerical_columns, categorical_columns):
    group_col = st.selectbox("Choose the group variable:", categorical_columns)
    value_col = st.selectbox("Choose the value variable:", numerical_columns)
    
    group1 = st.text_input("Name of group 1:")
    group2 = st.text_input("Name of group 2:")

    apply_log_scale = st.checkbox("Apply logarithmic scaling?", value=False)

    if st.button("Perform t-Test"):
        # Count original data points
        group1_data = df[df[group_col] == group1][value_col]
        group2_data = df[df[group_col] == group2][value_col]

        initial_count_group1 = len(group1_data)
        initial_count_group2 = len(group2_data)

        # Remove NaN values
        group1_data = group1_data.dropna()
        group2_data = group2_data.dropna()

        remaining_count_group1 = len(group1_data)
        remaining_count_group2 = len(group2_data)

        # Show results
        st.write(f"**Group 1 ({group1}):** Total number of data points: {initial_count_group1}, without NaN: {remaining_count_group1}")
        st.write(f"**Group 2 ({group2}):** Total number of data points: {initial_count_group2}, without NaN: {remaining_count_group2}")

        # Apply logarithmic scaling if activated
        if apply_log_scale:
            group1_data = np.log1p(group1_data)
            group2_data = np.log1p(group2_data)

        if not group1_data.empty and not group2_data.empty:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            st.markdown(f"**t-Statistic:** {t_stat}")
            st.markdown(f"**p-Value:** {p_value}")
            
            # Create boxplot
            filtered_df = df[df[group_col].isin([group1, group2])]
            fig, ax = plt.subplots()
            sns.boxplot(x=filtered_df[group_col], y=filtered_df[value_col], ax=ax, palette="Set2")
            ax.set_title(f"Boxplot for {group1} vs. {group2}")

            # Logarithmic scaling in the plot if activated
            if apply_log_scale:
                ax.set_yscale('log')

            st.pyplot(fig)

            # Analysis of outliers
            outliers_group1 = group1_data[group1_data > group1_data.quantile(0.75) + 1.5 * (group1_data.quantile(0.75) - group1_data.quantile(0.25))]
            outliers_group2 = group2_data[group2_data > group2_data.quantile(0.75) + 1.5 * (group2_data.quantile(0.75) - group2_data.quantile(0.25))]

            st.write("**Outlier Analysis:**")
            if not outliers_group1.empty:
                st.write(f"In group 1 ({group1}) there are {len(outliers_group1)} outliers.")
            else:
                st.write(f"In group 1 ({group1}) there are no significant outliers.")

            if not outliers_group2.empty:
                st.write(f"In group 2 ({group2}) there are {len(outliers_group2)} outliers.")
            else:
                st.write(f"In group 2 ({group2}) there are no significant outliers.")

            # LLM interpretation
            context = (
                f"Please interpret the results of the t-Test as a standalone statistical report. "
                f"Analyze the statistical significance of the difference between the groups '{group1}' and '{group2}' for the variable '{value_col}', "
                f"with the following results:\n"
                f"- t-Statistic: {t_stat:.2f}\n"
                f"- p-Value: {p_value:.4f}\n\n"
                "Present your analysis without using section headings or numbered steps, focusing solely on statistical interpretation."
            )

            response = query_llm_via_cli(context)
            st.write(f"**t-Test Interpretation:** {response}")


# Function for the ANOVA test
def anova_test(df, numerical_columns, categorical_columns):
    group_col = st.selectbox("Choose the group variable:", categorical_columns)
    value_col = st.selectbox("Choose the value variable:", numerical_columns)

    if st.button("Perform ANOVA"):
        df_clean = df[[group_col, value_col]].dropna()

        # Determine group sizes
        group_sizes = df_clean.groupby(group_col).size()
        valid_groups = group_sizes[group_sizes >= 2].index
        df_filtered = df_clean[df_clean[group_col].isin(valid_groups)]

        if len(valid_groups) < 2:
            st.error("After removing small groups, there are not enough groups left for the ANOVA test.")
        else:
            grouped_data = [group[value_col].values for name, group in df_filtered.groupby(group_col)]
            try:
                anova_result = stats.f_oneway(*grouped_data)
                st.markdown(f"**F-Value:** {anova_result.statistic}")
                st.markdown(f"**p-Value:** {anova_result.pvalue}")

                # Boxplot to visualize the results
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=group_col, y=value_col, data=df_filtered, ax=ax)
                plt.xticks(rotation=90)
                st.pyplot(fig)

                # Tukey's HSD test for significant ANOVA results
                if anova_result.pvalue < 0.05:
                    st.write("The ANOVA test is significant. Tukey's HSD test will be performed.")
                    try:
                        tukey = pairwise_tukeyhsd(endog=df_filtered[value_col], groups=df_filtered[group_col], alpha=0.05)
                        st.pyplot(tukey.plot_simultaneous())

                        # Show Tukey HSD results
                        tukey_results_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                        st.write("Results of the Tukey HSD test:")
                        st.dataframe(tukey_results_df, height=400)

                        # Download option for results
                        csv = tukey_results_df.to_csv(index=False)
                        st.download_button(label="Download Tukey HSD results as CSV", data=csv, file_name='tukey_hsd_results.csv', mime='text/csv')

                    except Exception as e:
                        st.error(f"An error occurred during Tukey's HSD test: {str(e)}")

                # LLM interpretation
                context = (
                    f"Performing an ANOVA for '{value_col}' across groups in '{group_col}':\n"
                    f"- F-Value: {anova_result.statistic:.2f}\n"
                    f"- p-Value: {anova_result.pvalue:.4f}\n"
                    "Interpret these results, concentrating on whether the ANOVA suggests significant differences among the groups. "
                    "Provide an objective analysis without conversational language, focusing on the implications of the findings."
                )

                response = query_llm_via_cli(context)
                st.write(f"**ANOVA Interpretation:** {response}")

            except ValueError as e:
                st.error(f"An error occurred: {str(e)}.")


# Function for the Chi-square test
def chi_square_test(df, categorical_columns):
    cat_var1 = st.selectbox("Choose the first group variable:", categorical_columns)
    cat_var2 = st.selectbox("Choose the second group variable:", categorical_columns)
    
    if st.button("Perform Chi-square test"):
        # Remove NaN values from the selected variables
        df_clean = df[[cat_var1, cat_var2]].dropna()

        # Filter the top 10 categories for both variables
        top_cat_var1 = df_clean[cat_var1].value_counts().nlargest(10).index
        top_cat_var2 = df_clean[cat_var2].value_counts().nlargest(10).index
        df_filtered = df_clean[df_clean[cat_var1].isin(top_cat_var1) & df_clean[cat_var2].isin(top_cat_var2)]

        # Create the contingency table
        try:
            contingency_table = pd.crosstab(df_filtered[cat_var1], df_filtered[cat_var2])

            # Check if the contingency table is valid
            if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                st.error("The contingency table is invalid. Check the variables.")
            else:
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                st.markdown(f"**Chi-square statistic:** {chi2}")
                st.markdown(f"**p-Value:** {p}")
                st.markdown(f"**Degrees of freedom:** {dof}")
                
                # Visualize the contingency table as a heatmap
                st.write("**Heatmap of the contingency table:**")
                fig, ax = plt.subplots(figsize=(12, 10))  # Larger display
                sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", ax=ax)
                ax.set_title(f"Heatmap of the contingency table: {cat_var1} vs. {cat_var2} top 10")
                plt.xticks(rotation=90)
                st.pyplot(fig)

                # LLM interpretation
                context = (
                    f"Chi-square test for association between '{cat_var1}' and '{cat_var2}':\n"
                    f"- Chi-square statistic: {chi2:.2f}\n"
                    f"- p-Value: {p:.4f}\n"
                    f"- Degrees of freedom: {dof}\n\n"
                    "Interpret these results in a concise manner, focusing on whether the p-value indicates a significant association between the variables. "
                    "Avoid conversational or exploratory language; describe only the statistical significance of the relationship."
                )

                response = query_llm_via_cli(context)
                st.write(f"**Chi-square Test Interpretation:** {response}")

        except ValueError as e:
            st.error(f"An error occurred: {str(e)}.")


# Function for regression analysis
def regression_analysis(df, numerical_columns):
    reg_type = st.selectbox("Choose the type of regression:", ["Linear regression", "Logistic regression", "Multivariate regression"])

    # Linear regression
    if reg_type == "Linear regression":
        st.markdown("""
        **Linear regression:**
        This method models the relationship between a dependent variable and one or more independent variables.
        It is used to predict continuous values.
        """)
        linear_regression(df, numerical_columns)

    # Logistic regression
    elif reg_type == "Logistic regression":
        st.markdown("""
        **Logistic regression:**
        This method is used to model binary outcomes (e.g., Yes/No). It is particularly useful for classification problems.
        """)
        logistic_regression(df, numerical_columns)

    # Multivariate regression
    elif reg_type == "Multivariate regression":
        st.markdown("""
        **Multivariate regression:**
        Multivariate regression analyzes multiple dependent variables simultaneously to model their relationship with the independent variables.
        """)
        multivariate_regression(df, numerical_columns)


# Function for linear regression with separate models
def linear_regression(df, numerical_columns):
    st.write("**Correlation matrix of numerical variables:**")
    corr_matrix = df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    dependent_var = st.selectbox("Choose the dependent variable:", numerical_columns)
    independent_vars = st.multiselect("Choose the independent variables:", numerical_columns)

    if independent_vars:
        if st.button("Perform regression"):
            X = df[independent_vars].dropna()
            y = df[dependent_var].loc[X.index]
            y = y.dropna()
            X = X.loc[y.index]

            if y.isnull().values.any():
                st.error("The dependent variable still contains missing values. Please clean the data.")
            else:
                # Full model with all independent variables
                model = LinearRegression()
                model.fit(X, y)

                st.markdown("**Regression coefficients:**")
                for var, coef in zip(independent_vars, model.coef_):
                    st.write(f"- {var}: {coef}")
                st.write(f"**Intercept:** {model.intercept_}")

                # Separate models and scatterplots for each variable
                for var in independent_vars:
                    X_single_var = X[[var]]  # Use only the current independent variable
                    model_single = LinearRegression()
                    model_single.fit(X_single_var, y)

                    fig, ax = plt.subplots()
                    ax.scatter(X[var], y, edgecolor='none', facecolors='blue', s=5, label='Data points')

                    # Regression line based on the single model
                    ax.plot(X[var], model_single.predict(X_single_var), color='red', label='Regression line')
                    ax.set_xlabel(var)
                    ax.set_ylabel(dependent_var)
                    ax.set_title(f"{dependent_var} vs {var}")
                    ax.legend()
                    st.pyplot(fig)

                # LLM interpretation for linear regression
                context = (
                    f"The linear regression analysis was performed with the dependent variable '{dependent_var}' "
                    f"and independent variables {', '.join(independent_vars)}.\n"
                    f"Regression coefficients:\n" +
                    "\n".join([f"- {var}: {coef:.2f}" for var, coef in zip(independent_vars, model.coef_)]) +
                    f"\nIntercept: {model.intercept_:.2f}\n\n"
                    "Provide a straightforward interpretation of the relationship between the dependent and independent variables based on these coefficients. "
                    "Focus on any notable trends without using conversational or exploratory language."
                )

                response = query_llm_via_cli(context)
                st.write(f"**Linear Regression Interpretation:** {response}")


# Function for logistic regression
def logistic_regression(df, numerical_columns):
    dependent_var = st.selectbox("Choose the dependent variable (binary):", numerical_columns)
    independent_vars = st.multiselect("Choose the independent variables:", numerical_columns)

    if independent_vars:
        if st.button("Perform logistic regression"):
            X = df[independent_vars].dropna()
            y = df[dependent_var].loc[X.index].dropna()
            X = X.loc[y.index]

            # Check if the dependent variable is binary
            unique_values = y.unique()
            if len(unique_values) != 2:
                st.error("The dependent variable must be binary (e.g., 0 and 1).")
            else:
                model = LogisticRegression()
                model.fit(X, y)

                st.write("**Logistic regression coefficients:**")
                for var, coef in zip(independent_vars, model.coef_[0]):
                    st.write(f"- {var}: {coef}")
                st.write(f"**Intercept:** {model.intercept_[0]}")

                # Visualization of the logistic function for each independent variable
                for var in independent_vars:
                    fig, ax = plt.subplots()

                    # Scatterplot of actual data points
                    ax.scatter(X[var], y, label='Data points')

                    # Create range of values for the current variable var
                    x_range = np.linspace(X[var].min(), X[var].max(), 300).reshape(-1, 1)

                    # Calculate means for all other variables and use them
                    X_copy = pd.DataFrame(np.tile(X.mean().values, (300, 1)), columns=X.columns)
                    X_copy[var] = x_range.flatten()  # Vary the current variable var

                    # Create predictions based on all variables
                    y_prob = model.predict_proba(X_copy)[:, 1]

                    # Plot logistic function (probabilities)
                    ax.plot(x_range, y_prob, color='red', label='Logistic function')
                    ax.set_xlabel(var)
                    ax.set_ylabel(f'Probability ({dependent_var})')
                    ax.set_title(f'Logistic regression: {dependent_var} vs {var}')
                    ax.legend()
                    st.pyplot(fig)

                # LLM interpretation for logistic regression
                context = (
                    f"The logistic regression analysis was performed with the binary dependent variable '{dependent_var}' "
                    f"and independent variables {', '.join(independent_vars)}.\n"
                    f"Logistic regression coefficients:\n" +
                    "\n".join([f"- {var}: {coef:.2f}" for var, coef in zip(independent_vars, model.coef_[0])]) +
                    f"\nIntercept: {model.intercept_[0]:.2f}\n\n"
                    "Interpret the impact of the independent variables on the likelihood of the dependent outcome in a statistical manner. "
                    "Avoid conversational language; describe only significant predictors and their effects on the outcome."
                )

                response = query_llm_via_cli(context)
                st.write(f"**Logistic Regression Interpretation:** {response}")

# Function for multivariate regression
def multivariate_regression(df, numerical_columns):
    dependent_vars = st.multiselect("**Choose the dependent variables (multiple):**", numerical_columns)
    independent_vars = st.multiselect("**Choose the independent variables:**", numerical_columns)

    if dependent_vars and independent_vars:
        if st.button("Perform multivariate regression"):
            X = df[independent_vars].dropna()
            Y = df[dependent_vars].loc[X.index].dropna()
            X = X.loc[Y.index]

            if X.shape[1] != len(independent_vars) or Y.shape[1] != len(dependent_vars):
                st.error("The number of independent or dependent variables does not match.")
                return

            model = LinearRegression()
            model.fit(X, Y)

            st.write("**Multivariate regression coefficients:**")
            for i, dep_var in enumerate(dependent_vars):
                st.write(f"\nFor the dependent variable: **{dep_var}**")
                st.write(f"Intercept: {model.intercept_[i]}")
                for var, coef in zip(independent_vars, model.coef_[i]):
                    st.write(f"- {var}: {coef}")

            # Visualization of the regression results
            for dep_var in dependent_vars:
                for var in independent_vars:
                    fig, ax = plt.subplots()
                    ax.scatter(X[var], Y[dep_var], label='Data points')

                    # Ensure that the prediction is made with all variables for the scatterplot
                    x_range = np.linspace(X[var].min(), X[var].max(), 300).reshape(-1, 1)

                    # Create a copy of X with the mean value for all variables except the current variable
                    X_copy = pd.DataFrame(np.tile(X.mean().values, (300, 1)), columns=X.columns)
                    X_copy[var] = x_range.flatten()

                    y_pred = model.predict(X_copy)

                    ax.plot(x_range, y_pred[:, dependent_vars.index(dep_var)], color='red', label='Regression line')
                    ax.set_xlabel(var)
                    ax.set_ylabel(dep_var)
                    ax.set_title(f'Multivariate regression: {dep_var} vs {var}')
                    ax.legend()
                    st.plotly_chart(fig)

            # LLM interpretation for multivariate regression
            context = (
                f"The multivariate regression analysis was conducted with dependent variables {', '.join(dependent_vars)} "
                f"and independent variables {', '.join(independent_vars)}.\n"
                "The regression coefficients for each dependent variable are as follows:\n" +
                "\n".join(
                    [f"For {dep_var}:\n" + "\n".join([f"- {var}: {coef:.2f}" for var, coef in zip(independent_vars, model.coef_[i])]) 
                    for i, dep_var in enumerate(dependent_vars)]
                ) +
                "\n\nProvide an objective interpretation of the relationship between the dependent and independent variables. "
                "Focus solely on the statistical influence of each independent variable on the dependent outcomes without conversational or exploratory language."
            )

            response = query_llm_via_cli(context)
            st.write(f"**Multivariate Regression Interpretation:** {response}")


# Function for time series analysis with LLM interpretation
def perform_time_series_analysis(df, time_var, value_var):
    # Convert the time variable to date format and remove invalid data
    df[time_var] = pd.to_datetime(df[time_var], errors='coerce')
    df = df.dropna(subset=[time_var])

    if df.empty:
        st.error("**Error:** The time variable has an incorrect format.")
    else:
        # Create a new column 'year' to allow yearly grouping
        df['year'] = df[time_var].dt.year
        yearly_avg = df.groupby('year')[value_var].mean().reset_index()

        # Determine the minimum and maximum values
        y_min = yearly_avg[value_var].min()
        y_max = yearly_avg[value_var].max()
        y_range = y_max - y_min
        y_buffer = y_range * 0.05

        # Calculate the overall average
        overall_avg = df[value_var].mean()

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(yearly_avg['year'], yearly_avg[value_var], marker='o', label='Yearly average')
        ax.axhline(overall_avg, color='red', linestyle='--', label=f'Overall average: {overall_avg:.2f}')
        ax.set_title(f'Average {value_var} per year')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Average {value_var}')
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

        # Show the overall average as text in the plot
        ax.text(yearly_avg['year'].max() - (yearly_avg['year'].max() - yearly_avg['year'].min()) * 0.05, 
                overall_avg + y_buffer, 
                f'{overall_avg:.2f}', color='red', ha='right', va='center')

        # Display yearly average values
        for i in range(len(yearly_avg)):
            if i % 2 == 0:
                ax.text(yearly_avg['year'][i], yearly_avg[value_var][i] + y_buffer/2, 
                        f'{yearly_avg[value_var][i]:.2f}', color='blue', ha='center', va='bottom')
            else:
                ax.text(yearly_avg['year'][i], yearly_avg[value_var][i] - y_buffer/2, 
                        f'{yearly_avg[value_var][i]:.2f}', color='blue', ha='center', va='top')

        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)

        # Output calculated statistics
        st.write(f"**Standard deviation:** {df[value_var].std():.2f}")
        st.write(f"**Variance:** {df[value_var].var():.2f}")
        st.write(f"**Minimum {value_var}:** {y_min:.2f} in year {yearly_avg.loc[yearly_avg[value_var].idxmin(), 'year']}")
        st.write(f"**Maximum {value_var}:** {y_max:.2f} in year {yearly_avg.loc[yearly_avg[value_var].idxmax(), 'year']}")

        
        context = (
            f"In this time series analysis of '{value_var}' over the years, the following key statistics were observed:\n"
            f"- Overall average: {overall_avg:.2f}\n"
            f"- Minimum value: {y_min:.2f} in year {yearly_avg.loc[yearly_avg[value_var].idxmin(), 'year']}\n"
            f"- Maximum value: {y_max:.2f} in year {yearly_avg.loc[yearly_avg[value_var].idxmax(), 'year']}\n"
            f"- Standard deviation: {df[value_var].std():.2f}\n"
            "Provide a clear interpretation of the observed trends and patterns over time. "
            "Avoid conversational language; focus on identifying any notable increases, decreases, or stability in the data."
        )

        # Send request to the LLM for analysis
        response = query_llm_via_cli(context)
        st.write(f"**Time Series Interpretation:** {response}")


# Function for clustering methods
def clustering_methods(df, numerical_columns):
    cluster_method = st.selectbox("Choose the clustering method:", ["k-Means", "Hierarchical Clustering", "DBSCAN"])
    
    # Show explanation based on the selected method
    if cluster_method == "k-Means":
        st.markdown("""
        **k-Means Clustering:**
        This method groups data points into k clusters by minimizing the distance between the points. 
        It is often used to find similarities in the data.
        """)
    elif cluster_method == "Hierarchical Clustering":
        st.markdown("""
        **Hierarchical Clustering:**
        Hierarchical clustering creates a hierarchy of clusters that can be represented by a dendrogram structure. 
        This allows understanding the structure of the data at different levels.
        """)
    elif cluster_method == "DBSCAN":
        st.markdown("""
        **DBSCAN (Density-Based Clustering):**
        DBSCAN groups points based on the density of their surroundings. It is particularly useful for identifying clusters of arbitrary shape 
        and can detect outliers (noise).
        """)

    # Selection of variables for clustering
    selected_vars = st.multiselect("Choose the variables for clustering:", numerical_columns)

    if selected_vars:
        # Parameters for k-Means or hierarchical clustering
        if cluster_method == "k-Means" or cluster_method == "Hierarchical Clustering":
            n_clusters = st.slider("**Number of clusters**", 2, 10, 3)
        # Parameters for DBSCAN
        elif cluster_method == "DBSCAN":
            eps = st.slider("Choose the radius (eps):", 0.1, 5.0, 0.5)
            min_samples = st.slider("Choose the minimum number of points per cluster (min_samples):", 1, 10, 5)

        # Button to start the clustering process
        if st.button("Perform clustering"):
            X = df[selected_vars].dropna()  # Remove rows with missing values
            if cluster_method == "k-Means":
                perform_kmeans(X, n_clusters)
            elif cluster_method == "Hierarchical Clustering":
                perform_hierarchical_clustering(X, n_clusters)
            elif cluster_method == "DBSCAN":
                perform_dbscan(X, eps, min_samples)
    else:
        st.error("Please select at least two variables!")


# Function for k-Means Clustering
def perform_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    X['Cluster'] = kmeans.fit_predict(X)
    
    visualize_clusters(X, 'k-Means Clustering')


# Function for hierarchical clustering
def perform_hierarchical_clustering(X, n_clusters):
    hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    X['Cluster'] = hierarchical_clustering.fit_predict(X)
       
    visualize_clusters(X, 'Hierarchical Clustering')


# Function for DBSCAN
def perform_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    X['Cluster'] = dbscan.fit_predict(X)
    
    visualize_clusters(X, 'DBSCAN Clustering')


# Function to visualize cluster statistics dynamically based on detected numerical variables
def display_cluster_statistics(cluster_stats, numerical_columns):
    for cluster, stats in cluster_stats.items():
        st.write(f"**Cluster {cluster}:**")
        
        # Erstelle das DataFrame dynamisch basierend auf den numerischen Spalten
        stats_data = {
            'Statistische Größe': ['Durchschnittswerte', 'Standardabweichung', 'IQR']
        }
        
        for column in numerical_columns:
            stats_data[column] = [
                stats['mean'].get(column, np.nan),
                stats['std_dev'].get(column, np.nan),
                stats['iqr'].get(column, np.nan)
            ]
        
        stats_df = pd.DataFrame(stats_data)
        
        st.table(stats_df)
        st.write(f"Größe: {stats['size']} Punkte")


def visualize_clusters(X, title):
    # Ensure that we have enough data points and variables
    num_samples, num_features = X.shape
    n_components = min(num_samples, num_features, 2)  # max 2 components, but fewer if not enough data

    if n_components < 2:
        st.error("Not enough data points or variables to perform PCA.")
        return

    pca = PCA(n_components=n_components)
    
    try:
        X_pca = pca.fit_transform(X.drop(columns=['Cluster']))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X['Cluster'], cmap='tab10', s=25, alpha=0.4)
        
        ax.set_title(title)
        ax.set_xlabel(f'PCA 1' if n_components >= 1 else '')
        ax.set_ylabel(f'PCA 2' if n_components == 2 else '')

        cluster_counts = X['Cluster'].value_counts()  
        legend_labels = [f"Cluster {int(cluster)} ({count} points)" for cluster, count in cluster_counts.items()]
        legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
        ax.add_artist(legend1)
        st.pyplot(fig)

        # Show the average values of the variables for each cluster
        st.write(f"**Average values per cluster:**")
        cluster_means = X.groupby('Cluster').mean()
        st.dataframe(cluster_means)

        # Generate a more readable summary context for the LLM
        cluster_summary = "\n".join(
            [
                f"Cluster {int(cluster)}: "
                + ", ".join([f"{var}: {value:.2f}" for var, value in means.items()])
                for cluster, means in cluster_means.iterrows()
            ]
        )

        context = (
            f"The following clusters were identified based on the selected variables:\n{cluster_summary}\n\n"
            "Provide a detailed statistical interpretation of these clusters, focusing strictly on patterns, "
            "variability, and insights into data distribution. Avoid conversational, exploratory, or speculative language."
        )

        # Send the context to the LLM for interpretation
        response = query_llm_via_cli(context)
        st.write(f"**Cluster Analysis Interpretation:** {response}")

    except ValueError as e:
        st.error(f"**Error:** Not enough variables selected.")


# Function to communicate with the LLM
def query_llm_via_cli(input_text):
    """Sends the question and context to the LLM and receives a response"""
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3.1p"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )
        stdout, stderr = process.communicate(input=f"{input_text}\n", timeout=40)

        if process.returncode != 0:
            return f"Error in the model request: {stderr.strip()}"

        response = re.sub(r'\x1b\[.*?m', '', stdout)
        return extract_relevant_answer(response)

    except subprocess.TimeoutExpired:
        process.kill()
        return "Timeout for the model request"
    except Exception as e:
        return f"An unexpected error has occurred: {str(e)}"

def extract_relevant_answer(full_response):
    response_lines = full_response.splitlines()
    if response_lines:
        return "\n".join(response_lines).strip()
    return "No answer received"


# Main function to start the app
def main():
    st.title("ASCVIT V1.5")

    # Sidebar for file upload
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("**Upload your data file**", type=["csv", "xlsx"])

    # Check if a file has been uploaded
    if uploaded_file:
        # Read file (either CSV or Excel)
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Reset numeric and categorical columns based on the new record
        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(include='object').columns.tolist()

        # Update the session state variables only when a new file has been uploaded
        if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
            st.session_state['numerical_columns'] = numerical_columns
            st.session_state['categorical_columns'] = categorical_columns
            st.session_state['last_uploaded_file'] = uploaded_file.name  # Speichere den Dateinamen

        # Show data only when the button is clicked
        if st.sidebar.button("Data Overview"):
            st.session_state['show_data'] = True
            st.session_state['show_analysis'] = None  # Disable analysis display

        # Analysis methods as buttons in the sidebar
        if st.sidebar.button("Descriptive Statistics"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Descriptive Statistics'

        if st.sidebar.button("Hypothesis Tests"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Hypothesis Tests'

        if st.sidebar.button("Regression Analysis"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Regression Analysis'

        if st.sidebar.button("Time Series Analysis"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Time Series Analysis'

        if st.sidebar.button("Clustering Methods"):
            st.session_state['show_data'] = False
            st.session_state['show_analysis'] = 'Clustering Methods'

        # Show data or analysis depending on the selection
        if 'show_data' in st.session_state and st.session_state['show_data']:
            st.header("Data Overview")
            st.write("**Data Preview:**")
            st.dataframe(df.head())
            display_data_info(df)

        if 'show_analysis' in st.session_state and st.session_state['show_analysis']:
            if st.session_state['show_analysis'] == 'Descriptive Statistics':
                st.header("Descriptive Statistics")
                descriptive_statistics(df, numerical_columns)

            if st.session_state['show_analysis'] == 'Hypothesis Tests':
                st.header("Hypothesis Tests")
                hypothesis_testing(df, numerical_columns, categorical_columns)

            if st.session_state['show_analysis'] == 'Regression Analysis':
                st.header("Regression Analysis")
                regression_analysis(df, numerical_columns)

            if st.session_state['show_analysis'] == 'Time Series Analysis':
                st.header("Time Series Analysis")
                time_var = st.selectbox("Choose the time variable:", df.columns)
                value_var = st.selectbox("Choose the value variable:", numerical_columns)
                
                # Explanation is shown immediately
                st.markdown("""
                **Time Series Analysis:**
                This method examines how a variable changes over time. 
                It helps to identify trends, seasonal patterns, and other time-dependent phenomena.
                """)

                if st.button("Perform time series analysis"):
                    if time_var and value_var:
                        perform_time_series_analysis(df, time_var, value_var)
                    else:
                        st.error("Please choose both a time variable and a value variable.")

            if st.session_state['show_analysis'] == 'Clustering Methods':
                st.header("Clustering Methods")
                clustering_methods(df, numerical_columns)

    else:
        st.sidebar.info("Please upload a data file to get started.")

if __name__ == "__main__":
    main()

