import streamlit as st
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
import re

from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool

st.set_page_config(page_title="Data Analytics Assistant", layout="wide")

st.title("📊 Data Analytics Assistant")
st.markdown("Upload your dataset and ask questions to gain insights.")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == ".xlsx":
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        df = None


if df is not None:
    st.header("Data Preview")
    st.dataframe(df.head())
    
    st.header("Basic Statistics")
    st.write(df.describe())
    
    st.header("Visualizations")
    with st.expander("Create a Plot"):
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_type = st.selectbox("Select Plot Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"])
        with col2:
            x_axis = st.selectbox("Select X-axis", df.columns)
        with col3:
            y_axis = st.selectbox("Select Y-axis (optional)", [None] + list(df.columns))
        
        if st.button("Generate Plot"):
            fig, ax = plt.subplots()
            try:
                if plot_type == "Bar Chart":
                    sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
                elif plot_type == "Line Chart":
                    sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
                elif plot_type == "Scatter Plot":
                    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
                elif plot_type == "Histogram":
                    sns.histplot(data=df, x=x_axis, ax=ax)
                elif plot_type == "Box Plot":
                    sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Plotting Error: {e}")
                

    st.header("SQL Query")
    with st.expander("Run a SQL Query"):
        sql_query = st.text_area("Enter your SQL query (table name is 'df')")
        if st.button("Run Query"):
            try:
                query_result = ps.sqldf(sql_query, locals())
                st.write(query_result)
            except Exception as e:
                st.error(f"SQL Error: {e}")
                
    st.header("Query the Data")
    user_query = st.text_input("Ask a question about your data:")
    
    if user_query:
        try:
            llm = AzureChatOpenAI(
                azure_deployment=st.secrets["AZURE_DEPLOYMENT"],
                azure_endpoint=st.secrets["AZURE_ENDPOINT"],
                openai_api_key=st.secrets["AZURE_API_KEY"],
                openai_api_version=st.secrets["AZURE_API_VERSION"],
                temperature=0
            )

            
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            
            with st.spinner("Analyzing..."):
                response = agent.run(user_query)
                st.success("Analysis Complete!")
                st.write(response)
                
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to get started.")








