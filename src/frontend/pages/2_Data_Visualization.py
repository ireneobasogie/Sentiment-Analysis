import sys
sys.path.append(".")
import streamlit as st
import pandas as pd
import plotly.express as px
from src.scripts.model_training import get_df

uploaded_file = st.file_uploader("Please update the file that you want to analyze")
st.caption("Attention, the acceptable formats are CSV, TSV and Excel")



    
def load_data():
    if uploaded_file is not None:
        filename = uploaded_file.name
        extension = filename.split(".")[1]
        if extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif extension == "tsv":
            df = pd.read_table(uploaded_file)
        elif extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        else:
            st.error("The format of your uploade file cannot be dealt with by this tool")
        # st.write(df)
    return filename, df

def data_visualization(uploaded_file):
    # df.columns = ['id', 'text', 'category']
    # # Whether to modify the DataFrame rather than creating a new one.
    # df.set_index('id', inplace=True)
    # st.write(df)
    # # Create a Plotly histogram of the "category" column
    # fig = px.histogram(df, x="category")

    # # Display the plot using Streamlit
    # st.write("This graph shows category before cleaning")
    # st.plotly_chart(fig)

    df, label_dict = get_df(filepath=uploaded_file)
    st.write(df)
    # Create a Plotly histogram of the "category" column
    fig = px.histogram(df, x="category")

    # Display the plot using Streamlit
    st.write("This graph shows category after cleaning")
    st.plotly_chart(fig)

    return df

if uploaded_file is not None:
    df = data_visualization(uploaded_file)
    st.write(df.category.value_counts())