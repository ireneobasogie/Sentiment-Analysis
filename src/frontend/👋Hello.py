import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome 👋")

st.sidebar.success("Select one function above.")

st.markdown(
    """
    Sentiment analysis, also referred to as opinion mining, is an approach 
    to natural language processing (NLP) that identifies the emotional tone 
    behind a body of text. This is a popular way for organizations to determine 
    and categorize opinions about a product, service, or idea.
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a function from the sidebar** to see some examples
    !
    ### There are two functions in our project?
    - **Sentiment Analysis**: where you can enter a movie review, 
    and you'll get a sentiment among 'happy', 'not-relevant', 'angry', 'disgust', 'sad', 'surprise'
    - **Data Visualization**: where you can load the movie review dataset that we used for training our model
    and see the distribution graph of all sentiment labels
    """
)