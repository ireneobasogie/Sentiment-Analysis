import sys
sys.path.append(".")
import streamlit as st
from sentiment_analysis.sentiment_analysis import Get_prediction, BERT_Pretrained_Model, tokenizer
import torch

st.title("Sentiment analysis for movie reviews based on Bert")
review_text = st.text_input("Please enter a movie review that you want to analyze: ")

uploaded_file = st.file_uploader("")
if review_text is not None:
    tokenizer = tokenizer()
    label_dict = {'happy': 0, 'not-relevant': 1, 'angry': 2, 'disgust': 3, 'sad': 4, 'surprise': 5}
    model = BERT_Pretrained_Model(label_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    get_prediction = Get_prediction(review_text,tokenizer,device)
    get_prediction.get_tokenized_sentence()
    get_prediction.get_label()
