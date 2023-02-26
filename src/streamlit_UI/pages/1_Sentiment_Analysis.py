import sys
sys.path.append(".")
import streamlit as st
import os
print(os.getcwd())
# from scripts.sentiment_analysis import Get_prediction, BERT_Pretrained_Model, tokenizer
from src.scripts.get_prediction import Get_prediction
from src.scripts.model_training import BERT_Pretrained_Model, tokenizer
import torch

st.title("💯Sentiment analysis for movie reviews based on Bert")
review_text = st.text_input("Please enter a movie review that you want to analyze: ")
# review_text = "Sentiment analysis for movie reviews based on Bert, Please enter a movie review that you want to analyze"
if review_text is not None:
    tokenizer = tokenizer()
    label_dict = {'happy': 0, 'not-relevant': 1, 'angry': 2, 'disgust': 3, 'sad': 4, 'surprise': 5}
    model = BERT_Pretrained_Model(label_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    get_prediction = Get_prediction(review_text,tokenizer,device)
    get_prediction.get_tokenized_sentence()
    get_prediction.get_label(model)


def get_performance_of_trained_model():
    pass


