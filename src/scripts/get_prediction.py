import torch
import streamlit as st
from transformers import BertForSequenceClassification
from .model_training import evaluate
from .performance import f1_score_func, accuracy_per_class

device = "fff"
label_dict = "ddd"
# Task 10: Loading and Evaluating our Model
def load_and_evaluate_our_model(dataloader_val):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = len(label_dict),
        output_attentions=False,# We don't want any unnecessary input from the model
        output_hidden_states=False # State just before the prediction, that might be useful for encoding 
    )

    model.to(device)

    model.load_state_dict(
        torch.load('./Bert_ft_epoch10.model',
                map_location=torch.device('cuda')))

    _, predictions, true_vals = evaluate(dataloader_val)


    return accuracy_per_class(predictions, true_vals)


class Get_prediction:
    def __init__(self, review_text: str, tokenizer, device):
        self.review_text = review_text
        self.tokenizer = tokenizer
        self.device = device

    def get_tokenized_sentence(self):
        # review_text = "# optimizer is for setting the learning rate, Adam Learning rate is a way to optimize our learning rate"
        tokens = self.tokenizer.tokenize(self.review_text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)


        st.write(f' Sentence to be analyzed: {self.review_text}')
        st.write(f'   Tokens of the sentence: {tokens}')
        st.write(f'Token IDs of each token: {token_ids}')

    def get_label(self, model):
        encoded_review = self.tokenizer.encode_plus(
            self.review_text,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)
        class_names = ["happy","not-relevant","angry","surprise","sad", "disgust"]

        output = model(input_ids, attention_mask)

        _, prediction = torch.max(output[0], dim=1)
        # print(f'Review text: {self.review_text}')
        # print(f'Sentiment  : {class_names[prediction]}')

        st.write(f'Review text: {self.review_text}')
        st.write(f'Sentiment  : {class_names[prediction]}')