import numpy as np
from sklearn.metrics import f1_score
# Task 8: Defining our Performance Metrics

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from model_training import *

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class:{label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')

    return y_true, y_preds


df, label_dict = get_df(filepath="/Users/liupan/Desktop/Cours/M2_S2/réseau_de_neurones/project/Sentiment_Analysis/smile-annotations-final.csv")
tokenizer = tokenizer()
dataset_train, dataset_val = encode_data(df, tokenizer)

model = BERT_Pretrained_Model(label_dict)

dataloader_train, dataloader_val = Data_loaders(dataset_train, dataset_val)
# We want to know if our model is overtraining
val_loss , predictions, true_vals = evaluate(dataloader_val)
val_f1 = f1_score_func(predictions, true_vals)
# tqdm.write(f'Validation loss: {val_loss}')
# tqdm.write(f'F1 score (weighted): {val_f1}')
y_true, y_preds = accuracy_per_class(preds, labels, label_dict)
def confusion_matrix(y_true, y_pred):
    # Generate some sample data
    # y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    # y_pred = np.array([0, 1, 1, 0, 2, 1, 0, 1, 2])

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Define class labels
    class_names = ['Class 0', 'Class 1', 'Class 2']

    # Plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        title='Confusion matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()
