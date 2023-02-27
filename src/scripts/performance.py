import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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

# def get_stats(self, test: pd.DataFrame) -> float:
#     ''' make predictions on test set + get classification report & confusion matrix '''

#     print('Loading model...')
#     model = pickle.load(open(self.model_filename, 'rb'))

#     print('Making predictions...')
#     X_test,y_test = test.doc, test.parti
#     X_test = self.vectorizer.transform(X_test)
#     y_pred = model.predict(X_test)

#     labels = model.classes_

#     print('Saving Confusion Matrix...')
#     cm = confusion_matrix(y_test, y_pred, labels=labels)
#     plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     plot.plot()
#     plt.savefig('eval/confusion_matrix.png')

#     print('Classification report is being sent into eval/classification-report.txt...')
#     with open('eval/classification-report.txt', 'w') as f:
#         f.write(f'{classification_report(y_test, y_pred, target_names=labels)}')

#     return model.score(X_test,y_test)


# Load the dataset
df, label_dict = get_df(filepath=r"C:\\Users\\irene\\Uni\\M2\\Arbres, graphes et réseaux\\Sentiment-Analysis\\smile-annotations-final.csv")
print(df)




# Initialize the tokenizer
tokenizer = tokenizer()

# Encode the dataset
dataset_train, dataset_val = encode_data(df, tokenizer)

dataloader_train, dataloader_val = Data_loaders(dataset_train, dataset_val)

model = BERT_Pretrained_Model(label_dict)
model.to("cuda")
model.load_state_dict(
    torch.load(r"C:\\Users\\irene\\Uni\\M2\\Arbres, graphes et réseaux\\Sentiment_Analysis\\Bert_ft_epoch10.model", map_location=torch.device("cuda")))

# We want to know if our model is overtraining
val_loss , predictions, true_vals = evaluate(dataloader_val)
val_f1 = f1_score_func(predictions, true_vals)
print(val_f1)
# # # tqdm.write(f'Validation loss: {val_loss}')
# # # tqdm.write(f'F1 score (weighted): {val_f1}')


result = accuracy_per_class(predictions, true_vals, label_dict)
print(result)






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
