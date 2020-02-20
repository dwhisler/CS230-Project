from data_extraction import get_loan_data
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc

# HYPERPARAMETERS
m = 100000
split = [95, 5]
layer_sizes = [24, 10]
batch_size = 32
epochs = 10
optimizer = keras.optimizers.Adam()
metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

def calculate_f1(precision, recall):
    return 2/(1/precision + 1/recall)

def run_loan_net():
    filename = "../data/accepted_2007_to_2018Q4.csv"
    cols_filename = "cols_all_test.txt"
    df, target = get_loan_data(filename, cols_filename, m, False)
    train_df, test_df, train_target, test_target = train_test_split(df, target, test_size=split[1]/100, random_state=5)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df.values, train_target.values))
    train_dataset = train_dataset.shuffle(len(train_df)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_df.values, test_target.values))
    test_dataset = test_dataset.shuffle(len(test_df)).batch(batch_size)

    model = keras.Sequential()
    for i in range(len(layer_sizes)):
        model.add(layers.Dense(layer_sizes[i], activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(train_dataset, epochs=epochs)
    print('Training F1 Score: ', calculate_f1(metrics[1].result().numpy(), metrics[2].result().numpy()))

    test_results = model.evaluate(test_dataset)
    print('Test F1 Score: ', calculate_f1(test_results[2], test_results[3]))

    # Plot ROC

    plt.figure()
    y_pred = model.predict(test_df).ravel()
    fpr, tpr, t = roc_curve(test_target, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='AUC = %0.2f' % roc_auc)
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Deep Network')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    run_loan_net()
