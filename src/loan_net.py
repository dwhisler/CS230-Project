from data_extraction import get_loan_data
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# HYPERPARAMETERS
m = 150000
train_test_split = [95, 5]
layer_sizes = [32, 10]
batch_size = 32
epochs = 5
optimizer = keras.optimizers.Adam()
metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

def calculate_f1(precision, recall):
    return 2/(1/precision + 1/recall)

def split_train_test(ds):
    

def run_loan_net():
    filename = "../data/accepted_2007_to_2018Q4.csv"
    cols_filename = "cols_all_test.txt"
    df, target = get_loan_data(filename, cols_filename, m, False)
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    train_dataset = dataset.shuffle(len(df)).batch(batch_size)
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

    model = keras.Sequential()
    for i in range(len(layer_sizes)):
        model.add(layers.Dense(layer_sizes[i], activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(train_dataset, epochs=epochs)

    print('F1 Score: ', calculate_f1(metrics[0].result().numpy(), metrics[1].result().numpy()))


if __name__ == '__main__':
    run_loan_net()
