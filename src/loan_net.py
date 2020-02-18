from data_extraction import get_loan_data
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def run_loan_net():
    filename = "../data/accepted_2007_to_2018Q4.csv"
    cols_filename = "cols_all_test.txt"
    df, target = get_loan_data(filename, cols_filename, 50000, False)
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    for feat, targ in dataset.take(5):
        print ('Features: {}, Target: {}'.format(feat, targ))

    train_dataset = dataset.shuffle(len(df)).batch(32)

    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

    layer_sizes = [128, 64, 64, 10]

    model = keras.Sequential()
    for i in range(len(layer_sizes)):
        model.add(layers.Dense(layer_sizes[i], activation='relu'))

    model.add(layers.Dense(1, activation='softmax'))

    optimizer = keras.optimizers.Adam()
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    metric = 'accuracy'


    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model.fit(train_dataset, epochs=5)


if __name__ == '__main__':
    run_loan_net()
