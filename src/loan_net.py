from data_extraction import get_loan_data
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def build_model(layer_sizes, optimizer, loss, metrics):
    model = keras.Sequential()
    for i in range(len(layer_sizes)):
        model.add(layers.Dense(layer_sizes[i], activation='relu'))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #model.summary()
    return model

def f1(precision, recall):
    return 2/(1/precision + 1/recall)

def plot_roc(model, x_test, y_test):
    plt.figure()
    y_pred = model.predict(x_test).ravel()
    fpr, tpr, t = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='AUC = %0.2f' % roc_auc)
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Deep Network')
    plt.legend(loc="lower right")
    plt.show()

def train_loan_net(model, batch_size, epochs):

    ds_train, ds_val = load_train_val_data(batch_size)
    model.fit(ds_train, epochs=epochs, verbose=1, validation_data=ds_val)
    f1_score_train = f1(metrics[0].result().numpy(), metrics[1].result().numpy())
    print("Training F1 Score = ", f1_score_train)
    results = model.evaluate(ds_val)
    f1_score_val = f1(results[1], results[2])
    print("Validation F1 Score = ", f1_score_val)

def test_loan_net(model):
    ds_test = load_test_data(batch_size)
    results = model.evaluate(ds_test)
    f1_score_test = f1(results[1], results[2])
    print("Validation F1 Score = ", f1_score_test)

def load_train_val_data(batch_size):
    x_train = pd.read_pickle("../pickle/x_train_1M_balanced.pkl")
    y_train = pd.read_pickle("../pickle/y_train_1M_balanced.pkl")
    x_val = pd.read_pickle("../pickle/x_val_1M_balanced.pkl")
    y_val = pd.read_pickle("../pickle/y_val_1M_balanced.pkl")
    ds_train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
    ds_train = ds_train.shuffle(len(x_train)).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values))
    ds_val = ds_val.shuffle(len(x_val)).batch(batch_size)
    return ds_train, ds_val

def load_test_data(batch_size):
    x_test = pd.read_pickle("../pickle/x_test_1M_balanced.pkl")
    y_test = pd.read_pickle("../pickle/y_test_1M_balanced.pkl")
    ds_test = tf.data.Dataset.from_tensor_slices((x_test.values, y_test.values))
    ds_test = ds_test.shuffle(len(x_train)).batch(batch_size)
    return ds_test

def prepare_data(filename, cols_filename, m, split):
    x, y = get_loan_data(filename, cols_filename, m, True)
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=(split[1]+split[2])/100, random_state=5)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=split[2]/100, random_state=5)
    x_train.to_pickle("../pickle/x_train_1M_balanced.pkl")
    y_train.to_pickle("../pickle/y_train_1M_balanced.pkl")
    x_val.to_pickle("../pickle/x_val_1M_balanced.pkl")
    y_val.to_pickle("../pickle/y_val_1M_balanced.pkl")
    x_test.to_pickle("../pickle/x_test_1M_balanced.pkl")
    y_test.to_pickle("../pickle/y_test_1M_balanced.pkl")

def run_loan_net():
    # HYPERPARAMETERS
    filename = "../data/accepted_2007_to_2018Q4.csv"
    cols_filename = "cols_all_test.txt"
    m = 1000000
    split = [95, 2.5, 2.5]
    layer_sizes = [128, 128, 64, 16]
    batch_size = 32
    epochs = 10
    optimizer = keras.optimizers.SGD()
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    #loss = focal_loss
    metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    # prepare_data(filename, cols_filename, m, split)

    model = build_model(layer_sizes, optimizer, loss, metrics)

    train_loan_net(model, batch_size, epochs)




if __name__ == '__main__':
    run_loan_net()
