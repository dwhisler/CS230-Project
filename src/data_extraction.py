import pandas as pd
import numpy as np
import tensorflow as tf


def sampling_k_elements(group, k):
    if len(group) < k:
        return group
    return group.sample(k)

def get_loan_data(filename, cols_filename, n, equalize=True):
    if n == 'all':
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, nrows=n)

    df = df.loc[df['loan_status'].isin(['Default', 'Fully Paid', 'Charged Off'])] #only use datapoints with relevant labels
    label_map = {'Default': 1, 'Charged Off': 1,'Fully Paid':0}
    df = df.applymap(lambda s: label_map.get(s) if s in label_map else s)

    if equalize:
        df = df.groupby('loan_status').apply(sampling_k_elements, min(df['loan_status'].value_counts())).reset_index(drop=True) # balance dataset in labels

    with open(cols_filename, 'r') as f:
        cols = f.readlines()
        cols = [col.strip() for col in cols]
    df = df[cols]

    emp_length_map = {'< 1 year': 0, '1 year': 1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9, '10 years':10,'10+ years':11}
    df = df.applymap(lambda s: emp_length_map.get(s) if s in emp_length_map else s)

    df = df.fillna(0)

    with open("cols_onehot.txt", 'r') as f:
        cols_onehot = f.readlines()
        cols_onehot = [col.strip() for col in cols_onehot]


    labels = df.pop('loan_status')
    one_hot = pd.get_dummies(df[cols_onehot])
    df = df.drop(columns=cols_onehot)
    df = (df - df.mean())/df.std() # normalize numeric columns before joining one hots
    df = df.join(one_hot)

    return df, labels

if __name__ == '__main__':
    filename = "../data/accepted_2007_to_2018Q4.csv"
    cols_filename = "cols_all_test.txt"
    X, y = get_loan_data(filename, cols_filename, 1000000)

    for x in X.loc[0,:]:
        print(x)
