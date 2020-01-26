import numpy as np
import pandas as pd

def read_csv_data(filename):
    data = pd.read_csv(filename)
    print(data)

if __name__ == '__main__':
    filename = "loan.csv"
    read_csv_data()
