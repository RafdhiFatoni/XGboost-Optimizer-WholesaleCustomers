import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def declare_features(df, y):
    if not isinstance(y, str):
        raise TypeError("Parameter 'y' must be string.")
    X = df.drop(y, axis=1)
    y = df[y]
    print(f"Features: {X.columns.tolist()}")
    print(f"Target: {y.name}")
    return X, y

def convert_labels(y, before, after):
    y[y == before] = after


def define_dmatrix(X, y):
    data_dmatrix = xgb.DMatrix(data=X,label=y)
    return data_dmatrix

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    return X_train, X_test, y_train, y_test


print("Data engineering module loaded successfully.")