import datetime
import os
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, String, DateTime, Float
from sqlalchemy.engine import URL
import dotenv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def split_series(stacked_data,n_steps_in,n_steps_out):
    X, y = list(), list()
    for i in range(len(stacked_data)):
        # End of the input window
        end_ix = i + n_steps_in
        # End of the output window (which is subsequeny window after Input)
        out_end_ix = end_ix + n_steps_out
        # check if we end of dataset has reached
        if out_end_ix > len(stacked_data):
            break
        
        # gather input and output windows of the series
        seq_x = stacked_data[i:end_ix, 1:]  #skip 1st col as its our target column
        seq_y = stacked_data[end_ix:out_end_ix, 0] #take 1st col as target column

        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def execute():
    dotenv.load_dotenv("./credentials.env")
    
    url = URL.create(
        drivername="mysql+mysqlconnector",   # or "mysql+pymysql"
        username=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),         # raw password, no encoding needed
        host=os.getenv("MYSQL_HOST"),
        port=os.getenv("MYSQL_PORT"),
        database=os.getenv("MYSQL_DB"),
    )

    engine = create_engine(url)

    # ---- 5) Read the data back into pandas ----
    with engine.connect() as conn:
        # You can use pandas.read_sql_table or read_sql with a query
        df_back = pd.read_sql(text(f"SELECT * FROM BTC_DATA"), conn)

    data_len = df_back.shape[0]
    stacked_data = None

    open_data = df_back["Open"].values.reshape((data_len, 1))
    high_data = df_back["High"].values.reshape((data_len, 1))
    low_data = df_back["Low"].values.reshape((data_len, 1))
    close_data = df_back["Price"].values.reshape((data_len, 1))
    stacked_data = np.hstack((close_data, open_data, high_data, low_data))

    n_features=3
    n_steps_in = 10
    n_steps_out = 5

    X,y = split_series(stacked_data,n_steps_in,n_steps_out)
    #load already created scalers 
    scaler_x = joblib.load("./mlflow_main/scaler_x.pkl") #MinMaxScaler()
    X_normalized = scaler_x.fit_transform(X.reshape(-1,n_features))
    X_normalized = X_normalized.reshape((X.shape[0],n_steps_in,n_features))

    X_inv = scaler_x.inverse_transform(X_normalized.reshape(-1,n_features))
    X_inv = X_inv.reshape((X.shape[0],n_steps_in,n_features))

    assert np.allclose(X,X_inv)

    scaler_y = joblib.load("./mlflow_main/scaler_y.pkl") #MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y)

    y_inv = scaler_y.inverse_transform(y_normalized)

    assert np.allclose(y,y_inv)

    del X_inv,y_inv

    train_set = X_normalized[:int(np.ceil(X_normalized.shape[0]*0.7))]
    test_set = X_normalized[train_set.shape[0]:]

    y_train = y_normalized[:int(np.ceil(y_normalized.shape[0]*0.7))]
    y_test = y_normalized[train_set.shape[0]:]

    return train_set,y_train,test_set,y_test,n_features,n_steps_in,n_steps_out
