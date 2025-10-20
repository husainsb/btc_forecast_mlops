import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import mlflow
import logging
import os
from datetime import datetime
from mlflow.client import MlflowClient
from data_prep import execute as dp_execute

def execute():
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.setLevel(logging.ERROR) 

    today_date = datetime.strftime(datetime.now().date(),'%Y%m%d')

    #mlflow.set_experiment(experiment_id=exp.experiment_id)

    train_set,y_train,_,_,n_features,n_steps_in,n_steps_out = dp_execute()

    model =Sequential()
    optimizer = Adam(learning_rate=0.001)

    # LSTM
    model.add(LSTM(100,activation='sigmoid',return_sequences=True,input_shape=(n_steps_in,n_features)))
    model.add(LSTM(100,activation='tanh',return_sequences=True))
    model.add(LSTM(100,activation='tanh'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=optimizer,loss='mse')

    # start MLflow config
    try:
        os.chdir('./mlflow_main')
    except FileNotFoundError:
        print(f"Error: Directory 'mlflow_main' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    exp = mlflow.get_experiment_by_name('BTC_New_Prediction')
    client = MlflowClient()  #tracking/registry URI will be same as os.chdir
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name='LSTM_3L_'+today_date,experiment_id= exp.experiment_id) as run:
        model.fit(train_set, y_train, epochs=150, batch_size=32, verbose=0,validation_split=0.15,validation_freq=10)
        #mlflow.keras.log_model(model,"model")
        run_id = run.info.run_id

    runs_list = mlflow.search_logged_models([exp.experiment_id])

    model_id = runs_list[runs_list['source_run_id']==run_id]['model_id'].reset_index(drop=True)[0]
    model_uri = "models:/"+model_id
    model_name = 'LSTM_BTC_Forecast'
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = int(result.version)  #new version of model

    client.set_registered_model_alias(
        name=model_name,
        alias="Challenger",
        version=version)
    

    status = client.get_model_version_by_alias(model_name,alias='Challenger').status

    return status

if __name__ == "__main__":
    status = execute()
    print(status)  #it should be READY
