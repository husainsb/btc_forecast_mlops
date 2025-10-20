import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import mlflow
import logging
import os, tempfile
from datetime import datetime
from mlflow.client import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.models import infer_signature
import yaml,json, joblib

class MLflowCustomTransformer(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load any necessary artifacts here (e.g., pre-trained models, tokenizers)
        self.xscaler = context.artifacts["scaler_x"]
        self.yscaler = context.artifacts["scaler_y"]
        self.model = context.artifacts["model"]

    def split_series(self,input_data):
        X, y = list(), list()
        for i in range(len(input_data)):
            # End of the input window
            end_ix = i + self.n_steps_in
            # End of the output window (which is subsequeny window after Input)
            out_end_ix = end_ix + self.n_steps_out
            # check if we end of dataset has reached
            if out_end_ix > len(input_data):
                break
            
            # gather input and output windows of the series
            seq_x = input_data[i:end_ix, 1:]  #skip 1st col as its our target column
            seq_y = input_data[end_ix:out_end_ix, 0] #take 1st col as target column

            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def scaler_x(self,input_data):
        X,_ = self.split_series(input_data)
        
        scaler = joblib.load(self.xscaler)
        X_normalized = scaler.fit_transform(X.reshape(-1,self.n_features))
        X_normalized = X_normalized.reshape((X.shape[0],self.n_steps_in,self.n_features))
        return X_normalized 

    def scaler_y(self,predicted_data):

        scaler = joblib.load(self.yscaler)
        y_inv = scaler.inverse_transform(predicted_data)

        return y_inv

    def predict(self, context,model_input):
        self.n_steps_in = 10
        self.n_steps_out = 5
        self.n_features = model_input.shape[1]-1
        # Apply your custom transformation
        saved_model = tensorflow.keras.models.load_model(self.model)
        transformed_df = self.scaler_x(model_input)
        preds_normalized = saved_model.predict(transformed_df)
        preds_data = self.scaler_y(preds_normalized)
        return preds_data

def execute(reg_model_name: str = 'LSTM_BTC_Forecast_Pyfunc'):
    today_date = datetime.strftime(datetime.now().date(),'%Y%m%d')

    #os.chdir("./mlflow_main")
    try:
        os.chdir('./mlflow_main')
    except FileNotFoundError:
        print(f"Maybe in 'mlflow_main' directory")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    exp = mlflow.get_experiment_by_name('BTC_New_Prediction')
    client = MlflowClient()
    #get original trained model from registry - on this Pyfunc wrapper to be created
    model_name='LSTM_BTC_Forecast'
    model_details = client.get_model_version_by_alias(name=model_name,alias='Challenger')
    model_uri = f"models:/{model_name}@Challenger"
    model_id = model_details.model_id

    #get artifacts(original model) location
    artifacts_location = mlflow.get_logged_model(model_id).artifact_location
    artifacts_location = artifacts_location.replace('file://','')

    #create temp folder where original model and scalers can be downloaded
    tmpdir = tempfile.mkdtemp()

    #download all original model and scalers to temp folder and get their path
    model_pkl = download_artifacts(artifact_uri=os.path.join(artifacts_location,'data','model.keras'), dst_path=tmpdir)
    scaler_x_pkl = download_artifacts(artifact_uri=os.path.join('.','scaler_x.pkl'), dst_path=tmpdir)
    scaler_y_pkl = download_artifacts(artifact_uri=os.path.join('.','scaler_y.pkl'), dst_path=tmpdir)

    artifacts = {'scaler_x':scaler_x_pkl,
                'scaler_y':scaler_y_pkl,
                'model':model_pkl}

    mymodel = MLflowCustomTransformer()
    #preds_validation = mymodel.predict(new_validation_data)

    signature = infer_signature(np.ndarray((10,4)),np.ndarray((10,5)))

    with open(os.path.join(artifacts_location,'conda.yaml'),'r') as f:
        conda_env = yaml.safe_load(f)

    #set run name
    run_name = 'LSTM_BTC_Pyfunc_'+today_date

    with mlflow.start_run(run_name=run_name,experiment_id= exp.experiment_id) as run:
        mlflow.pyfunc.log_model(
        name=run_name,
        python_model=mymodel,
        artifacts=artifacts,
        signature=signature,
        conda_env=conda_env)
        
        run_id = run.info.run_id
        #model_uri = f"runs:/{run_id}/model"
        
    runs_list = mlflow.search_logged_models([exp.experiment_id])
    #get Pyfunc model details
    model_id = runs_list[runs_list['source_run_id']==run_id]['model_id'].reset_index(drop=True)[0]
    model_uri = "models:/"+model_id
    result = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    version = int(result.version)  #new version of model

    #set Pyfunc model as Champion
    client.set_registered_model_alias(
        name=reg_model_name,
        alias="Champion",
        version=version)

    status = client.get_model_version_by_alias(reg_model_name,alias='Champion').status

    return status

if __name__ == "__main__":
    status = execute()
    print(status)  #it should be READY
