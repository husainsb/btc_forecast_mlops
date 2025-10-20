import os
from fetch_data import execute as fetch
from train_model import execute as train
from train_pyfunc_model import execute as pyfunc_model
from predict import execute as predictions
from predict import get_data

reg_model_name = 'LSTM_BTC_Forecast_Pyfunc'

def main():
    print("Starting main processing....")
    chk = fetch()
    if chk==1:
        print("Data fetched sucessfully!")
        chk = train()
        if chk=='READY':
            print("Data transformed & model trained & logged sucessfully!")
            chk=pyfunc_model(reg_model_name)
            if chk=='READY':
                print("Pyfunc model logged sucessfully!")
                input_data = get_data()
                results = predictions(input_data,reg_model_name)
                print("====Predictions are======")
                print(results)

if __name__ == "__main__":
    main()
