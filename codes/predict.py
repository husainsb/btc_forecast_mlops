import datetime
import os
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, String, DateTime, Float
from sqlalchemy.engine import URL
import dotenv
import pandas as pd
import numpy as np
import joblib
import mlflow

def get_data():
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
        df_back = pd.read_sql(text(f"""select Date, Price, Open, High, Low from 
                                   (select BTC_DATA.*,rank() over(order by Date desc) rn from BTC_DATA) a
                                 where rn<=60"""), conn)

    data_len = df_back.shape[0]
    stacked_data = None
    df_back.sort_values("Date",ascending=False,inplace=True)

    open_data = df_back["Open"].values.reshape((data_len, 1))
    high_data = df_back["High"].values.reshape((data_len, 1))
    low_data = df_back["Low"].values.reshape((data_len, 1))
    close_data = df_back["Price"].values.reshape((data_len, 1))
    stacked_data = np.hstack((close_data, open_data, high_data, low_data))

    return stacked_data

def execute(input: list,reg_model_name: str='LSTM_BTC_Forecast_Pyfunc'):
    try:
        os.chdir('./mlflow_main')
    except FileNotFoundError:
        print(f"Maybe in 'mlflow_main' directory")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    loaded = mlflow.pyfunc.load_model(f"models:/{reg_model_name}@Champion")

    predictions = loaded.predict(input)

    return predictions
if __name__=="__main__":
    input_data = get_data()
    res = execute(input_data)
    print(res)
