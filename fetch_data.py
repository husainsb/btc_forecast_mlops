import datetime
import requests
import os
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, String, DateTime, Float
from sqlalchemy.engine import URL
import dotenv
import pandas as pd
import numpy as np
import joblib

def execute():
    dotenv.load_dotenv("./credentials.env")

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    querystring = {"vs_currency":"usd","days":"30"}
    headers = {"x-cg-demo-api-key": os.getenv("API_KEY")} 

    response = requests.get(url, headers=headers, params=querystring)

    btc_new_data = pd.DataFrame(response.json(),columns=['Datetime', 'Open','High','Low','Price'])

    btc_new_data['Datetime'] = btc_new_data['Datetime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000, datetime.UTC))
    btc_new_data['Hour'] = btc_new_data['Datetime'].apply(lambda x: int(datetime.datetime.strftime(x,"%H")))

    btc_new_data = btc_new_data[btc_new_data['Hour']==0].copy()

    btc_new_data.rename(columns={'Datetime':'Date'},inplace=True)
    btc_new_data.drop(columns='Hour',inplace=True)

    url = URL.create(
        drivername="mysql+mysqlconnector",   # or "mysql+pymysql"
        username=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),         # raw password, no encoding needed
        host=os.getenv("MYSQL_HOST"),
        port=os.getenv("MYSQL_PORT"),
        database=os.getenv("MYSQL_DB"),
    )

    engine = create_engine(url)

    all_cols = list(btc_new_data.columns)               # all target columns, same order as target table
    tmp_table = "TMP_BTC_DATA"

    with engine.begin() as conn:
        # 1) Create staging table schema identical to target (quick hack: create empty copy)
        conn.execute(text(f"DROP TEMPORARY TABLE IF EXISTS {tmp_table};"))
        conn.execute(text(f"CREATE TEMPORARY TABLE {tmp_table} SELECT * FROM BTC_DATA LIMIT 1;"))

        # 2) Bulk load DataFrame into staging
        btc_new_data.to_sql(tmp_table, conn, if_exists="append", index=False, method="multi", chunksize=50)

        # 3) Insert new rows only
        cols_csv = ", ".join(f"`{c}`" for c in all_cols)

        sql = f"""
            INSERT INTO BTC_DATA ({cols_csv})
            SELECT {cols_csv}
            FROM {tmp_table} t
            WHERE NOT EXISTS (SELECT 1 FROM BTC_DATA m WHERE m.Date=t.Date);
        """
        conn.execute(text(sql))

        return 1