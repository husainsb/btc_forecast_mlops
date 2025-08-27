from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import asyncio

app = FastAPI()

#mlflow.set_tracking_uri('/Users/husainsb/Documents/btc_forecast/ingest_train_project/ingest_train/mlflow_main/mlruns')
# Load your MLflow pyfunc model which is in root directory
model = mlflow.pyfunc.load_model(".")

class ListInput(BaseModel):
    data: list  # expects a nested list for multi-dimensional arrays

def predict(input):
    return model.predict(input)

@app.post("/invocations")
async def infer(input: ListInput):
    """
    The payload must be a JSON object like: {"data": <input_data>}
    """
    #payload = await request.json()
    payload = np.array(input.data)

    loop = asyncio.get_event_loop()
    # Run sync prediction in a thread pool, non-blocking the event loop
    preds = await loop.run_in_executor(None,predict,payload)
    # try to convert to list if numpy or pandas
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    return {"predictions": preds}

'''
@app.post("/syncinvocations")
def infer_sync(input: ListInput):
    """
    The payload must be a JSON object like: {"data": <input_data>}
    """
    #payload = await request.json()
    payload = np.array(input.data)
    preds = predict(payload)

    # try to convert to list if numpy or pandas
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    return {"predictions": preds}
'''