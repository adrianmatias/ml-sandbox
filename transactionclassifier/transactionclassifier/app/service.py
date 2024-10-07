from fastapi import FastAPI

from fraud.app.model import ModelClassification
from fraud.app.prediction import Prediction
from fraud.app.schema_pydantic import Request, Response

app = FastAPI()


@app.get("/status")
async def get_status():
    return {"status": "API is running"}


@app.post("/predict", response_model=Response)
async def predict(request: Request):
    model = ModelClassification()
    prediction = Prediction(model=model)
    return prediction.predict(request=request)
