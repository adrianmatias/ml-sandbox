from fastapi import FastAPI

from src.app.model import ModelClassification
from src.app.prediction import Prediction
from src.app.schema_pydantic import Request, Response

app = FastAPI()


@app.get("/status")
async def get_status():
    return {"status": "API is running"}


@app.post("/predict", response_model=Response)
async def predict(request: Request):
    model = ModelClassification()
    prediction = Prediction(model=model)
    return prediction.predict(request=request)
