import spacy
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
nlp = spacy.load("en_core_web_sm")


class InputData(BaseModel):
    text: str


@app.post("/predict")
async def predict(input_data: InputData):
    doc = nlp(input_data.text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {"entities": entities}
