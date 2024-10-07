import time

from ray import serve

from fraud.app.conf import read_conf
from fraud.app.model import ModelClassification
from fraud.app.prediction import Prediction
from fraud.app.service import app  # Import the FastAPI app

serve.start()


@serve.deployment
@serve.ingress(app)  # Wrap the FastAPI app with Ray Serve
class FraudService:
    def __init__(self):
        self.prediction = Prediction(model=ModelClassification())


FraudService.deploy()

while True:
    time.sleep(3600)
