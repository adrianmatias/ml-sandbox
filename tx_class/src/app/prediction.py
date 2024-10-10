from src.app.model import ModelClassification
from src.app.schema_pydantic import Request, Response


class Prediction:
    def __init__(self, model: ModelClassification):
        self.model = model

    def predict(self, request: Request):
        transaction = request.transaction
        is_fraud_prob = self.model.predict_proba(
            X=self.model.get_feature_dict(transaction=transaction)
        )
        response = Response(
            id="some_generated_id",
            id_request=request.id,
            is_fraud_prob=is_fraud_prob,
        )
        return response
