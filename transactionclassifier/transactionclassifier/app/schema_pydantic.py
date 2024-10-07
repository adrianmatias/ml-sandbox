from pydantic import BaseModel


class Transaction(BaseModel):
    id: str
    dtime: str = "1970-01-01 23:59:59"
    amount: float = -999999.99
    transaction_type: str = "unknown"
    code: str = "x"


class TransactionLabeled(Transaction):
    is_fraud: bool


class Request(BaseModel):
    id: str
    transaction: Transaction


class Response(BaseModel):
    id: str
    id_request: str
    is_fraud_prob: float
