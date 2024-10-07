import pytest

from fraud.app.model import ModelClassification, ModelClassificationCatBoost
from fraud.app.schema_pydantic import Transaction
from fraud.app.train.data import Data
from tests.fixture_set import spark


def test_basic():
    transaction = Transaction(
        id="id_1",
        amount=10,
        transaction_type="transaction_type_1",
        code="a43",
    )
    model = ModelClassification()

    feature_dict = model.get_feature_dict(transaction=transaction)
    assert feature_dict == {
        "amount": 10.0,
        "code_0": "a",
        "code_1": "4",
        "code_2": "3",
        "dt_day": 1,
        "dt_hour": 23,
        "dt_month": 1,
        "dt_weekday": 3,
        "dt_year": 1970,
        "transaction_type": "transaction_type_1",
    }
    assert model.predict_proba(X=feature_dict) == 0.1


def test_model_train_and_evaluate(spark):
    count = 1000

    data = Data(spark=spark)
    data.build_random(count=count)
    data.get_df(model=ModelClassification())
    data.df.show()

    data.split(test_min_dtime="2024-01-01")

    df_train = data.df_train.toPandas()
    df_test = data.df_test.toPandas()

    model = ModelClassificationCatBoost()
    model_card = model.train_and_evaluate(df_train=df_train, df_test=df_test)

    assert model_card.evaluation_dict.keys() == {"train", "test", "train_cv"}
    assert model_card.feat_importance.keys() == set(model.schema.x)
    assert sum(model_card.feat_importance.values()) == pytest.approx(1)
