import shutil

from fraud.app.model import ModelClassification
from fraud.app.train.data import Data, TransactionRandom
from tests.fixture_set import spark


def test_basic():
    assert TransactionRandom().code is not None


def test_build_random(spark):
    count = 3

    data = Data(spark=spark)
    data.build_random(count=count)
    transaction_list = data.transaction_rdd.collect()

    assert len(transaction_list) == count
    assert transaction_list[0].is_fraud is not None


def test_persistence(spark):
    count = 3
    path = "test_persistence.parquet"

    data = Data(spark=spark)
    data.build_random(count=count)

    remove_if_exists(path=path)
    data.save(path=path)

    data.transaction_rdd = None
    data.load(path=path)

    assert data.transaction_rdd.count() == count
    remove_if_exists(path=path)


def test_data_spit(spark):
    count = 20

    data = Data(spark=spark)
    data.build_random(count=count)
    data.get_df(model=ModelClassification())
    data.df.show()

    data.split(test_min_dtime="2024-01-01")

    col_list = ["dt_year", "dt_month", "dt_day"]
    data.df_train.select(col_list).describe().show()
    data.df_test.select(col_list).describe().show()


def remove_if_exists(path: str):
    try:
        shutil.rmtree(path=path)
    except:
        pass
