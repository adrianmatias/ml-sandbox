import random
import uuid
from datetime import datetime, timedelta

from pyspark import RDD
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, concat, lit, to_date

from src.app.date_time import DateTime
from src.app.model import ModelClassification
from src.app.schema_pydantic import TransactionLabeled


class TransactionRandom(TransactionLabeled):
    def __init__(self):
        super().__init__(
            id=uuid.uuid4().hex,
            is_fraud=int(random.random() < 0.1),
            amount=round(random.uniform(10.0, 1000.0), 2),
            transaction_type=random.choice(["credit", "debit"]),
            code=random.choice(["ab1", "ab2", "ac1", "ba1"]),
            dtime=self.get_dtime_random(),
        )

    def get_dtime_random(
        self,
    ) -> str:
        dt_min = datetime(2023, 12, 1)
        dt_max = datetime(2024, 2, 1)

        delta = dt_max - dt_min
        dt_obj = dt_min + timedelta(
            seconds=random.randint(0, int(delta.total_seconds()))
        )
        return DateTime().get_str(dt_obj=dt_obj)


class Data:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.transaction_rdd: RDD = spark.sparkContext.parallelize([])
        self.df: DataFrame = None
        self.df_train: DataFrame = None
        self.df_test: DataFrame = None

    def build_random(self, count: int):
        sc = self.spark.sparkContext
        sc.parallelize(range(count))
        self.transaction_rdd = self.spark.sparkContext.parallelize(range(count)).map(
            lambda _: TransactionRandom()
        )

    def save(self, path: str):
        self.transaction_rdd.map(lambda t: t.dict()).toDF().write.parquet(path=path)

    def load(self, path: str):
        self.transaction_rdd = self.spark.read.parquet(path)

    def get_df(self, model: ModelClassification):
        self.df = self.transaction_rdd.map(
            lambda t: model.get_feature_and_label_dict(transaction=t)
        ).toDF()

    def split(self, test_min_dtime: str):
        col_date = "date"
        df = self.df.withColumn(
            colName=col_date,
            col=to_date(
                concat(
                    col("dt_year"), lit("-"), col("dt_month"), lit("-"), col("dt_day")
                ),
                "yyyy-MM-dd",
            ),
        )

        self.df_train = df.filter(col(col_date) < lit(test_min_dtime)).drop(col_date)
        self.df_test = df.filter(col(col_date) >= lit(test_min_dtime)).drop(col_date)
