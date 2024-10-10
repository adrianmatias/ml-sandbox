import pytest
from pyspark.sql import SparkSession
from src.app.conf import read_conf


@pytest.fixture(scope="module")
def conf():
    return read_conf(env="test")


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[1]").appName("pytest").getOrCreate()
