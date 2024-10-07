import pytest
from pyspark.sql import SparkSession

from fraud.app.conf import read_conf
from fraud.app.logger_custom import LoggerCustom


@pytest.fixture(scope="module")
def conf():
    return read_conf(env="test")


@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[1]").appName("pytest").getOrCreate()

#
# @pytest.fixture(scope="module")
# def logger():
#     return LoggerCustom().logger
