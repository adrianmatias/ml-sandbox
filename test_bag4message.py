import logging
import string
import time
from typing import Callable

import pytest

from bag4message_solution import is_message_in_bag_naive, is_message_in_bag_efficient

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
LOGGER.info(f"pytest version {pytest.__version__}")

N_TIMES = 1000000


class TestBag4Message:
    def test_description_case(self):
        assert is_message_in_bag_efficient(
            message="hello world", bag="oll hw or aidsj iejsjhllalelilolu"
        )

    def test_empty_fields(self):
        assert is_message_in_bag_efficient(message="", bag="")

    def test_empty_message(self):
        assert is_message_in_bag_efficient(
            message="", bag="oll hw or aidsj iejsjhllalelilolu"
        )

    def test_insufficient_bag(self):
        assert not is_message_in_bag_efficient(message="abc", bag="fb")

    def test_sufficient_bag(self):
        assert is_message_in_bag_efficient(message="abc", bag="fbacbbb")

    def test_equal_answer(self):
        for message, bag in [
            ("", ""),
            ("casax", "casa"),
            ("abc", "fb"),
            ("abc", "fbacbbb"),
            ("abcac", "fbacbbb"),
            ("hello world", "oll hw or aidsj iejsjhllalelilolu"),
        ]:
            answer_naive = is_message_in_bag_naive(message=message, bag=bag)
            answer_efficient = is_message_in_bag_efficient(message=message, bag=bag)
            assert answer_naive == answer_efficient

    def test_heavy_input(self):

        message = string.ascii_lowercase * N_TIMES
        bag = string.ascii_lowercase * N_TIMES
        answer = True

        with pytest.raises(TimeoutError):
            is_message_in_bag_naive(message=message, bag=bag)

        assert is_message_in_bag_efficient(message=message, bag=bag) == answer


def get_time(func: Callable, message: str, bag: str) -> float:
    n_rounds = 10
    n_rounds_warmup = 3

    start_time = None
    for n in range(n_rounds):
        if n == n_rounds_warmup:
            start_time = time.time()
        func(message, bag)

    time_mean = (time.time() - start_time) / (n_rounds - n_rounds_warmup)
    LOGGER.info(f"{func.__name__} time_mean: {time_mean} s")

    return time_mean
