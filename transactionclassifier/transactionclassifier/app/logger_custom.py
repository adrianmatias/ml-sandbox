import logging


class LoggerCustom:
    def __init__(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            " ".join(
                [
                    "%(asctime)s,%(msecs)d",
                    "%(levelname)-8s",
                    "[%(filename)s:%(funcName)s:%(lineno)d]",
                    "%(message)s",
                ]
            )
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger
