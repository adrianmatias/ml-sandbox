import functools
import inspect
import logging
import logging.config


def log_init(cls):
    """Class decorator that logs class name and init attributes after __init__."""

    original_init = cls.__init__
    logger = logging.getLogger(cls.__module__)
    source_file = inspect.getfile(cls)

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        record = logger.makeRecord(
            name=logger.name,
            level=logging.INFO,
            fn=source_file,
            lno=0,
            msg=f"{cls.__name__}: {self.__dict__}",
            args=(),
            exc_info=None,
            func="__init__",
        )
        logger.handle(record)

    cls.__init__ = new_init
    return cls


class LoggerCustom:
    def __init__(self):
        format_string = " ".join(
            [
                "%(asctime)s,%(msecs)d",
                "%(levelname)-8s",
                "[%(filename)s:%(funcName)s:%(lineno)d]",
                "%(message)s",
            ]
        )

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "simple": {
                        "format": format_string,
                    },
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "DEBUG",
                        "formatter": "simple",
                        "stream": "ext://sys.stdout",
                    },
                },
                "loggers": {
                    "": {
                        "level": "INFO",
                        "handlers": ["console"],
                    },
                },
            }
        )

    @staticmethod
    def get_logger():
        return logging.getLogger(__name__)


LOGGER = LoggerCustom().get_logger()
