import logging
import logging.config


class LoggerCustom:
    def __init__(self):
        format_string = "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"

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
                    "": {  # root logger
                        "level": "INFO",
                        "handlers": ["console"],
                    },
                },
            }
        )

    @staticmethod
    def get_logger():
        return logging.getLogger(__name__)
