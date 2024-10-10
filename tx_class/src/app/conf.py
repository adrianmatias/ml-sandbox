import os.path

from pydantic import BaseModel


class Api(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Conf(BaseModel):
    api: Api = Api()


def read_conf(env: str) -> Conf:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, f"conf/conf_{env}.json")
    print(file_name)

    return Conf.parse_file(file_name)


CONF_DEFAULT = Conf()
