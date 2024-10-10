from datetime import datetime


class DateTime:
    def __init__(self):
        self.format = "%Y-%m-%d %H:%M:%S"

    def get_str(self, dt_obj: datetime) -> str:
        return dt_obj.strftime(self.format)

    def get_obj(self, dt_str: str) -> datetime:
        return datetime.strptime(dt_str, self.format)
