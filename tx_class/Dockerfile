FROM python:3.7-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install "ray[serve]"

COPY . /app

CMD ["python", "src/app/main.py"]
