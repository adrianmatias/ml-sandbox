# tx_class

This module implements a generic classifier for card transactions.
A rest api exposes the classifier predict. The restapi defines request and response to enable prediction.
The app is dockerized and distributed using ray-serve.
The data consists on a set generated out of random transactions.


## install

pred
```
pip install -r requirements.txt
```
train
```
pip install -r requirements_train.txt
```
dev
```
pip install -r requirements.txt -r requirements_dev.txt
```

docker python3.6
```
cd docker-python36
docker-python36$ docker build -t python36-env .
docker-python36$ docker exec -it python36-container bash
```

## lint
```
(.venv_dev)$ sh scripts/lint.sh
```
## test
```
(.venv_dev)$ sh scripts/test.sh
```

## launch dockerized api
```
docker build -t tx_class-api .
docker run -d -p 8000:8000 --env ENV=<env> tx_class-api
```

## request prediction
```
curl -X POST http://0.0.0.0:8000/predict -H "Content-Type: application/json" -d '{"id": "1", "transaction": {"amount": 15, "transaction_type": "purchase"}}'
```