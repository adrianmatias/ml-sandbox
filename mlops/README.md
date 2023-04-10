# Spacy-Sagemaker API

This project demonstrates how to serve a Spacy NER model as a FastAPI application within a Docker container. The API can be easily launched using Docker Compose.

## Prerequisites

- Docker
- Docker Compose

## Build and launch
```
docker build -t spacy-sagemaker .
docker-compose up
docker-compose down // to stop and remove containers
```
## Stop and remove containers
```
docker-compose down
```

This is sample curl to request a prediction
```
curl - X POST "http://0.0.0.0:8080/predict" -
  H "Content-Type: application/json" -
  d '{"text": "Apple is looking at buying U.K. startup for $1 billion"}'
{"entities":[["Apple","ORG"],["U.K.","GPE"],["$1 billion","MONEY"]]}(ml-sandbox-mlops-test)
```
And its response.
```
{"entities":[["Apple","ORG"],["U.K.","GPE"],["$1 billion","MONEY"]]}
```

## Integration Tests
```
conda env create -f environment.yaml
behave feature
```

A successful run should look like
```
Feature: Test the /predict endpoint # features/predict_endpoint.feature:1

  Scenario: Get predictions for a given text                                                                                 # features/predict_endpoint.feature:3
    Given the API is running                                                                                                 # steps/predict_endpoint.py:10 0.000s
    When I send a POST request to the "/predict" endpoint with text "Apple is looking at buying U.K. startup for $1 billion" # steps/predict_endpoint.py:14 0.011s
    Then the response status code should be 200                                                                              # steps/predict_endpoint.py:19 0.000s
    And the response should contain entities                                                                                 # steps/predict_endpoint.py:23 0.000s

1 feature passed, 0 failed, 0 skipped
1 scenario passed, 0 failed, 0 skipped
4 steps passed, 0 failed, 0 skipped, 0 undefined
Took 0m0.012s
```
