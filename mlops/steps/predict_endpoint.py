import time

import requests
from behave import given, then, when

DOCKER_IMAGE = "spacy-sagemaker"
PORT = "8080"
API_URL = f"http://localhost:{PORT}"


@given("the API is running")
def step_api_is_running(context):
    pass


@when('I send a POST request to the "/predict" endpoint with text "{text}"')
def step_send_post_request(context, text):
    response = requests.post(f"{API_URL}/predict", json={"text": text})
    context.response = response


@then("the response status code should be {status_code:d}")
def step_response_status_code(context, status_code):
    assert context.response.status_code == status_code


@then("the response should contain entities")
def step_response_contains_entities(context):
    response_data = context.response.json()
    assert "entities" in response_data
    assert len(response_data["entities"]) > 0
