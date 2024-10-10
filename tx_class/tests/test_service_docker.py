import subprocess
import time

import pytest
import requests
from tests.fixture_set import conf


@pytest.fixture(scope="module")
def launch_docker(conf):
    print("Building Docker image...")
    subprocess.run(["docker", "build", "-t", "fraud-api", "."], check=True)

    print("Running Docker container...")
    env = "test"
    container_id = (
        subprocess.check_output(
            [
                "docker",
                "run",
                "-d",
                "-p",
                f"{conf.api.port}:{conf.api.port}",
                "--env",
                f"ENV={env}",
                "fraud-api",
            ]
        )
        .decode()
        .strip()
    )

    url = f"http://{conf.api.host}:{conf.api.port}/status"
    max_retries = 10
    for _ in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Service is up and running!")
                break
        except requests.exceptions.ConnectionError:
            print("Waiting for service...")
            time.sleep(1)
    else:
        subprocess.run(["docker", "stop", container_id], check=True)
        subprocess.run(["docker", "rm", container_id], check=True)
        pytest.fail("API service failed to start in time.")

    yield container_id

    print("Stopping and removing Docker container...")
    subprocess.run(["docker", "stop", container_id], check=True)
    subprocess.run(["docker", "rm", container_id], check=True)


def test_api_predict(launch_docker, conf):
    url = f"http://{conf.api.host}:{conf.api.port}/predict"
    headers = {"Content-Type": "application/json"}
    payload = {"id": "1", "transaction": {"amount": 15, "transaction_type": "purchase"}}

    response = requests.post(url, json=payload, headers=headers)

    assert response.status_code == 200

    response_json = response.json()
    assert "id" in response_json
