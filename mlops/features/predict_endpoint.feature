Feature: Test the /predict endpoint

  Scenario: Get predictions for a given text
    Given the API is running
    When I send a POST request to the "/predict" endpoint with text "Apple is looking at buying U.K. startup for $1 billion"
    Then the response status code should be 200
    And the response should contain entities
