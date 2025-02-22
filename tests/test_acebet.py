import unittest
from fastapi.testclient import TestClient

# Initializing unit tests with the TestClient to simulate HTTP requests.
from acebet.app.main import app


class TestAceBetAPI(unittest.TestCase):
    def setUp(self):
        # Setting up the test environment with the FastAPI TestClient instance.
        self.client = TestClient(app)

    def test_login_for_access_token(self):
        # Testing user authentication by sending a POST request for an access token.
        form_data = {"username": "johndoe", "password": "secret"}
        # Sending the POST request.
        response = self.client.post("/token", data=form_data)
        # Asserting the expected response status (HTTP 200 - OK).
        self.assertEqual(response.status_code, 200)
        # Validating the response content for the presence of access token.
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")

    def test_read_users_me(self):
        # Testing retrieval of user profile using an access token.
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        # Sending a GET request for the user's profile.
        response = self.client.get("/users/me/", headers=headers)
        # Ensuring the expected response status (HTTP 200 - OK).
        self.assertEqual(response.status_code, 200)
        # Verifying the retrieved user data against the expected values.
        data = response.json()
        self.assertEqual(data["username"], "johndoe")

    def test_read_own_items(self):
        # Testing retrieval of user-owned items using an access token.
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        # Sending a GET request for the user's items.
        response = self.client.get("/users/me/items/", headers=headers)
        # Checking the expected response status (HTTP 200 - OK).
        self.assertEqual(response.status_code, 200)
        # Validating the retrieved item data against the user's ownership.
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["owner"], "johndoe")

    def test_predict_match_outcome(self):
        # Testing match outcome prediction using an access token.
        access_token = self.get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        # Preparing prediction data - player names and match date.
        prediction_data = {
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "testing": True,
        }
        # Sending a POST request to predict the match outcome.
        response = self.client.post("/predict/", headers=headers, json=prediction_data)
        # Verifying the expected response status (HTTP 200 - OK).
        self.assertEqual(response.status_code, 200)
        # Validating the prediction result - player name, probability, and class.
        data = response.json()
        self.assertIn("player_name", data)
        self.assertIn("prob", data)
        self.assertIn("class_", data)

    def get_access_token(self):
        # Simulating a user login to acquire an access token.
        form_data = {"username": "johndoe", "password": "secret"}
        # Sending the login POST request and extracting the access token.
        response = self.client.post("/token", data=form_data)
        data = response.json()
        return data["access_token"]


# Executing the test suite if run as the main module.
if __name__ == "__main__":
    unittest.main()


# {
#   "p1_name": "Fognini F.",
#   "p2_name": "Jarry N.",
#   "date": "2018-03-04",
#   "testing": false
# }
