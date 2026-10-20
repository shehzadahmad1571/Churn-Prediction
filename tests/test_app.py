import unittest

import app as churn_app


class ChurnAppTests(unittest.TestCase):
    def setUp(self):
        churn_app.app.config.update(TESTING=True)
        self.client = churn_app.app.test_client()

    def valid_customer(self):
        return {
            "gender": "Male",
            "SeniorCitizen": "0",
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": "12",
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": "65.50",
            "TotalCharges": "789.00",
        }

    def low_risk_customer(self):
        return {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": "72",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": "55.00",
            "TotalCharges": "3960.00",
        }

    def test_homepage_shows_customer_form(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Churn prediction", response.data)
        self.assertIn(b"Payment method", response.data)

    def test_expected_churn_customer_gets_churn_prediction(self):
        response = self.client.post("/predict", data=self.valid_customer())

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated churn probability", response.data)
        self.assertIn(b"Likely to churn", response.data)

    def test_expected_no_churn_customer_gets_stay_prediction(self):
        response = self.client.post("/predict", data=self.low_risk_customer())

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Likely to stay", response.data)

    def test_missing_value_is_reported_without_prediction(self):
        customer = self.valid_customer()
        customer["tenure"] = ""

        response = self.client.post("/predict", data=customer)

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"This field is required.", response.data)
        self.assertNotIn(b"Estimated churn probability", response.data)

    def test_out_of_range_value_is_reported(self):
        customer = self.valid_customer()
        customer["tenure"] = "100"

        response = self.client.post("/predict", data=customer)

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Enter a value from 0 to 72.", response.data)

    def test_unexpected_category_is_rejected(self):
        customer = self.valid_customer()
        customer["Contract"] = "Lifetime"

        response = self.client.post("/predict", data=customer)

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Choose one of the listed options.", response.data)

    def test_allowed_phone_service_options_are_accepted(self):
        customer = self.valid_customer()
        customer["PhoneService"] = "No"

        response = self.client.post("/predict", data=customer)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated churn probability", response.data)


if __name__ == "__main__":
    unittest.main()
