import math
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from app import FEATURES, SUPPORTED_VERSIONS, create_app


PAYLOAD = dict(zip(FEATURES, (0.02, -0.044, 0.06, -0.03, -0.02,
                             0.03, -0.02, 0.02, 0.02, -0.001)))


class ModelApiTests(unittest.TestCase):
    def test_health_reports_selected_version(self):
        for version in SUPPORTED_VERSIONS:
            with self.subTest(version=version):
                with TestClient(create_app(version)) as client:
                    response = client.get("/health")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {
                        "status": "ok", "model_version": version,
                    })

    def test_prediction_matches_selected_model(self):
        for version in SUPPORTED_VERSIONS:
            with self.subTest(version=version):
                application = create_app(version)
                features = pd.DataFrame([PAYLOAD], columns=FEATURES)
                expected = float(application.state.model.predict(features)[0])
                with TestClient(application) as client:
                    response = client.post("/predict", json=PAYLOAD)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(set(response.json()), {"prediction"})
                    prediction = response.json()["prediction"]
                    self.assertIsInstance(prediction, (int, float))
                    self.assertTrue(math.isfinite(prediction))
                    self.assertAlmostEqual(prediction, expected)

    def test_missing_feature_returns_json_validation_error(self):
        for version in SUPPORTED_VERSIONS:
            with self.subTest(version=version):
                with TestClient(create_app(version)) as client:
                    for feature in FEATURES:
                        with self.subTest(feature=feature):
                            payload = dict(PAYLOAD)
                            del payload[feature]
                            response = client.post("/predict", json=payload)
                            self.assertEqual(response.status_code, 422)
                            self.assertTrue(any(
                                error["loc"] == ["body", feature]
                                for error in response.json()["detail"]
                            ))


if __name__ == "__main__":
    unittest.main()
