import unittest

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.observability import setup_metrics


class PrometheusObservabilityTests(unittest.TestCase):
    def test_core_metrics_and_metrics_endpoint_exclusion(self):
        app = FastAPI()

        @app.get("/ok")
        async def ok():
            return {"status": "ok"}

        @app.get("/failure")
        async def failure():
            raise HTTPException(status_code=503, detail="unavailable")

        setup_metrics(app)
        client = TestClient(app)

        self.assertEqual(client.get("/ok").status_code, 200)
        self.assertEqual(client.get("/failure").status_code, 503)

        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        body = response.text

        self.assertIn("http_requests_total", body)
        self.assertIn("http_request_duration_seconds_bucket", body)
        self.assertIn("http_requests_inprogress", body)
        self.assertIn('handler="/ok"', body)
        self.assertIn('handler="/failure"', body)
        self.assertIn('status="2xx"', body)
        self.assertIn('status="5xx"', body)
        self.assertNotIn('handler="/metrics"', body)

    def test_metrics_can_be_disabled(self):
        app = FastAPI()
        setup_metrics(app, enabled=False)

        response = TestClient(app).get("/metrics")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
