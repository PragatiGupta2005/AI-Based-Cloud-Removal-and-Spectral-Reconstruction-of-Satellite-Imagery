"""
FastAPI REST Endpoints Test Suite for CloudClear AI.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.main import app

client = TestClient(app)


class TestCloudClearAPI(unittest.TestCase):

    def test_01_health(self):
        resp = client.get("/api/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["service"], "CloudClear AI")

    def test_02_samples_list(self):
        resp = client.get("/api/v1/samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreater(len(data["samples"]), 0)

    def test_03_predict_sample(self):
        resp = client.post("/api/v1/predict", json={
            "image_id": "scene_west_bengal_nadia",
            "strategy": "adaptive"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("data", data)
        self.assertEqual(data["data"]["status"], "Completed")

    def test_04_quality_and_analytics_endpoints(self):
        pred_id = "scene_west_bengal_nadia"
        q_resp = client.get(f"/api/v1/quality/{pred_id}")
        self.assertEqual(q_resp.status_code, 200)
        q_data = q_resp.json()
        self.assertIn("metrics", q_data)
        self.assertGreater(q_data["metrics"]["psnr"], 20.0)

        ndvi_resp = client.get(f"/api/v1/ndvi/{pred_id}")
        self.assertEqual(ndvi_resp.status_code, 200)
        self.assertIn("ndvi", ndvi_resp.json())

        lc_resp = client.get(f"/api/v1/landcover/{pred_id}")
        self.assertEqual(lc_resp.status_code, 200)
        self.assertIn("landcover", lc_resp.json())

    def test_05_download_endpoints(self):
        pred_id = "scene_west_bengal_nadia"
        rep_resp = client.get(f"/api/v1/report/{pred_id}")
        self.assertEqual(rep_resp.status_code, 200)
        self.assertEqual(rep_resp.headers["content-type"], "application/pdf")

        dl_resp = client.get(f"/api/v1/download/{pred_id}")
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn("image/tiff", dl_resp.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
