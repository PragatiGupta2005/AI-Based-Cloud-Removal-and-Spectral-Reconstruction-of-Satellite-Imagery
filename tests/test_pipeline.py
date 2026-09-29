"""
End-to-End Automated Test Suite for CloudClear AI.
Verifies all 11 modules of the geospatial pipeline.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_loader import GeoTIFFLoader, validate_geotiff
from src.preprocessing.preprocessor import ImagePreprocessor
from src.models.cloud_detector import CloudDetectionModel
from src.models.change_detector import ChangeDetectionModel
from src.models.mrr_reconstructor import MRRReconstructionModel
from src.fusion.decision_engine import AdaptiveDecisionEngine
from src.qan.quality_network import QualityAssessmentNetwork
from src.confidence.confidence_estimator import ConfidenceEstimator
from src.analysis.ndvi import NDVIAnalyzer
from src.analysis.landcover import LandCoverClassifier
from src.reports.report_generator import PDFReportGenerator
from src.pipeline.cloudclear_pipeline import CloudClearPipeline


class TestCloudClearAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.sample_cloudy = os.path.join(cls.base_dir, "data", "cloudy", "scene_west_bengal_nadia_cloudy.tif")
        cls.sample_hist = os.path.join(cls.base_dir, "data", "historical", "scene_west_bengal_nadia_historical.tif")
        cls.sample_sar = os.path.join(cls.base_dir, "data", "sar", "scene_west_bengal_nadia_sar.tif")
        cls.sample_clear = os.path.join(cls.base_dir, "data", "clear", "scene_west_bengal_nadia_clear.tif")

    def test_01_geotiff_validation_and_loading(self):
        self.assertTrue(os.path.exists(self.sample_cloudy), f"Sample not found at {self.sample_cloudy}")
        is_valid, msg, meta = validate_geotiff(self.sample_cloudy)
        self.assertTrue(is_valid, f"GeoTIFF validation failed: {msg}")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.bands, 4)

        data, m = GeoTIFFLoader.load_raster(self.sample_cloudy)
        self.assertEqual(data.shape, (512, 512, 4))
        self.assertGreaterEqual(data.min(), 0.0)
        self.assertLessEqual(data.max(), 1.0)

    def test_02_preprocessor_and_normalization(self):
        data, _ = GeoTIFFLoader.load_raster(self.sample_cloudy)
        norm, stats = ImagePreprocessor.normalize_radiometric(data)
        self.assertEqual(norm.shape, data.shape)
        self.assertEqual(norm.dtype, np.float32)

        rgb = ImagePreprocessor.extract_rgb_preview(norm)
        self.assertEqual(rgb.shape, (512, 512, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_03_cloud_and_shadow_detection(self):
        data, _ = GeoTIFFLoader.load_raster(self.sample_cloudy)
        model = CloudDetectionModel(patch_size=256)
        res = model.predict(data)
        self.assertIn("cloud_mask", res)
        self.assertIn("shadow_mask", res)
        self.assertIn("cloud_percentage", res)
        self.assertGreater(res["cloud_percentage"], 0.0)
        self.assertEqual(res["cloud_mask"].shape, (512, 512))

    def test_04_change_detection(self):
        c_data, _ = GeoTIFFLoader.load_raster(self.sample_cloudy)
        h_data, _ = GeoTIFFLoader.load_raster(self.sample_hist)
        s_data, _ = GeoTIFFLoader.load_raster(self.sample_sar)

        cd_model = ChangeDetectionModel(patch_size=256)
        res = cd_model.predict(curr_optical=c_data, hist_optical=h_data, sar_image=s_data)
        self.assertIn("change_probability", res)
        self.assertIn("stable_area", res)
        self.assertEqual(res["change_probability"].shape, (512, 512))

    def test_05_adaptive_decision_engine(self):
        engine = AdaptiveDecisionEngine()
        dec = engine.evaluate_strategy(cloud_pct=25.0, change_score=0.15, changed_area_pct=8.0)
        self.assertEqual(dec["strategy"], "historical")
        self.assertEqual(dec["recommended_candidate"], "C1")

        dec_mod = engine.evaluate_strategy(cloud_pct=30.0, change_score=0.45, changed_area_pct=30.0)
        self.assertEqual(dec_mod["strategy"], "adaptive")
        self.assertEqual(dec_mod["recommended_candidate"], "C3")

    def test_06_mrr_reconstruction(self):
        c_data, _ = GeoTIFFLoader.load_raster(self.sample_cloudy)
        h_data, _ = GeoTIFFLoader.load_raster(self.sample_hist)
        s_data, _ = GeoTIFFLoader.load_raster(self.sample_sar)
        c_mask = np.zeros((512, 512), dtype=np.uint8)
        c_mask[200:300, 200:300] = 1

        mrr = MRRReconstructionModel(patch_size=256)
        cands = mrr.reconstruct_all(c_data, h_data, s_data, c_mask)
        self.assertIn("C1", cands)
        self.assertIn("C2", cands)
        self.assertIn("C3", cands)
        self.assertEqual(cands["C3"].shape, (512, 512, 4))

    def test_07_quality_assessment_network(self):
        c_data, _ = GeoTIFFLoader.load_raster(self.sample_cloudy)
        ref_data, _ = GeoTIFFLoader.load_raster(self.sample_clear)

        qan = QualityAssessmentNetwork()
        metrics = qan.calculate_metrics(c_data, ref_data)
        self.assertGreater(metrics.psnr, 15.0)
        self.assertGreater(metrics.ssim, 0.5)
        self.assertLess(metrics.sam, 25.0)
        self.assertLess(metrics.ergas, 10.0)

    def test_08_confidence_and_analytics(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        prob = np.zeros((512, 512), dtype=np.float32)
        ch_prob = np.zeros((512, 512), dtype=np.float32)

        conf_est = ConfidenceEstimator()
        crep = conf_est.estimate(mask, prob, ch_prob)
        self.assertGreater(crep.high_pct, 50.0)
        self.assertEqual(crep.colored_heatmap.shape, (512, 512, 3))

        c_data, _ = GeoTIFFLoader.load_raster(self.sample_clear)
        ndvi_analyzer = NDVIAnalyzer()
        ndvi_rep = ndvi_analyzer.analyze(c_data)
        self.assertGreater(ndvi_rep.mean_ndvi, -1.0)
        self.assertLessEqual(ndvi_rep.mean_ndvi, 1.0)

        lc_classifier = LandCoverClassifier()
        lc_rep = lc_classifier.classify(c_data)
        self.assertIn("vegetation", lc_rep.distribution)

    def test_09_full_end_to_end_pipeline(self):
        pipeline = CloudClearPipeline(output_dir="outputs_test", reports_dir="reports_test")
        packet = pipeline.run(
            cloudy_path=self.sample_cloudy,
            historical_path=self.sample_hist,
            sar_path=self.sample_sar,
            clear_reference_path=self.sample_clear,
            image_id="test_nadia_01"
        )
        self.assertEqual(packet.status, "Completed")
        self.assertIsNotNone(packet.cloud_free_geotiff_path)
        self.assertTrue(os.path.exists(packet.cloud_free_geotiff_path))
        self.assertTrue(os.path.exists(packet.report_pdf_path))
        self.assertGreater(packet.quality_metrics.psnr, 25.0)
        self.assertGreater(packet.quality_metrics.ssim, 0.80)


if __name__ == "__main__":
    unittest.main()
