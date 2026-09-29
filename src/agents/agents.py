"""A1..A10 agents wrapping existing CloudClear modules.

Agents do NOT duplicate model code; they own preconditions, invocation,
and structured logging so the pipeline is an agent system, not just
sequential function calls.
"""
import os
from typing import Any, Dict
import numpy as np

from .base import BaseAgent, AgentMessage


class DataRetrievalAgent(BaseAgent):
    name = "A1-data"
    role = "retrieve + validate optical/SAR/historical"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        from ..preprocessing.data_loader import GeoTIFFLoader
        loader: GeoTIFFLoader = state["loader"]
        cloudy, meta = loader.load_raster(state["cloudy_path"])
        H, W = cloudy.shape[:2]
        hist = loader.load_raster(state["historical_path"])[0] \
            if state.get("historical_path") and os.path.exists(state["historical_path"]) \
            else cloudy.copy()
        sar = loader.load_raster(state["sar_path"])[0] \
            if state.get("sar_path") and os.path.exists(state["sar_path"]) \
            else np.zeros((H, W, 2), dtype=np.float32)
        ref = loader.load_raster(state["clear_reference_path"])[0] \
            if state.get("clear_reference_path") and os.path.exists(state["clear_reference_path"]) \
            else hist.copy()
        state.update({"meta": meta, "cloudy_raw": cloudy, "hist_raw": hist,
                      "sar_raw": sar, "ref_raw": ref})
        return AgentMessage(self.name, "retrieved",
                            f"loaded {meta.width}x{meta.height}x{cloudy.shape[-1]} sensor={meta.sensor}",
                            {"has_sar": bool(state.get("sar_path") and os.path.exists(state["sar_path"])),
                             "has_hist": bool(state.get("historical_path") and os.path.exists(state["historical_path"]))})


class PreprocessingAgent(BaseAgent):
    name = "A2-preprocess"
    role = "radiometric normalization"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        pre = state["preprocessor"]
        state["cloudy_n"], _ = pre.normalize_radiometric(state["cloudy_raw"])
        state["hist_n"], _ = pre.normalize_radiometric(state["hist_raw"])
        state["sar_n"], _ = pre.normalize_radiometric(state["sar_raw"])
        state["ref_n"], _ = pre.normalize_radiometric(state["ref_raw"])
        return AgentMessage(self.name, "normalized", "radiometric normalization done")


class CloudDetectionAgent(BaseAgent):
    name = "A3-cloud"
    role = "cloud/shadow segmentation"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        res = state["cloud_model"].predict(state["cloudy_n"])
        state["cloud_res"] = res
        return AgentMessage(self.name, "segmented",
                            f"cloud={res['cloud_percentage']}% shadow={res['shadow_percentage']}%",
                            {"cloud_pct": res["cloud_percentage"]})


class ChangeDetectionAgent(BaseAgent):
    name = "A4-change"
    role = "temporal change"

    def can_handle(self, state: Dict[str, Any]) -> bool:
        # Autonomous skip: if cloud covers everything, change is unreliable;
        # still run but flag low confidence downstream.
        return "cloudy_n" in state

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        res = state["change_model"].predict(
            curr_optical=state["cloudy_n"], hist_optical=state["hist_n"],
            sar_image=state["sar_n"], cloud_mask=state["cloud_res"]["cloud_mask"])
        state["change_res"] = res
        return AgentMessage(self.name, "compared",
                            f"change_score={res['change_score']} changed={res['changed_area']}%",
                            {"change_score": res["change_score"]})


class DecisionAgent(BaseAgent):
    name = "A5-decision"
    role = "strategy routing"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        dec = state["decision_engine"].evaluate_strategy(
            cloud_pct=state["cloud_res"]["cloud_percentage"],
            change_score=state["change_res"]["change_score"],
            changed_area_pct=state["change_res"]["changed_area"],
            has_sar=state.get("has_sar", True), has_historical=state.get("has_hist", True))
        ov = state.get("strategy_override")
        if ov in ["historical", "sar", "adaptive"]:
            dec["strategy"] = ov
            dec["recommended_candidate"] = {"historical": "C1", "sar": "C2", "adaptive": "C3"}[ov]
            dec["rationale"] += " [human override]" if "rationale" in dec else "[human override]"
        state["decision"] = dec
        return AgentMessage(self.name, "routed", f"strategy={dec['strategy']} -> {dec['recommended_candidate']}",
                            {"strategy": dec["strategy"]})


class FusionReconstructionAgent(BaseAgent):
    name = "A6-A7-fusion-recon"
    role = "cross-attention fusion + MRR C1/C2/C3"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        import numpy as np
        cr = state["cloud_res"]
        occ = ((cr["cloud_mask"] > 0) | (cr.get("shadow_mask", 0) > 0)).astype(np.uint8)
        state["occ_mask"] = occ
        cands = state["mrr_model"].reconstruct_all(
            cloudy_optical=state["cloudy_n"], hist_optical=state["hist_n"],
            sar_image=state["sar_n"], cloud_mask=occ,
            cloud_prob=cr["cloud_probability"],
            change_prob=state["change_res"]["change_probability"])
        state["candidates"] = cands
        return AgentMessage(self.name, "reconstructed", f"candidates={sorted(cands.keys())}")


class QualityAgent(BaseAgent):
    name = "A8-quality"
    role = "QAN ranking"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        best, allm = state["qan"].rank_candidates(
            candidates=state["candidates"], reference=state["ref_n"], cloud_mask=state["occ_mask"])
        ov = state.get("strategy_override")
        sel = ov if ov in state["candidates"] else best
        state["best"], state["all_metrics"] = sel, allm
        state["recon"] = state["candidates"][sel]
        state["quality"] = allm[sel]
        return AgentMessage(self.name, "ranked", f"best={sel} score={allm[sel].composite_score}",
                            {"best": sel})


class ConfidenceAgent(BaseAgent):
    name = "A9-confidence"
    role = "pixel reliability"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        rep = state["confidence_estimator"].estimate(
            cloud_mask=state["cloud_res"]["cloud_mask"],
            cloud_prob=state["cloud_res"]["cloud_probability"],
            change_prob=state["change_res"]["change_probability"],
            candidates=state["candidates"])
        state["conf"] = rep
        return AgentMessage(self.name, "scored", f"mean_conf={rep.mean_confidence}")


class AnalyticsAgent(BaseAgent):
    name = "A9b-analytics"
    role = "NDVI + landcover + sub-cloud"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        meta = state["meta"]
        state["ndvi"] = state["ndvi_analyzer"].analyze(
            reconstructed_image=state["recon"], reference_image=state["ref_n"])
        state["lc"] = state["landcover_classifier"].classify(state["recon"])
        state["sub"] = state["sub_cloud_predictor"].predict_sub_cloud_features(
            cloud_mask=state["cloud_res"]["cloud_mask"],
            reconstructed_image=state["recon"], sar_image=state["sar_n"],
            pixel_resolution_m=meta.resolution if meta else 10.0)
        lc_method = getattr(state["landcover_classifier"], "method", "rule-based")
        return AgentMessage(self.name, "analyzed",
                            f"ndvi={state['ndvi'].mean_ndvi} lc_method={lc_method}")


class DeliveryAgent(BaseAgent):
    name = "A10-delivery"
    role = "GeoTIFF + PDF + metadata export"

    def act(self, state: Dict[str, Any]) -> AgentMessage:
        import os
        from ..preprocessing.preprocessor import ImagePreprocessor
        import numpy as np, json
        loader, meta = state["loader"], state["meta"]
        iid = state["image_id"]
        cf = os.path.join(state["output_dir"], f"{iid}_cloud_free.tif")
        cf2 = os.path.join(state["output_dir"], f"{iid}_confidence.tif")
        ch = os.path.join(state["output_dir"], f"{iid}_change_map.tif")
        loader.save_raster(cf, state["recon"], reference_meta=meta)
        loader.save_raster(cf2, state["conf"].confidence_map, reference_meta=meta)
        loader.save_raster(ch, state["change_res"]["change_probability"], reference_meta=meta)
        H, W = state["recon"].shape[:2]
        rep = os.path.join(state["reports_dir"], f"{iid}_quality_report.pdf")
        mjson = os.path.join(state["output_dir"], f"{iid}_metadata.json")
        state["pdf_gen"].generate_report(rep, {
            "image_id": iid, "metadata": meta.to_dict() if meta else {},
            "metrics": state["quality"].to_dict(), "decision": state["decision"],
            "cloud_percentage": state["cloud_res"]["cloud_percentage"],
            "confidence_stats": state["conf"].to_dict(),
            "ndvi": state["ndvi"].to_dict(), "landcover": state["lc"].to_dict(),
            "best_candidate": state["best"],
            "cloudy_rgb": ImagePreprocessor.extract_rgb_preview(state["cloudy_n"]),
            "cloud_mask_rgb": np.stack([state["cloud_res"]["cloud_mask"] * 255,
                                        state["cloud_res"]["shadow_mask"] * 128,
                                        np.zeros((H, W), dtype=np.uint8)], axis=-1),
            "reconstructed_rgb": ImagePreprocessor.extract_rgb_preview(state["recon"]),
            "confidence_rgb": state["conf"].colored_heatmap})
        state.update({"cf": cf, "cf2": cf2, "ch": ch, "rep": rep, "mjson": mjson})
        return AgentMessage(self.name, "delivered", f"wrote {cf}, {rep}")
