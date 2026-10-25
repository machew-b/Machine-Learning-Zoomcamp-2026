"""Tests for train.py and the Flask service, using a small model trained on synthetic data.

Run from the project folder:  python -m unittest discover -s tests -v
"""
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import common  # noqa: E402
import synthetic  # noqa: E402
import train  # noqa: E402

SAMPLE = json.loads((ROOT / "sample_request.json").read_text(encoding="utf-8"))

# small, fast hyper-parameters: these tests check the plumbing, not the accuracy
FAST_PARAMS = {
    "ridge": {"alpha": 1.0},
    "decision_tree": {"max_depth": 6, "min_samples_leaf": 10},
    "random_forest": {"n_estimators": 15, "max_depth": 8, "min_samples_leaf": 5, "max_features": 0.8},
    "hist_gradient_boosting": {
        "learning_rate": 0.1, "max_iter": 40, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0,
    },
}

_tmp = None
FOLDER = None
TRAIN_OUTPUT = ""
predict = None
client = None


def setUpModule():
    global _tmp, FOLDER, TRAIN_OUTPUT, predict, client
    _tmp = tempfile.TemporaryDirectory()
    FOLDER = Path(_tmp.name)
    synthetic.write_raw_csv(FOLDER / "SeoulBikeData.csv", days=150)
    common.save_params(FOLDER / "params.json", FAST_PARAMS, {"status": "tuned"})
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = train.main(
            [
                "--data", str(FOLDER / "SeoulBikeData.csv"),
                "--params", str(FOLDER / "params.json"),
                "--model-out", str(FOLDER / "model.bin"),
                "--reports-dir", str(FOLDER / "reports"),
            ]
        )
    TRAIN_OUTPUT = buffer.getvalue()
    assert code == 0, TRAIN_OUTPUT
    os.environ["MODEL_PATH"] = str(FOLDER / "model.bin")
    sys.modules.pop("predict", None)
    predict = importlib.import_module("predict")
    client = predict.app.test_client()


def tearDownModule():
    os.environ.pop("MODEL_PATH", None)
    sys.modules.pop("predict", None)
    _tmp.cleanup()


class TrainOutputTests(unittest.TestCase):
    def test_metrics_file(self):
        metrics = json.loads((FOLDER / "reports" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["seed"], common.RANDOM_STATE)
        self.assertEqual(set(metrics["cv"]), set(common.MODEL_NAMES))
        self.assertIn(metrics["selected_model"], common.MODEL_NAMES)
        best = min(metrics["cv"], key=lambda name: metrics["cv"][name]["rmse_mean"])
        self.assertEqual(metrics["selected_model"], best)  # chosen by CV only
        for key in ("selected_model", "baseline_hour_daytype_mean", "baseline_training_mean"):
            self.assertEqual(set(metrics["test"][key]), {"rmse", "mae", "r2", "wape"})
        data = metrics["data"]
        self.assertEqual(data["train_rows"] + data["test_rows"], data["rows_clean"])
        self.assertEqual(metrics["params_status"], "tuned")
        self.assertIn("refit_on", metrics["final_model"])

    def test_results_markdown(self):
        text = (FOLDER / "reports" / "results.md").read_text(encoding="utf-8")
        self.assertIn("(selected)", text)
        for name in common.MODEL_NAMES:
            self.assertIn(f"| {name}", text)

    def test_bundle_metadata(self):
        metadata = common.load_bundle(FOLDER / "model.bin")["metadata"]
        metrics = json.loads((FOLDER / "reports" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_name"], metrics["selected_model"])
        self.assertEqual(metadata["fit_rows"], metrics["data"]["rows_clean"])
        self.assertEqual(set(metadata["training_ranges"]), set(common.WEATHER_COLUMNS))
        self.assertIn("rmse", metadata["holdout_test"])

    def test_output_has_no_local_paths(self):
        self.assertNotIn(str(FOLDER), TRAIN_OUTPUT)
        self.assertIn("Selected model", TRAIN_OUTPUT)

    def test_missing_data_file_is_reported(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = train.main(["--data", str(FOLDER / "nope.csv")])
        self.assertEqual(code, 1)
        self.assertIn("download_data.py", buffer.getvalue())
        self.assertNotIn(str(FOLDER), buffer.getvalue())

    def test_untuned_params_trigger_a_warning(self):
        common.save_params(FOLDER / "untuned.json", FAST_PARAMS, {"status": "defaults"})
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            train.main(
                [
                    "--data", str(FOLDER / "SeoulBikeData.csv"),
                    "--params", str(FOLDER / "untuned.json"),
                    "--model-out", str(FOLDER / "other.bin"),
                    "--reports-dir", str(FOLDER / "other_reports"),
                ]
            )
        self.assertIn("WARNING", buffer.getvalue())


class ApiTests(unittest.TestCase):
    def post(self, path, payload=None, raw=None):
        if raw is not None:
            return client.post(path, data=raw, content_type="application/json")
        return client.post(path, json=payload)

    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
        self.assertIn(response.get_json()["model"], common.MODEL_NAMES)

    def test_index_describes_the_service(self):
        body = client.get("/").get_json()
        self.assertEqual(body["service"], "seoul-bike-demand")
        self.assertEqual(set(body["input_fields"]), set(common.INPUT_COLUMNS))
        self.assertIn("POST /predict", body["endpoints"])
        self.assertIn("training_ranges", body["model"])

    def test_predict_sample_request(self):
        response = self.post("/predict", SAMPLE)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(set(body), {"predicted_rentals", "unit", "model", "warnings"})
        self.assertGreaterEqual(body["predicted_rentals"], 0)
        self.assertEqual(body["unit"], "bikes per hour")
        self.assertIsInstance(body["warnings"], list)
        self.assertEqual(body["predicted_rentals"], round(body["predicted_rentals"], 1))

    def test_request_inside_the_training_range_has_no_warnings(self):
        ranges = predict.metadata["training_ranges"]
        middle = {name: round((low + high) / 2, 1) for name, (low, high) in ranges.items()}
        low_dew, high_dew = ranges["dew_point_c"]
        middle["dew_point_c"] = round(min(max(low_dew, middle["temperature_c"] - 3), high_dew), 1)
        body = self.post("/predict", dict(SAMPLE, **middle)).get_json()
        self.assertEqual(body["warnings"], [], body)

    def test_prediction_matches_the_saved_pipeline(self):
        record, errors = common.validate_record(dict(SAMPLE))
        self.assertEqual(errors, [])
        expected = common.predict_counts(predict.pipeline, common.records_to_frame([record]))[0]
        self.assertAlmostEqual(self.post("/predict", SAMPLE).get_json()["predicted_rentals"], expected, places=1)

    def test_prediction_is_deterministic(self):
        first = self.post("/predict", SAMPLE).get_json()
        second = self.post("/predict", SAMPLE).get_json()
        self.assertEqual(first, second)

    def test_evening_rush_hour_is_busier_than_the_night(self):
        night = self.post("/predict", dict(SAMPLE, hour=3, solar_radiation_mj_m2=0.0)).get_json()["predicted_rentals"]
        rush = self.post("/predict", dict(SAMPLE, hour=18)).get_json()["predicted_rentals"]
        self.assertGreater(rush, night)

    def test_integer_valued_numbers_are_accepted(self):
        record = dict(SAMPLE, humidity_pct=55, visibility_10m=1800, rainfall_mm=0, hour=18.0)
        self.assertEqual(self.post("/predict", record).status_code, 200)

    def test_invalid_requests_return_400_with_reasons(self):
        bad = {
            "hour out of range": dict(SAMPLE, hour=24),
            "missing field": {k: v for k, v in SAMPLE.items() if k != "temperature_c"},
            "unknown field": dict(SAMPLE, season="Summer"),
            "bad date": dict(SAMPLE, date="2018-02-30"),
            "humidity above 100": dict(SAMPLE, humidity_pct=120),
            "text instead of number": dict(SAMPLE, wind_speed_ms="windy"),
            "text instead of boolean": dict(SAMPLE, holiday="no"),
            "negative snowfall": dict(SAMPLE, snowfall_cm=-2),
        }
        for label, record in bad.items():
            with self.subTest(label):
                response = self.post("/predict", record)
                self.assertEqual(response.status_code, 400)
                body = response.get_json()
                self.assertEqual(body["error"], "invalid input")
                self.assertTrue(body["details"])

    def test_malformed_bodies_return_400(self):
        for label, raw in {"not json": b"hello", "empty": b"", "array": b"[1, 2, 3]", "null": b"null"}.items():
            with self.subTest(label):
                response = self.post("/predict", raw=raw)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_out_of_training_range_input_gets_a_warning(self):
        low, high = predict.metadata["training_ranges"]["temperature_c"]
        hotter = min(high + 5.0, 59.0)
        record = dict(SAMPLE, temperature_c=hotter, dew_point_c=min(hotter - 3, 39.0))
        body = self.post("/predict", record).get_json()
        self.assertTrue(any("temperature_c" in w for w in body["warnings"]), body)

    def test_implausible_dew_point_gets_a_warning(self):
        body = self.post("/predict", dict(SAMPLE, temperature_c=10.0, dew_point_c=20.0)).get_json()
        self.assertTrue(any("dew_point_c" in w for w in body["warnings"]), body)

    def test_batch_prediction(self):
        records = [dict(SAMPLE, hour=h) for h in range(24)]
        response = self.post("/predict_batch", {"records": records})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["count"], 24)
        self.assertEqual([p["index"] for p in body["predictions"]], list(range(24)))
        for hour in (0, 8, 18):
            single = self.post("/predict", records[hour]).get_json()["predicted_rentals"]
            self.assertEqual(body["predictions"][hour]["predicted_rentals"], single)

    def test_batch_reports_the_failing_record(self):
        records = [SAMPLE, dict(SAMPLE, hour=30), dict(SAMPLE, date="nope")]
        response = self.post("/predict_batch", {"records": records})
        self.assertEqual(response.status_code, 400)
        details = " ".join(response.get_json()["details"])
        self.assertIn("record 1", details)
        self.assertIn("record 2", details)
        self.assertNotIn("record 0", details)

    def test_batch_size_and_shape_limits(self):
        cases = {
            "empty list": {"records": []},
            "too many": {"records": [SAMPLE] * (common.MAX_BATCH_SIZE + 1)},
            "not a list": {"records": "x"},
            "wrong key": {"rows": [SAMPLE]},
            "extra key": {"records": [SAMPLE], "note": "hi"},
        }
        for label, payload in cases.items():
            with self.subTest(label):
                self.assertEqual(self.post("/predict_batch", payload).status_code, 400)
        full = {"records": [SAMPLE] * common.MAX_BATCH_SIZE}
        self.assertEqual(self.post("/predict_batch", full).status_code, 200)

    def test_unknown_route_and_wrong_method_return_json(self):
        response = client.get("/nothing")
        self.assertEqual((response.status_code, response.get_json()["error"]), (404, "not found"))
        response = client.get("/predict")
        self.assertEqual(response.status_code, 405)
        self.assertIn("error", response.get_json())
        response = client.post("/health", json={})
        self.assertEqual(response.status_code, 405)

    def test_oversized_body_is_rejected(self):
        response = self.post("/predict", raw=b"x" * 1_100_000)
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
