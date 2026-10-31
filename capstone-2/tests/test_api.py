"""Tests of the Flask service (flask test client, tiny model trained on synthetic data)."""
from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import synthetic  # noqa: E402

SAMPLE = json.loads((HERE.parent / "sample_request.json").read_text(encoding="utf-8"))
_state = {}


def setUpModule():
    raw = common.standardise_raw(synthetic.make_raw_frame(2500, 21))
    data, _ = common.clean_with_report(raw)
    X, y = data[common.INPUT_COLUMNS], data[common.TARGET]
    pipeline = common.build_pipeline(
        "random_forest",
        {"n_estimators": 15, "max_depth": 6, "min_samples_leaf": 10, "max_features": "sqrt", "max_samples": 0.5},
    ).fit(X, y)
    common.make_serving_ready(pipeline)
    metadata = {
        "model_name": "random_forest",
        "classes": common.CLASS_LABELS,
        "decision_weights": [1.0, 1.3, 3.0],
        "base_rates": [float(v) for v in np.bincount(y, minlength=3) / len(y)],
        "training_ranges": common.training_ranges(X),
        "training_categories": common.training_categories(X),
        "fit_rows": int(len(X)),
        "params_status": "defaults",
    }
    _state["tmp"] = tempfile.TemporaryDirectory()
    path = Path(_state["tmp"].name) / "model.bin"
    common.save_bundle(path, pipeline, common.to_builtin(metadata))
    os.environ["MODEL_PATH"] = str(path)
    sys.modules.pop("predict", None)
    _state["predict"] = importlib.import_module("predict")
    _state["client"] = _state["predict"].app.test_client()


def tearDownModule():
    os.environ.pop("MODEL_PATH", None)
    sys.modules.pop("predict", None)
    _state["tmp"].cleanup()


class ServiceTests(unittest.TestCase):
    @property
    def client(self):
        return _state["client"]

    def post(self, record, path="/predict"):
        return self.client.post(path, json=record)

    def test_health_and_index(self):
        response = self.client.get("/health")
        self.assertEqual((response.status_code, response.get_json()["status"]), (200, "ok"))
        body = self.client.get("/").get_json()
        self.assertEqual(set(body["input_fields"]), set(common.INPUT_SPEC))
        self.assertTrue(body["input_fields"]["diag_1"]["required"])
        self.assertFalse(body["input_fields"]["diag_2"]["required"])
        self.assertIn("disclaimer", body)
        self.assertEqual(body["model"]["model_name"], "random_forest")

    def test_predict_returns_consistent_probabilities(self):
        response = self.post(SAMPLE)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        probabilities = body["probabilities"]
        self.assertEqual(list(probabilities), common.CLASS_KEYS)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=0.001)
        self.assertEqual(body["early_readmission_risk"], probabilities["readmission_within_30_days"])
        self.assertAlmostEqual(body["any_readmission_risk"], 1 - probabilities["no_readmission"], places=3)
        self.assertIn(body["predicted_class"], common.CLASS_KEYS)
        self.assertEqual(body["predicted_label"], common.CLASS_LABELS[common.CLASS_KEYS.index(body["predicted_class"])])
        self.assertGreater(body["early_risk_vs_average"], 0)
        self.assertIn("disclaimer", body)
        self.assertIsInstance(body["warnings"], list)

    def test_decision_uses_the_stored_weights(self):
        body = self.post(SAMPLE).get_json()
        proba = np.array([[body["probabilities"][key] for key in common.CLASS_KEYS]])
        expected = common.CLASS_KEYS[int(common.decide(proba, [1.0, 1.3, 3.0])[0])]
        self.assertEqual(body["predicted_class"], expected)

    def test_optional_fields_can_be_omitted(self):
        record = {k: v for k, v in SAMPLE.items() if k not in
                  ("payer_code", "medical_specialty", "diag_2", "diag_3", "medications", "max_glu_serum", "A1Cresult")}
        self.assertEqual(self.post(record).status_code, 200)

    def test_prediction_is_deterministic(self):
        self.assertEqual(self.post(SAMPLE).get_json(), self.post(SAMPLE).get_json())

    def test_more_prior_inpatient_visits_raise_the_risk(self):
        risks = [self.post(dict(SAMPLE, number_inpatient=n)).get_json()["early_readmission_risk"] for n in (0, 8)]
        self.assertGreater(risks[1], risks[0])

    def test_unusual_values_warn(self):
        body = self.post(dict(SAMPLE, time_in_hospital=40, payer_code="ZZ", medical_specialty="Unseen")).get_json()
        self.assertEqual(len(body["warnings"]), 3)

    def test_batch(self):
        records = [dict(SAMPLE, number_inpatient=n) for n in range(4)]
        response = self.post({"records": records}, "/predict_batch")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["count"], 4)
        single = self.post(records[2]).get_json()
        self.assertEqual(body["predictions"][2]["probabilities"], single["probabilities"])
        self.assertEqual([p["index"] for p in body["predictions"]], [0, 1, 2, 3])

    def test_batch_errors(self):
        self.assertEqual(self.post({"records": []}, "/predict_batch").status_code, 400)
        self.assertEqual(self.post({"records": [SAMPLE] * (common.MAX_BATCH_SIZE + 1)}, "/predict_batch").status_code, 400)
        self.assertEqual(self.post([SAMPLE], "/predict_batch").status_code, 400)
        self.assertEqual(self.post({"records": [SAMPLE], "extra": 1}, "/predict_batch").status_code, 400)
        response = self.post({"records": [SAMPLE, dict(SAMPLE, age="old")]}, "/predict_batch")
        self.assertEqual(response.status_code, 400)
        self.assertTrue(any(d.startswith("record 1:") for d in response.get_json()["details"]))

    def test_invalid_records_are_rejected_with_reasons(self):
        cases = {
            "age": dict(SAMPLE, age="70-80"),
            "discharge_disposition_id": dict(SAMPLE, discharge_disposition_id=11),
            "diag_1": dict(SAMPLE, diag_1="bad"),
            "medications": dict(SAMPLE, medications={"aspirin": "Steady"}),
            "race": dict(SAMPLE, race="Other"),
            "time_in_hospital": dict(SAMPLE, time_in_hospital="five"),
        }
        for field, record in cases.items():
            with self.subTest(field=field):
                response = self.post(record)
                self.assertEqual(response.status_code, 400)
                body = response.get_json()
                self.assertEqual(body["error"], "invalid input")
                self.assertTrue(any(field in detail for detail in body["details"]))

    def test_malformed_requests(self):
        self.assertEqual(self.client.post("/predict", data="not json", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/predict", data="[1, 2]", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/predict", data="", content_type="application/json").status_code, 400)

    def test_error_responses_are_json(self):
        for response in (self.client.get("/nothing"), self.client.get("/predict"),
                         self.client.post("/predict", data="x" * 1_100_000, content_type="application/json")):
            self.assertIn("error", response.get_json())
        self.assertEqual(self.client.get("/nothing").status_code, 404)
        self.assertEqual(self.client.get("/predict").status_code, 405)
        self.assertEqual(self.client.post("/predict", data="x" * 1_100_000, content_type="application/json").status_code, 413)

    def test_request_is_not_modified(self):
        record = copy.deepcopy(SAMPLE)
        self.post(record)
        self.assertEqual(record, SAMPLE)

    def test_missing_model_file_stops_start_up(self):
        previous = os.environ["MODEL_PATH"]
        os.environ["MODEL_PATH"] = str(Path(previous).with_name("absent.bin"))
        saved = sys.modules.pop("predict")
        try:
            with self.assertRaises(FileNotFoundError):
                importlib.import_module("predict")
        finally:
            os.environ["MODEL_PATH"] = previous
            sys.modules["predict"] = saved


if __name__ == "__main__":
    unittest.main()
