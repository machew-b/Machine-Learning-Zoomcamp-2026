"""Unit tests for common.py (no network, synthetic data in the real file format)."""
from __future__ import annotations

import copy
import io
import json
import math
import sys
import tempfile
import unittest
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
import synthetic  # noqa: E402

SAMPLE = json.loads((HERE.parent / "sample_request.json").read_text(encoding="utf-8"))
SMALL_PARAMS = {
    "logistic_regression": {"C": 0.1},
    "decision_tree": {"max_depth": 4, "min_samples_leaf": 20},
    "random_forest": {"n_estimators": 15, "max_depth": 6, "min_samples_leaf": 10, "max_features": "sqrt", "max_samples": 0.5},
    "hist_gradient_boosting": {"learning_rate": 0.1, "max_iter": 15, "max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 1.0},
}


@lru_cache(maxsize=4)
def clean_frame(n_rows: int = 2500, seed: int = 0):
    raw = common.standardise_raw(synthetic.make_raw_frame(n_rows, seed))
    data, report = common.clean_with_report(raw)
    return data, report


class HeaderAndLoadingTests(unittest.TestCase):
    def test_normalise_header(self):
        self.assertEqual(common.normalise_header("glyburide-metformin"), "glyburide_metformin")
        self.assertEqual(common.normalise_header("\ufeffencounter_id"), "encounter_id")
        self.assertEqual(common.normalise_header("A1Cresult"), "a1cresult")

    def test_raw_schema_has_50_unique_columns(self):
        self.assertEqual(len(common.RAW_COLUMNS), 50)
        self.assertEqual(len(set(common.RAW_COLUMNS)), 50)
        self.assertEqual(len(common.MED_COLUMNS), 23)
        self.assertEqual(len(common.INPUT_COLUMNS), 44)
        for column in common.AUDIT_COLUMNS + ["weight", "encounter_id", "patient_nbr", "readmitted"]:
            self.assertNotIn(column, common.INPUT_COLUMNS)

    def test_load_raw_through_csv_keeps_none_and_no_as_values(self):
        frame = synthetic.make_raw_frame(600, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diabetic_data.csv"
            path.write_text(synthetic.to_csv_text(frame), encoding="latin-1", newline="")
            raw = common.load_raw(path)
        self.assertEqual(list(raw.columns), common.RAW_COLUMNS)
        self.assertIn("None", set(raw["max_glu_serum"].dropna()))        # not turned into a missing value
        self.assertFalse(raw["max_glu_serum"].isna().any())
        self.assertFalse(raw["readmitted"].isna().any())                  # 'NO' is a class, not a missing value
        self.assertEqual(set(raw["readmitted"]), set(common.CLASS_LABELS))
        self.assertTrue(raw["payer_code"].isna().any())                   # '?' became missing
        self.assertNotIn("?", set(raw["race"].dropna()))

    def test_column_order_and_underscore_headers_are_accepted(self):
        frame = synthetic.make_raw_frame(300, 2)
        shuffled = frame[list(reversed(frame.columns))].rename(columns=lambda c: c.replace("-", "_"))
        raw = common.standardise_raw(shuffled)
        self.assertEqual(list(raw.columns), common.RAW_COLUMNS)

    def test_missing_column_is_reported(self):
        frame = synthetic.make_raw_frame(100, 3).drop(columns=["number_inpatient"])
        with self.assertRaises(ValueError) as ctx:
            common.standardise_raw(frame)
        self.assertIn("number_inpatient", str(ctx.exception))

    def test_unknown_target_value_is_rejected(self):
        frame = synthetic.make_raw_frame(100, 3)
        frame.loc[0, "readmitted"] = "MAYBE"
        with self.assertRaises(ValueError):
            common.standardise_raw(frame)

    def test_value_spelling_is_normalised(self):
        frame = synthetic.make_raw_frame(100, 4)
        frame["insulin"] = frame["insulin"].str.lower()
        frame["readmitted"] = frame["readmitted"].str.lower()
        raw = common.standardise_raw(frame)
        self.assertTrue(set(raw["insulin"]) <= set(common.MED_STATUS))
        self.assertTrue(set(raw["readmitted"]) <= set(common.CLASS_LABELS))


class CleaningTests(unittest.TestCase):
    def test_report_adds_up(self):
        raw = common.standardise_raw(synthetic.make_raw_frame(2000, 5))
        data, report = common.clean_with_report(raw)
        dropped = report["duplicate_rows_dropped"] + report["duplicate_encounter_ids_dropped"] + \
            report["expired_or_hospice_dropped"] + report["rows_dropped_missing_key_fields"]
        self.assertEqual(report["rows_raw"] - dropped, report["rows_clean"])
        self.assertEqual(report["rows_clean"], len(data))
        self.assertGreater(report["expired_or_hospice_dropped"], 0)
        self.assertEqual(sum(report["class_counts"].values()), len(data))

    def test_expired_and_hospice_are_removed(self):
        data, _ = clean_frame()
        self.assertFalse(data["discharge_disposition_id"].isin(common.EXCLUDED_DISCHARGE_IDS).any())

    def test_duplicates_are_removed(self):
        raw = common.standardise_raw(synthetic.make_raw_frame(500, 6))
        doubled = pd.concat([raw, raw.iloc[:20]], ignore_index=True)
        _, report = common.clean_with_report(doubled)
        self.assertEqual(report["duplicate_rows_dropped"], 20)
        again = raw.copy()
        again.loc[3, "encounter_id"] = again.loc[2, "encounter_id"]
        _, report = common.clean_with_report(again)
        self.assertEqual(report["duplicate_encounter_ids_dropped"], 1)

    def test_target_is_integer_class_id(self):
        data, _ = clean_frame()
        self.assertTrue(pd.api.types.is_integer_dtype(data[common.TARGET]))
        for i, label in enumerate(common.CLASS_LABELS):
            self.assertTrue((data.loc[data[common.TARGET] == i, "readmitted"] == label).all())
        for column in common.INT_RAW_COLUMNS:
            self.assertTrue(pd.api.types.is_integer_dtype(data[column]), column)

    def test_everything_dropped_raises(self):
        raw = common.standardise_raw(synthetic.make_raw_frame(50, 7))
        raw["discharge_disposition_id"] = 11
        with self.assertRaises(ValueError):
            common.clean_with_report(raw)


class SplitTests(unittest.TestCase):
    def test_hold_out_split_never_splits_a_patient(self):
        data, _ = clean_frame()
        y, g = data[common.TARGET], data[common.GROUP_COLUMN].to_numpy()
        train, test = common.grouped_stratified_split(y, g)
        self.assertFalse(set(g[train]) & set(g[test]))
        self.assertTrue(np.all(train ^ test))
        self.assertAlmostEqual(test.mean(), 0.2, delta=0.05)
        shares_train = np.bincount(y[train], minlength=3) / train.sum()
        shares_test = np.bincount(y[test], minlength=3) / test.sum()
        np.testing.assert_allclose(shares_train, shares_test, atol=0.03)
        again = common.grouped_stratified_split(y, g)
        np.testing.assert_array_equal(test, again[1])

    def test_cv_folds_never_split_a_patient(self):
        data, _ = clean_frame()
        y, g = data[common.TARGET], data[common.GROUP_COLUMN].to_numpy()
        folds = list(common.grouped_cv_splitter().split(np.zeros(len(y)), y, g))
        self.assertEqual(len(folds), common.N_SPLITS)
        covered = np.concatenate([valid for _, valid in folds])
        self.assertEqual(sorted(covered.tolist()), list(range(len(y))))
        for train_idx, valid_idx in folds:
            self.assertFalse(set(g[train_idx]) & set(g[valid_idx]))

    def test_sample_rows_by_group_keeps_whole_groups(self):
        data, _ = clean_frame()
        g = data[common.GROUP_COLUMN].to_numpy()
        mask = common.sample_rows_by_group(g, 800, seed=1)
        chosen = set(g[mask])
        self.assertFalse(chosen & set(g[~mask]))
        self.assertGreaterEqual(mask.sum(), 800)
        self.assertLess(mask.sum(), 900)
        np.testing.assert_array_equal(mask, common.sample_rows_by_group(g, 800, seed=1))
        self.assertTrue(common.sample_rows_by_group(g, 10**9).all())


class FeatureTests(unittest.TestCase):
    def test_icd9_groups(self):
        cases = {
            "250.83": "Diabetes", "250": "Diabetes", "428": "Circulatory", "414": "Circulatory", "785": "Circulatory",
            "486": "Respiratory", "786": "Respiratory", "578": "Digestive", "787": "Digestive", "820": "Injury",
            "715": "Musculoskeletal", "599": "Genitourinary", "788": "Genitourinary", "162": "Neoplasms",
            "276": "Other", "038": "Other", "V57": "Other", "E909": "Other", "??": "Other", "": "Unknown",
            "?": "Unknown", None: "Unknown", float("nan"): "Unknown",
        }
        for code, expected in cases.items():
            self.assertEqual(common.icd9_group(code), expected, code)

    def test_feature_frame_shape_and_content(self):
        data, _ = clean_frame()
        X = data[common.INPUT_COLUMNS]
        features = common.build_features(X)
        self.assertEqual(list(features.columns), common.FEATURE_COLUMNS)
        self.assertEqual(len(features), len(X))
        self.assertFalse(features[common.CATEGORICAL_FEATURES].isna().any().any())
        self.assertNotIn("race", features.columns)
        self.assertNotIn("gender", features.columns)
        self.assertTrue(features["age_mid"].between(5, 95).all())
        self.assertTrue((features["n_prior_visits"] ==
                         X["number_outpatient"] + X["number_emergency"] + X["number_inpatient"]).all())
        self.assertTrue(set(features["diag_1_group"]) <= {"Diabetes", "Circulatory", "Respiratory", "Digestive",
                                                          "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms",
                                                          "Other", "Unknown"})

    def test_medication_counts(self):
        data, _ = clean_frame()
        row = data[common.INPUT_COLUMNS].iloc[[0]].copy()
        for drug in common.MED_COLUMNS:
            row[drug] = "No"
        row["insulin"], row["glipizide"], row["metformin"] = "Up", "Steady", "Down"
        features = common.build_features(row)
        self.assertEqual(features["n_meds_active"].iloc[0], 3)
        self.assertEqual(features["n_meds_changed"].iloc[0], 2)
        self.assertEqual(features["insulin"].iloc[0], "Up")

    def test_id_levels_are_merged(self):
        data, _ = clean_frame()
        row = data[common.INPUT_COLUMNS].iloc[[0]].copy()
        row["admission_type_id"], row["admission_source_id"], row["discharge_disposition_id"] = 6, 17, 25
        features = common.build_features(row)
        self.assertEqual(features["admission_type"].iloc[0], "unknown")
        self.assertEqual(features["admission_source"].iloc[0], "unknown")
        self.assertEqual(features["discharge_disposition"].iloc[0], "unknown")
        row["admission_type_id"] = 1
        self.assertEqual(common.build_features(row)["admission_type"].iloc[0], "1")

    def test_missing_values_are_handled(self):
        data, _ = clean_frame()
        row = data[common.INPUT_COLUMNS].iloc[[0]].copy().astype(object)
        for column in ("payer_code", "medical_specialty", "diag_2", "diag_3", "age"):
            row[column] = None
        features = common.build_features(row)
        self.assertEqual(features["payer_code"].iloc[0], "Unknown")
        self.assertEqual(features["diag_3_group"].iloc[0], "Unknown")
        self.assertTrue(math.isnan(features["age_mid"].iloc[0]))

    def test_features_do_not_depend_on_other_rows(self):
        data, _ = clean_frame()
        X = data[common.INPUT_COLUMNS]
        pd.testing.assert_frame_equal(common.build_features(X).iloc[10:20], common.build_features(X.iloc[10:20]))

    def test_missing_input_column_raises(self):
        data, _ = clean_frame()
        with self.assertRaises(ValueError):
            common.build_features(data[common.INPUT_COLUMNS].drop(columns=["age"]))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, _ = clean_frame()
        cls.X, cls.y = data[common.INPUT_COLUMNS], data[common.TARGET]
        cls.groups = data[common.GROUP_COLUMN].to_numpy()
        cls.train, cls.test = common.grouped_stratified_split(cls.y, cls.groups)

    def test_every_model_fits_and_predicts_probabilities(self):
        for name in common.MODEL_NAMES:
            with self.subTest(model=name):
                pipeline = common.build_pipeline(name, SMALL_PARAMS[name]).fit(self.X[self.train], self.y[self.train])
                proba = common.predict_proba(pipeline, self.X[self.test])
                self.assertEqual(proba.shape, (self.test.sum(), 3))
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
                self.assertTrue((proba >= 0).all())

    def test_unseen_category_at_prediction_time_is_handled(self):
        pipeline = common.build_pipeline("hist_gradient_boosting", SMALL_PARAMS["hist_gradient_boosting"]).fit(
            self.X[self.train], self.y[self.train])
        odd = self.X[self.test].iloc[:5].copy()
        odd["payer_code"], odd["medical_specialty"], odd["diag_1"] = "ZZ", "NeverSeen", "999.99"
        proba = common.predict_proba(pipeline, odd)
        self.assertTrue(np.isfinite(proba).all())

    def test_unknown_model_name_raises(self):
        with self.assertRaises(ValueError):
            common.build_pipeline("xgboost")

    def test_serving_mode_makes_the_forest_single_threaded(self):
        pipeline = common.build_pipeline("random_forest", SMALL_PARAMS["random_forest"], n_jobs=2)
        self.assertEqual(pipeline.named_steps["model"].n_jobs, 2)
        common.make_serving_ready(pipeline)
        self.assertEqual(pipeline.named_steps["model"].n_jobs, 1)
        common.make_serving_ready(common.build_pipeline("hist_gradient_boosting"))   # no n_jobs: must not fail

    def test_n_jobs_does_not_change_the_result(self):
        a = common.build_pipeline("random_forest", SMALL_PARAMS["random_forest"], n_jobs=1).fit(self.X[self.train], self.y[self.train])
        b = common.build_pipeline("random_forest", SMALL_PARAMS["random_forest"], n_jobs=2).fit(self.X[self.train], self.y[self.train])
        np.testing.assert_allclose(common.predict_proba(a, self.X[self.test]), common.predict_proba(b, self.X[self.test]))

    def test_models_learn_something(self):
        pipeline = common.build_pipeline("logistic_regression", SMALL_PARAMS["logistic_regression"]).fit(
            self.X[self.train], self.y[self.train])
        metrics = common.probability_metrics(self.y[self.test], common.predict_proba(pipeline, self.X[self.test]))
        reference = common.probability_metrics(self.y[self.test], common.prior_probabilities(self.y[self.train], self.test.sum()))
        self.assertGreater(metrics["roc_auc_early"], 0.55)
        self.assertLess(metrics["log_loss"], reference["log_loss"])

    def test_cross_validation_returns_complete_out_of_fold_predictions(self):
        pipeline = common.build_pipeline("decision_tree", SMALL_PARAMS["decision_tree"])
        X, y, g = self.X[self.train], self.y[self.train], self.groups[self.train]
        result = common.cross_validate_model(pipeline, X, y, g)
        self.assertEqual(len(result["folds"]), common.N_SPLITS)
        self.assertFalse(np.isnan(result["oof_proba"]).any())
        np.testing.assert_allclose(result["oof_proba"].sum(axis=1), 1.0, atol=1e-6)
        self.assertIn("log_loss_mean", result["summary"])

    def test_cross_validation_accepts_another_splitter(self):
        from sklearn.model_selection import StratifiedKFold
        pipeline = common.build_pipeline("decision_tree", SMALL_PARAMS["decision_tree"])
        X, y, g = self.X[self.train], self.y[self.train], self.groups[self.train]
        splitter = StratifiedKFold(3, shuffle=True, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)    # "groups is ignored by StratifiedKFold" is the point
            result = common.cross_validate_model(pipeline, X, y, g, splitter=splitter)
        self.assertEqual(len(result["folds"]), 3)

    def test_permutation_importance_finds_the_informative_inputs(self):
        pipeline = common.build_pipeline("logistic_regression", SMALL_PARAMS["logistic_regression"]).fit(
            self.X[self.train], self.y[self.train])
        result = common.grouped_permutation_importance(pipeline, self.X[self.test], self.y[self.test], n_repeats=2)
        self.assertEqual(set(result["variable"]), set(common.PERMUTATION_GROUPS))
        self.assertIn("number_inpatient", result["variable"].head(5).tolist())
        self.assertTrue(result["log_loss_increase"].is_monotonic_decreasing)


class MetricTests(unittest.TestCase):
    def test_decide_uses_weights(self):
        proba = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        self.assertEqual(common.decide(proba, [1, 1, 1]).tolist(), [0, 0])
        self.assertEqual(common.decide(proba, [1, 1, 3]).tolist(), [2, 2])
        self.assertEqual(common.decide(proba, [1, 2, 1]).tolist(), [1, 1])

    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2, 0])
        proba = np.eye(3)[y] * 0.97 + 0.01
        metrics = common.classification_metrics(y, proba, [1, 1, 1])
        self.assertAlmostEqual(metrics["roc_auc_macro_ovr"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[3, 0, 0], [0, 2, 0], [0, 0, 2]])

    def test_class_frequency_reference_carries_no_information(self):
        y = np.array([0] * 60 + [1] * 30 + [2] * 10)
        metrics = common.classification_metrics(y, common.prior_probabilities(y, len(y)), [1, 1, 1])
        self.assertAlmostEqual(metrics["roc_auc_macro_ovr"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["per_class"]["<30"]["recall"], 0.0)
        entropy = -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1))
        self.assertAlmostEqual(metrics["log_loss"], entropy, places=6)

    def test_tuned_weights_do_not_hurt_macro_f1(self):
        rng = np.random.default_rng(0)
        y = rng.choice(3, size=3000, p=[0.55, 0.35, 0.10])
        logits = rng.normal(size=(3000, 3)) + 0.8 * np.eye(3)[y]
        proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        weights, tuned = common.tune_decision_weights(y, proba)
        plain = common.decision_metrics(y, proba, [1, 1, 1])["macro_f1"]
        self.assertGreaterEqual(tuned, plain)
        self.assertEqual(weights[0], 1.0)
        self.assertAlmostEqual(common.decision_metrics(y, proba, weights)["macro_f1"], tuned, places=9)

    def test_rare_class_needs_a_weight_to_be_predicted(self):
        y = np.array([0] * 80 + [2] * 20)
        proba = np.array([[0.80, 0.15, 0.05]] * 80 + [[0.45, 0.20, 0.35]] * 20)
        self.assertEqual(common.decision_metrics(y, proba, [1, 1, 1])["per_class"]["<30"]["recall"], 0.0)
        weights, tuned = common.tune_decision_weights(y, proba)
        self.assertGreater(weights[2], 1.0)
        self.assertGreater(common.decision_metrics(y, proba, weights)["per_class"]["<30"]["recall"], 0.9)

    def test_summarise_folds(self):
        folds = [{m: float(i) for m in common.FOLD_METRICS} for i in (1, 2, 3)]
        summary = common.summarise_folds(folds)
        self.assertAlmostEqual(summary["log_loss_mean"], 2.0)
        self.assertAlmostEqual(summary["log_loss_std"], math.sqrt(2 / 3))

    def test_predict_proba_reorders_classes(self):
        class Stub:
            classes_ = np.array([2, 0])

            @staticmethod
            def predict_proba(X):
                return np.array([[0.7, 0.3]])

        proba = common.predict_proba(Stub, None)
        np.testing.assert_allclose(proba, [[0.3, 0.0, 0.7]])


class ParamsAndBundleTests(unittest.TestCase):
    def test_defaults_when_file_is_missing(self):
        params, meta = common.load_params(Path(tempfile.gettempdir()) / "does-not-exist-params.json")
        self.assertEqual(meta["status"], "defaults")
        self.assertEqual(set(params), set(common.MODEL_NAMES))

    def test_round_trip_and_unknown_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.json"
            params = copy.deepcopy(common.DEFAULT_PARAMS)
            params["random_forest"]["max_depth"] = np.int64(9)
            common.save_params(path, params, {"status": "tuned"})
            loaded, meta = common.load_params(path)
            self.assertEqual(meta["status"], "tuned")
            self.assertEqual(loaded["random_forest"]["max_depth"], 9)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["svm"] = {}
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                common.load_params(path)

    def test_committed_params_file_is_valid(self):
        params, meta = common.load_params(HERE.parent / "params.json")
        self.assertIn(meta.get("status"), ("defaults", "tuned"))
        for name in common.MODEL_NAMES:
            common.build_pipeline(name, params[name])

    def test_plain_params(self):
        self.assertEqual(common.plain_params({"model__C": np.float64(0.5), "model__max_depth": np.int64(3)}),
                         {"C": 0.5, "max_depth": 3})

    def test_bundle_round_trip_and_errors(self):
        data, _ = clean_frame()
        X, y = data[common.INPUT_COLUMNS], data[common.TARGET]
        pipeline = common.build_pipeline("decision_tree", SMALL_PARAMS["decision_tree"]).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            common.save_bundle(path, pipeline, {"model_name": "decision_tree"})
            bundle = common.load_bundle(path)
            self.assertEqual(bundle["metadata"]["model_name"], "decision_tree")
            np.testing.assert_allclose(common.predict_proba(bundle["pipeline"], X.iloc[:5]),
                                       common.predict_proba(pipeline, X.iloc[:5]))
            with self.assertRaises(FileNotFoundError):
                common.load_bundle(Path(tmp) / "missing.bin")
            import pickle
            (Path(tmp) / "bad.bin").write_bytes(pickle.dumps([1, 2, 3]))
            with self.assertRaises(ValueError):
                common.load_bundle(Path(tmp) / "bad.bin")


class ValidationTests(unittest.TestCase):
    def ok(self, **changes):
        record = {**copy.deepcopy(SAMPLE), **changes}
        clean, errors = common.validate_record(record)
        self.assertEqual(errors, [], errors)
        return clean

    def bad(self, **changes):
        record = {**copy.deepcopy(SAMPLE), **changes}
        clean, errors = common.validate_record(record)
        self.assertIsNone(clean)
        self.assertTrue(errors)
        return errors

    def test_sample_request_is_valid_and_complete(self):
        clean = self.ok()
        self.assertEqual(set(clean), set(common.INPUT_SPEC))
        self.assertEqual(set(clean["medications"]), set(common.MED_COLUMNS))
        self.assertEqual(clean["medications"]["insulin"], "Up")
        self.assertEqual(clean["medications"]["repaglinide"], "No")

    def test_optional_fields_default(self):
        record = {k: v for k, v in SAMPLE.items() if k not in
                  ("payer_code", "medical_specialty", "diag_2", "diag_3", "medications", "max_glu_serum", "A1Cresult")}
        clean, errors = common.validate_record(record)
        self.assertEqual(errors, [])
        self.assertIsNone(clean["payer_code"])
        self.assertIsNone(clean["diag_2"])
        self.assertEqual(clean["A1Cresult"], "None")
        self.assertEqual(set(clean["medications"].values()), {"No"})
        clean, errors = common.validate_record({**SAMPLE, "payer_code": None, "medications": None})
        self.assertEqual(errors, [])

    def test_required_field_missing(self):
        for field in ("age", "diag_1", "time_in_hospital", "change", "diabetesMed", "number_inpatient"):
            record = {k: v for k, v in SAMPLE.items() if k != field}
            _, errors = common.validate_record(record)
            self.assertTrue(any(field in e for e in errors), field)

    def test_unknown_field_and_wrong_container(self):
        self.assertTrue(any("race" in e for e in self.bad(race="Other")))
        self.assertTrue(any("gender" in e for e in self.bad(gender="Female")))
        self.assertEqual(common.validate_record([1, 2])[1], ["a record must be a JSON object"])

    def test_integer_fields(self):
        self.assertEqual(self.ok(time_in_hospital=5.0)["time_in_hospital"], 5)
        for value in (5.5, "5", True, None, float("nan"), float("inf"), 0, 61):
            with self.subTest(value=value):
                record = {**SAMPLE, "time_in_hospital": value}
                self.assertTrue(common.validate_record(record)[1])
        self.bad(number_inpatient=-1)
        self.bad(num_procedures=31)

    def test_discharge_disposition_rules(self):
        for ids in common.EXCLUDED_DISCHARGE_IDS:
            self.assertTrue(any("outside the scope" in e for e in self.bad(discharge_disposition_id=ids)))
        self.ok(discharge_disposition_id=22)
        self.bad(discharge_disposition_id=31)

    def test_choice_fields(self):
        self.bad(age="70-80")
        self.bad(age="[70-80]")
        self.bad(A1Cresult="high")
        self.bad(change="Yes")
        self.bad(diabetesMed=True)
        self.assertEqual(self.ok(max_glu_serum=">300")["max_glu_serum"], ">300")

    def test_icd9_codes(self):
        self.assertEqual(self.ok(diag_1="v57")["diag_1"], "V57")
        for code in ("250.83", "428", "E909", "V45.81", "038"):
            self.ok(diag_1=code)
        for code in ("25O.83", "4288", "", "ABC", "250.", 428, "E9090"):
            self.bad(diag_1=code)

    def test_payer_and_specialty(self):
        self.ok(payer_code="SP")
        self.bad(payer_code="mc")
        self.bad(payer_code="MCX")
        self.ok(medical_specialty="Surgery-General")
        self.ok(medical_specialty="Family/GeneralPractice")
        self.bad(medical_specialty="<script>")
        self.bad(medical_specialty="x" * 61)

    def test_medications(self):
        self.assertEqual(self.ok(medications={"glyburide_metformin": "Down"})["medications"]["glyburide_metformin"], "Down")
        self.assertTrue(any("aspirin" in e for e in self.bad(medications={"aspirin": "Steady"})))
        self.assertTrue(any("insulin" in e for e in self.bad(medications={"insulin": "Increased"})))
        self.assertTrue(any("object" in e for e in self.bad(medications=["insulin"])))
        self.bad(medications={"glyburide-metformin": "Up"})      # API names use underscores

    def test_records_to_frame_matches_the_training_path(self):
        """A raw training row, sent through validation and records_to_frame, gives identical features."""
        data, _ = clean_frame(2500, 11)
        checked = 0
        for _, row in data.iterrows():
            record = {column: (None if pd.isna(row[column]) else row[column]) for column in
                      common.COUNT_INPUTS + common.ID_INPUTS + common.TEXT_INPUTS}
            for column in common.COUNT_INPUTS + common.ID_INPUTS:
                record[column] = int(record[column])
            record["medications"] = {drug: row[drug] for drug in common.MED_COLUMNS}
            clean, errors = common.validate_record({k: v for k, v in record.items() if v is not None})
            if errors:    # e.g. a count above the generous API limits: not the case for synthetic data
                self.fail(errors)
            served = common.build_features(common.records_to_frame([clean]))
            trained = common.build_features(data[common.INPUT_COLUMNS].loc[[row.name]])
            pd.testing.assert_frame_equal(served.reset_index(drop=True), trained.reset_index(drop=True), check_dtype=False)
            checked += 1
            if checked >= 60:
                break
        self.assertEqual(checked, 60)

    def test_input_warnings(self):
        ranges = {"time_in_hospital": [1, 14], "number_inpatient": [0, 21]}
        categories = {"payer_code": ["MC", "SP"], "medical_specialty": ["InternalMedicine"]}
        clean = self.ok()
        self.assertEqual(common.input_warnings(clean, ranges, categories), [])
        clean = self.ok(time_in_hospital=30, payer_code="ZZ", medical_specialty="Dermatology")
        messages = common.input_warnings(clean, ranges, categories)
        self.assertEqual(len(messages), 3)
        self.assertTrue(any("time_in_hospital" in m for m in messages))
        self.assertTrue(any("payer_code" in m for m in messages))
        clean = self.ok(num_medications=0, num_procedures=2)
        self.assertTrue(any("num_medications" in m for m in common.input_warnings(clean, ranges, categories)))


if __name__ == "__main__":
    unittest.main()
