"""Tests of train.py and download_data.py (offline, synthetic data)."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import download_data  # noqa: E402
import synthetic  # noqa: E402
import train  # noqa: E402

TINY_PARAMS = {
    "_meta": {"status": "tuned"},
    "logistic_regression": {"C": 0.1},
    "decision_tree": {"max_depth": 4, "min_samples_leaf": 20},
    "random_forest": {"n_estimators": 12, "max_depth": 6, "min_samples_leaf": 10, "max_features": "sqrt", "max_samples": 0.5},
    "hist_gradient_boosting": {"learning_rate": 0.1, "max_iter": 12, "max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 1.0},
}


def quiet(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        synthetic.write_csv(root / "diabetic_data.csv", 3000, 31)
        (root / "params.json").write_text(json.dumps(TINY_PARAMS), encoding="utf-8")
        cls.args = ["--data", str(root / "diabetic_data.csv"), "--params", str(root / "params.json"),
                    "--model-out", str(root / "model.bin"), "--reports-dir", str(root / "reports"), "--n-jobs", "1"]
        cls.status, cls.output = quiet(train.main, cls.args)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_and_writes_everything(self):
        self.assertEqual(self.status, 0)
        for name in ("model.bin", "reports/metrics.json", "reports/results.md"):
            self.assertTrue((self.root / name).exists(), name)
        self.assertIn("Selected model", self.output)
        self.assertNotIn("WARNING", self.output)          # params status is 'tuned'
        self.assertNotIn(str(self.root), self.output)      # no local paths in the log

    def test_metrics_file_is_complete_and_consistent(self):
        metrics = json.loads((self.root / "reports" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(set(metrics["cv"]), set(common.MODEL_NAMES))
        selected = metrics["selected_model"]
        best = min(metrics["cv"], key=lambda n: metrics["cv"][n]["log_loss_mean"])
        self.assertEqual(selected, best)
        data = metrics["data"]
        self.assertEqual(data["train_rows"] + data["test_rows"], data["rows_clean"])
        self.assertAlmostEqual(sum(data["class_share_train"]), 1.0)
        test = metrics["test"]["selected_model"]
        for key in ("log_loss", "roc_auc_macro_ovr", "roc_auc_early", "pr_auc_early", "macro_f1", "confusion_matrix"):
            self.assertIn(key, test)
        self.assertEqual(sum(sum(row) for row in test["confusion_matrix"]), data["test_rows"])
        self.assertLess(test["log_loss"], metrics["test"]["baseline_class_frequencies"]["log_loss"] + 0.05)
        self.assertEqual(metrics["cv"][selected]["decision_weights"], test["weights"])
        self.assertEqual(metrics["params_status"], "tuned")

    def test_results_markdown_mentions_every_model(self):
        text = (self.root / "reports" / "results.md").read_text(encoding="utf-8")
        for name in common.MODEL_NAMES:
            self.assertIn(name, text)
        self.assertIn("Confusion matrix", text)
        self.assertNotIn("\r", text)

    def test_saved_bundle_is_ready_for_serving(self):
        bundle = common.load_bundle(self.root / "model.bin")
        meta = bundle["metadata"]
        self.assertEqual(meta["classes"], common.CLASS_LABELS)
        self.assertEqual(len(meta["decision_weights"]), 3)
        self.assertAlmostEqual(sum(meta["base_rates"]), 1.0)
        self.assertIn("payer_code", meta["training_categories"])
        model = bundle["pipeline"].named_steps["model"]
        if hasattr(model, "n_jobs"):
            self.assertEqual(model.n_jobs, 1)
        raw, _ = common.clean_with_report(common.load_raw(self.root / "diabetic_data.csv"))
        proba = common.predict_proba(bundle["pipeline"], raw[common.INPUT_COLUMNS].iloc[:10])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_training_is_reproducible(self):
        again = Path(self.tmp.name) / "again"
        args = list(self.args)
        args[args.index("--model-out") + 1] = str(again / "model.bin")
        args[args.index("--reports-dir") + 1] = str(again / "reports")
        again.mkdir()
        quiet(train.main, args)
        first = json.loads((self.root / "reports" / "metrics.json").read_text(encoding="utf-8"))
        second = json.loads((again / "reports" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(first["test"], second["test"])
        self.assertEqual(first["selected_model"], second["selected_model"])

    def test_missing_data_file_gives_a_clear_message(self):
        status, output = quiet(train.main, ["--data", str(self.root / "nope.csv"), "--n-jobs", "1"])
        self.assertEqual(status, 1)
        self.assertIn("download_data.py", output)

    def test_untuned_parameters_trigger_a_warning(self):
        params = dict(TINY_PARAMS, _meta={"status": "defaults"})
        path = self.root / "params_default.json"
        path.write_text(json.dumps(params), encoding="utf-8")
        args = list(self.args)
        args[args.index("--params") + 1] = str(path)
        args[args.index("--model-out") + 1] = str(self.root / "model2.bin")
        args[args.index("--reports-dir") + 1] = str(self.root / "reports2")
        _, output = quiet(train.main, args)
        self.assertIn("WARNING", output)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_text = synthetic.to_csv_text(synthetic.make_raw_frame(400, 41))
        cls.zip_bytes = synthetic.make_zip_bytes(cls.csv_text)

    def test_csv_is_found_inside_the_zip_next_to_other_files(self):
        self.assertEqual(download_data.csv_from_zip(self.zip_bytes).decode("utf-8"), self.csv_text)

    def test_zip_without_the_data_file_is_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("other.csv", "a,b\n1,2\n")
        with self.assertRaises(ValueError) as ctx:
            download_data.csv_from_zip(buffer.getvalue())
        self.assertIn("other.csv", str(ctx.exception))

    def test_nested_folder_inside_the_zip_is_fine(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("dataset_diabetes/diabetic_data.csv", self.csv_text)
        self.assertEqual(download_data.csv_from_zip(buffer.getvalue()).decode("utf-8"), self.csv_text)

    def test_main_with_zip_csv_and_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "d.zip").write_bytes(self.zip_bytes)
            status, output = quiet(download_data.main, ["--dest-dir", str(root / "data"), "--from-zip", str(root / "d.zip")])
            self.assertEqual(status, 0)
            self.assertTrue((root / "data" / common.CSV_NAME).exists())
            self.assertIn("400 rows", output)
            self.assertIn("CC BY 4.0", output)
            self.assertNotIn(tmp, output)
            status, output = quiet(download_data.main, ["--dest-dir", str(root / "data")])
            self.assertIn("already exists", output)
            (root / "x.csv").write_text(self.csv_text, encoding="latin-1", newline="")
            status, _ = quiet(download_data.main, ["--dest-dir", str(root / "data2"), "--from-csv", str(root / "x.csv")])
            self.assertEqual(status, 0)

    def test_download_path_uses_fetch(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(download_data, "fetch", return_value=self.zip_bytes) as fetch:
            status, output = quiet(download_data.main, ["--dest-dir", tmp])
            self.assertEqual(status, 0)
            fetch.assert_called_once_with(common.DATASET_URL)
            self.assertTrue(Path(tmp, common.CSV_NAME).exists())

    def test_failures_are_reported_not_raised(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(download_data, "fetch", side_effect=urllib.error.URLError("offline")):
                status, output = quiet(download_data.main, ["--dest-dir", tmp, "--force"])
            self.assertEqual(status, 1)
            self.assertIn("--from-zip", output)
            Path(tmp, "bad.zip").write_bytes(b"not a zip")
            status, output = quiet(download_data.main, ["--dest-dir", tmp, "--from-zip", str(Path(tmp, "bad.zip"))])
            self.assertEqual(status, 1)
            Path(tmp, "wrong.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            status, output = quiet(download_data.main, ["--dest-dir", tmp, "--from-csv", str(Path(tmp, "wrong.csv"))])
            self.assertEqual(status, 1)
            self.assertIn("Missing column", output)


if __name__ == "__main__":
    unittest.main()
