"""Unit tests for common.py. Run from the project folder:  python -m unittest discover -s tests -v"""
import datetime
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

import common  # noqa: E402
import synthetic  # noqa: E402


class HeaderAndLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv = Path(cls.tmp.name) / "SeoulBikeData.csv"
        synthetic.write_raw_csv(cls.csv, days=60)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_normalise_header(self):
        cases = {
            "Temperature(\u00b0C)": "temperature_c",
            "Temperature(\u00c2\u00b0C)": "temperature_c",  # degree sign decoded wrongly
            "Humidity(%)": "humidity",
            "Wind speed (m/s)": "wind_speed_m_s",
            "Visibility (10m)": "visibility_10m",
            "Dew point temperature(\u00b0C)": "dew_point_temperature_c",
            "Solar Radiation (MJ/m2)": "solar_radiation_mj_m2",
            "Rainfall(mm)": "rainfall_mm",
            "Snowfall (cm)": "snowfall_cm",
            "Functioning Day": "functioning_day",
            "\u00ef\u00bb\u00bfDate": "date",  # UTF-8 byte-order mark read as latin-1
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(common.normalise_header(raw), expected)

    def test_load_raw_gives_canonical_columns_and_dtypes(self):
        raw = common.load_raw(self.csv)
        self.assertEqual(list(raw.columns), common.RAW_COLUMNS)
        self.assertEqual(len(raw), 60 * 24)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(raw["date"]))
        self.assertEqual(raw["date"].min(), pd.Timestamp("2017-12-01"))
        self.assertEqual(raw["holiday"].dtype, bool)
        self.assertEqual(raw["functioning_day"].dtype, bool)
        self.assertTrue(set(raw["season"]) <= set(common.SEASONS))
        for column in common.NUMERIC_RAW_COLUMNS:
            self.assertTrue(pd.api.types.is_numeric_dtype(raw[column]), column)

    def test_utf8_encoded_degree_sign_is_tolerated(self):
        path = Path(self.tmp.name) / "utf8.csv"
        synthetic.make_raw_frame(days=10).to_csv(path, index=False, encoding="utf-8")
        raw = common.load_raw(path)
        self.assertEqual(list(raw.columns), common.RAW_COLUMNS)

    def test_functional_day_spelling_variant(self):
        frame = synthetic.make_raw_frame(days=10)
        frame = frame.rename(columns={"Functioning Day": "Functional Day"})
        frame["Functional Day"] = frame["Functional Day"].map({"Yes": "Fun", "No": "NoFunc"})
        raw = common.standardise_raw(frame)
        reference = common.standardise_raw(synthetic.make_raw_frame(days=10))
        self.assertTrue((raw["functioning_day"] == reference["functioning_day"]).all())

    def test_missing_column_is_reported(self):
        frame = synthetic.make_raw_frame(days=5).drop(columns=["Rainfall(mm)"])
        with self.assertRaises(ValueError) as ctx:
            common.standardise_raw(frame)
        self.assertIn("rainfall_mm", str(ctx.exception))

    def test_unexpected_category_is_reported(self):
        frame = synthetic.make_raw_frame(days=5)
        frame.loc[0, "Seasons"] = "Monsoon"
        with self.assertRaises(ValueError) as ctx:
            common.standardise_raw(frame)
        self.assertIn("monsoon", str(ctx.exception))
        frame = synthetic.make_raw_frame(days=5)
        frame.loc[0, "Holiday"] = "Maybe"
        with self.assertRaises(ValueError):
            common.standardise_raw(frame)

    def test_wrong_date_format_is_reported(self):
        frame = synthetic.make_raw_frame(days=5)
        frame["Date"] = pd.to_datetime(frame["Date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
        with self.assertRaises(ValueError):
            common.standardise_raw(frame)

    def test_non_numeric_cells_become_missing_not_errors(self):
        frame = synthetic.make_raw_frame(days=5)
        frame["Rainfall(mm)"] = frame["Rainfall(mm)"].astype(object)
        frame.loc[3, "Rainfall(mm)"] = "n/a"
        raw = common.standardise_raw(frame)
        self.assertTrue(math.isnan(raw.loc[3, "rainfall_mm"]))


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = common.standardise_raw(synthetic.make_raw_frame(days=120))

    def test_cleaning_report_adds_up(self):
        clean, report = common.clean_with_report(self.raw)
        self.assertEqual(report["rows_raw"], len(self.raw))
        self.assertEqual(report["non_functioning_rows_dropped"], int((~self.raw["functioning_day"]).sum()))
        self.assertEqual(
            report["rows_raw"]
            - report["duplicate_rows_dropped"]
            - report["non_functioning_rows_dropped"]
            - report["rows_dropped_missing_key_fields"],
            report["rows_clean"],
        )
        self.assertEqual(report["rows_clean"], len(clean))

    def test_cleaned_table_properties(self):
        clean = common.clean_data(self.raw)
        self.assertNotIn("functioning_day", clean.columns)
        self.assertEqual(len(clean), int(self.raw["functioning_day"].sum()))  # exactly the open hours remain
        self.assertTrue(clean["hour"].between(0, 23).all())
        order = list(zip(clean["date"], clean["hour"]))
        self.assertEqual(order, sorted(order))
        self.assertEqual(clean.index.tolist(), list(range(len(clean))))

    def test_exact_duplicates_are_dropped(self):
        doubled = pd.concat([self.raw, self.raw.iloc[:10]], ignore_index=True)
        _, report = common.clean_with_report(doubled)
        self.assertEqual(report["duplicate_rows_dropped"], 10)

    def test_negative_counts_are_rejected(self):
        broken = self.raw.copy()
        open_hours = broken.index[broken["functioning_day"]]
        broken.loc[open_hours[0], common.TARGET] = -5
        with self.assertRaises(ValueError):
            common.clean_data(broken)

    def test_hour_out_of_range_is_rejected(self):
        broken = self.raw.copy()
        open_hours = broken.index[broken["functioning_day"]]
        broken.loc[open_hours[0], "hour"] = 24
        with self.assertRaises(ValueError):
            common.clean_data(broken)


class SplitTests(unittest.TestCase):
    def setUp(self):
        raw = common.standardise_raw(synthetic.make_raw_frame(days=100))
        self.data = common.clean_data(raw)

    def test_chronological_split(self):
        train, test, cutoff = common.time_split_masks(self.data["date"])
        self.assertEqual(int(train.sum() + test.sum()), len(self.data))
        self.assertFalse((train & test).any())
        self.assertLess(self.data.loc[train, "date"].max(), cutoff)
        self.assertGreaterEqual(self.data.loc[test, "date"].min(), cutoff)
        n_days = self.data["date"].nunique()
        test_days = self.data.loc[test, "date"].nunique()
        self.assertEqual(test_days, round(n_days * common.TEST_FRACTION))

    def test_split_needs_more_than_one_day(self):
        one_day = pd.Series(pd.to_datetime(["2018-01-01"] * 24))
        with self.assertRaises(ValueError):
            common.time_split_masks(one_day)

    def test_block_groups_keep_weeks_together(self):
        dates = pd.Series(pd.date_range("2018-01-01", periods=14 * 24, freq="h"))
        groups = common.block_groups(dates)
        self.assertEqual(sorted(set(groups.tolist())), [0, 1])
        self.assertTrue((groups[: 7 * 24] == 0).all())
        self.assertTrue((groups[7 * 24 :] == 1).all())


class FeatureTests(unittest.TestCase):
    def frame(self, **overrides):
        row = {
            "date": "2018-07-18", "hour": 18, "holiday": False, "temperature_c": 28.4, "humidity_pct": 55.0,
            "wind_speed_ms": 1.8, "visibility_10m": 1800.0, "dew_point_c": 18.5,
            "solar_radiation_mj_m2": 0.45, "rainfall_mm": 0.0, "snowfall_cm": 0.0,
        }
        row.update(overrides)
        frame = pd.DataFrame([row])
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def test_columns_and_order(self):
        features = common.build_features(self.frame())
        self.assertEqual(list(features.columns), common.FEATURE_COLUMNS)

    def test_calendar_features(self):
        wednesday = common.build_features(self.frame(date="2018-07-18")).iloc[0]
        self.assertEqual((wednesday["day_of_week"], wednesday["is_weekend"], wednesday["day_off"]), (2, 0, 0))
        self.assertEqual(wednesday["hour_daytype"], "18_work")
        saturday = common.build_features(self.frame(date="2018-07-21", hour=7)).iloc[0]
        self.assertEqual((saturday["day_of_week"], saturday["is_weekend"], saturday["day_off"]), (5, 1, 1))
        self.assertEqual(saturday["hour_daytype"], "07_off")
        holiday = common.build_features(self.frame(date="2018-03-01", holiday=True)).iloc[0]  # a Thursday
        self.assertEqual((holiday["is_weekend"], holiday["holiday"], holiday["day_off"]), (0, 1, 1))

    def test_precipitation_flags(self):
        dry = common.build_features(self.frame()).iloc[0]
        self.assertEqual((dry["is_raining"], dry["is_snowing"]), (0, 0))
        wet = common.build_features(self.frame(rainfall_mm=0.5, snowfall_cm=1.0)).iloc[0]
        self.assertEqual((wet["is_raining"], wet["is_snowing"]), (1, 1))

    def test_zero_humidity_is_treated_as_missing(self):
        features = common.build_features(self.frame(humidity_pct=0.0))
        self.assertTrue(math.isnan(features.loc[0, "humidity_pct"]))
        features = common.build_features(self.frame(humidity_pct=1.0))
        self.assertEqual(features.loc[0, "humidity_pct"], 1.0)

    def test_missing_input_column_is_reported(self):
        with self.assertRaises(ValueError):
            common.build_features(self.frame().drop(columns=["hour"]))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = common.standardise_raw(synthetic.make_raw_frame(days=100))
        cls.data = common.clean_data(raw)
        cls.X = cls.data[common.INPUT_COLUMNS]
        cls.y = cls.data[common.TARGET]

    def test_every_model_fits_and_predicts_non_negative_counts(self):
        small = {
            "ridge": {"alpha": 1.0},
            "decision_tree": {"max_depth": 5, "min_samples_leaf": 5},
            "random_forest": {"n_estimators": 10, "max_depth": 6},
            "hist_gradient_boosting": {"max_iter": 30},
        }
        for name in common.MODEL_NAMES:
            with self.subTest(model=name):
                pipeline = common.build_pipeline(name, small[name]).fit(self.X, self.y)
                predictions = common.predict_counts(pipeline, self.X.head(50))
                self.assertEqual(predictions.shape, (50,))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue((predictions >= 0).all())

    def test_unknown_model_name(self):
        with self.assertRaises(ValueError):
            common.build_pipeline("xgboost")

    def test_params_override_defaults(self):
        pipeline = common.build_pipeline("random_forest", {"n_estimators": 7})
        self.assertEqual(pipeline.named_steps["model"].n_estimators, 7)
        self.assertEqual(pipeline.named_steps["model"].max_depth, common.DEFAULT_PARAMS["random_forest"]["max_depth"])

    def test_pipeline_survives_missing_weather_values(self):
        pipeline = common.build_pipeline("hist_gradient_boosting", {"max_iter": 20}).fit(self.X, self.y)
        row = self.X.head(3).copy()
        row["visibility_10m"] = np.nan
        self.assertTrue(np.isfinite(common.predict_counts(pipeline, row)).all())

    def test_training_and_serving_paths_agree(self):
        """A record sent to the API must give exactly the prediction of the same row in the training frame."""
        pipeline = common.build_pipeline("random_forest", {"n_estimators": 15, "max_depth": 8}).fit(self.X, self.y)
        zero_humidity = self.data.index[self.data["humidity_pct"] == 0][:1].tolist()
        rows = self.data.sample(6, random_state=1).index.tolist() + zero_humidity
        records = []
        for i in rows:
            row = self.data.loc[i]
            record = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "hour": int(row["hour"]),
                "holiday": bool(row["holiday"]),
            }
            record.update({column: float(row[column]) for column in common.WEATHER_COLUMNS})
            clean, errors = common.validate_record(record)
            self.assertEqual(errors, [])
            records.append(clean)
        served = common.predict_counts(pipeline, common.records_to_frame(records))
        trained = common.predict_counts(pipeline, self.X.loc[rows])
        np.testing.assert_allclose(served, trained)

    def test_profile_baseline_uses_training_means(self):
        train, test = self.data.iloc[:1500], self.data.iloc[1500:1800]
        prediction = common.profile_baseline(train, test)
        self.assertEqual(prediction.shape, (len(test),))
        self.assertTrue(np.isfinite(prediction).all())
        key = common.build_features(train[common.INPUT_COLUMNS])["hour_daytype"].to_numpy()
        expected = train[common.TARGET].groupby(key).mean()
        first_key = common.build_features(test[common.INPUT_COLUMNS])["hour_daytype"].iloc[0]
        self.assertAlmostEqual(prediction[0], expected[first_key])

    def test_cross_validation_returns_folds_and_oof_predictions(self):
        groups = common.block_groups(self.data["date"])
        result = common.cross_validate_model(common.build_pipeline("ridge"), self.X, self.y, groups, n_splits=4)
        self.assertEqual(len(result["folds"]), 4)
        self.assertEqual(len(result["oof_pred"]), len(self.y))
        self.assertFalse(np.isnan(result["oof_pred"]).any())
        for key in ("rmse_mean", "rmse_std", "mae_mean", "r2_mean", "wape_mean"):
            self.assertIn(key, result["summary"])

    def test_ridge_target_transform_option(self):
        with_log = common.build_pipeline("ridge", {"log_target": True}).named_steps["model"]
        raw_count = common.build_pipeline("ridge", {"log_target": False}).named_steps["model"]
        self.assertIs(with_log.func, np.log1p)
        self.assertIsNone(raw_count.func)
        for log_target in (True, False):
            pipeline = common.build_pipeline("ridge", {"log_target": log_target}).fit(self.X, self.y)
            self.assertTrue(np.isfinite(common.predict_counts(pipeline, self.X.head(20))).all())

    def test_cross_validation_with_a_shuffled_splitter(self):
        splitter = KFold(n_splits=3, shuffle=True, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # groups=None must not trigger "groups is ignored" warnings
            result = common.cross_validate_model(common.build_pipeline("ridge"), self.X, self.y, None, splitter=splitter)
        self.assertEqual(len(result["folds"]), 3)
        self.assertFalse(np.isnan(result["oof_pred"]).any())

    def test_training_ranges(self):
        ranges = common.training_ranges(self.X)
        self.assertEqual(set(ranges), set(common.WEATHER_COLUMNS))
        self.assertGreater(ranges["humidity_pct"][0], 0)  # the zero readings are excluded
        for low, high in ranges.values():
            self.assertLessEqual(low, high)


class MetricsTests(unittest.TestCase):
    def test_clipped_rmse_matches_the_scorer(self):
        self.assertAlmostEqual(common.clipped_rmse([1, 2], [-5, 2]), math.sqrt(0.5))  # -5 is clipped to 0
        self.assertAlmostEqual(common.clipped_rmse([3, 3], [3, 3]), 0.0)

        model = DummyRegressor(strategy="constant", constant=-5.0).fit([[0], [0]], [1, 2])  # always predicts -5
        self.assertAlmostEqual(common.RMSE_SCORER(model, [[0], [0]], [1, 2]), -math.sqrt((1**2 + 2**2) / 2))

    def test_summarise_folds(self):
        folds = [{"rmse": 1.0, "mae": 1.0, "r2": 0.5, "wape": 0.1}, {"rmse": 3.0, "mae": 2.0, "r2": 0.7, "wape": 0.3}]
        summary = common.summarise_folds(folds)
        self.assertAlmostEqual(summary["rmse_mean"], 2.0)
        self.assertAlmostEqual(summary["rmse_std"], 1.0)
        self.assertAlmostEqual(summary["r2_mean"], 0.6)

    def test_perfect_prediction(self):
        m = common.regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual((m["rmse"], m["mae"], m["r2"], m["wape"]), (0.0, 0.0, 1.0, 0.0))

    def test_known_values(self):
        m = common.regression_metrics([2, 4], [1, 5])
        self.assertAlmostEqual(m["rmse"], 1.0)
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["r2"], 0.0)
        self.assertAlmostEqual(m["wape"], 2 / 6)

    def test_constant_target_gives_nan_r2(self):
        self.assertTrue(math.isnan(common.regression_metrics([5, 5], [4, 6])["r2"]))


class ValidationTests(unittest.TestCase):
    VALID = {
        "date": "2018-07-18", "hour": 18, "holiday": False, "temperature_c": 28.4, "humidity_pct": 55,
        "wind_speed_ms": 1.8, "visibility_10m": 1800, "dew_point_c": 18.5, "solar_radiation_mj_m2": 0.45,
        "rainfall_mm": 0, "snowfall_cm": 0,
    }

    def test_valid_record(self):
        clean, errors = common.validate_record(dict(self.VALID))
        self.assertEqual(errors, [])
        self.assertEqual(clean["date"], datetime.date(2018, 7, 18))
        self.assertEqual(clean["hour"], 18)
        self.assertIsInstance(clean["humidity_pct"], float)
        self.assertIs(clean["holiday"], False)

    def test_spec_matches_input_columns(self):
        self.assertEqual(set(common.INPUT_SPEC), set(common.INPUT_COLUMNS))
        self.assertEqual(set(self.VALID), set(common.INPUT_COLUMNS))

    def test_integer_valued_float_hour_is_accepted(self):
        clean, errors = common.validate_record(dict(self.VALID, hour=18.0))
        self.assertEqual((errors, clean["hour"]), ([], 18))

    def test_invalid_values(self):
        bad = {
            "hour above range": dict(self.VALID, hour=24),
            "hour below range": dict(self.VALID, hour=-1),
            "hour with fraction": dict(self.VALID, hour=18.5),
            "hour as boolean": dict(self.VALID, hour=True),
            "hour as text": dict(self.VALID, hour="18"),
            "number as text": dict(self.VALID, temperature_c="warm"),
            "number as boolean": dict(self.VALID, temperature_c=True),
            "nan": dict(self.VALID, temperature_c=float("nan")),
            "infinity": dict(self.VALID, wind_speed_ms=float("inf")),
            "humidity above 100": dict(self.VALID, humidity_pct=100.5),
            "negative rainfall": dict(self.VALID, rainfall_mm=-0.1),
            "temperature out of physical range": dict(self.VALID, temperature_c=99),
            "date not a string": dict(self.VALID, date=20180718),
            "date wrong format": dict(self.VALID, date="18/07/2018"),
            "date does not exist": dict(self.VALID, date="2018-02-30"),
            "date year out of range": dict(self.VALID, date="1800-01-01"),
            "holiday as number": dict(self.VALID, holiday=1),
            "holiday as text": dict(self.VALID, holiday="true"),
        }
        for label, record in bad.items():
            with self.subTest(label):
                clean, errors = common.validate_record(record)
                self.assertIsNone(clean)
                self.assertTrue(errors)

    def test_unknown_and_missing_fields(self):
        clean, errors = common.validate_record(dict(self.VALID, season="Summer"))
        self.assertIsNone(clean)
        self.assertIn("season", " ".join(errors))
        partial = {k: v for k, v in self.VALID.items() if k not in ("hour", "date")}
        clean, errors = common.validate_record(partial)
        self.assertIsNone(clean)
        self.assertIn("date", " ".join(errors))
        self.assertIn("hour", " ".join(errors))

    def test_non_object_records(self):
        for value in (None, [], [1, 2], "text", 5):
            with self.subTest(value=value):
                clean, errors = common.validate_record(value)
                self.assertIsNone(clean)
                self.assertEqual(errors, ["a record must be a JSON object"])

    def test_all_errors_are_reported_together(self):
        clean, errors = common.validate_record(dict(self.VALID, hour=30, humidity_pct=-4))
        self.assertIsNone(clean)
        self.assertEqual(len(errors), 2)

    def test_warnings(self):
        ranges = {c: [-100.0, 10000.0] for c in common.WEATHER_COLUMNS}
        clean, _ = common.validate_record(dict(self.VALID))
        self.assertEqual(common.input_warnings(clean, ranges), [])
        outside = dict(clean, temperature_c=20000.0)
        warnings = common.input_warnings(outside, ranges)
        self.assertTrue(any("temperature_c" in w for w in warnings))
        implausible = dict(clean, dew_point_c=30.0, temperature_c=20.0)
        self.assertTrue(any("dew_point_c is higher" in w for w in common.input_warnings(implausible, None)))


class ParamsAndBundleTests(unittest.TestCase):
    def test_missing_params_file_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            params, meta = common.load_params(Path(folder) / "nope.json")
        self.assertEqual(params, common.DEFAULT_PARAMS)
        self.assertEqual(meta["status"], "defaults")

    def test_params_roundtrip_and_numpy_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "params.json"
            params = {name: dict(values) for name, values in common.DEFAULT_PARAMS.items()}
            params["ridge"] = {"alpha": np.float64(3.5)}
            params["random_forest"]["max_depth"] = np.int64(12)
            params["random_forest"]["max_features"] = None
            common.save_params(path, params, {"status": "tuned"})
            loaded, meta = common.load_params(path)
        self.assertEqual(meta["status"], "tuned")
        self.assertEqual(loaded["ridge"]["alpha"], 3.5)
        self.assertEqual(loaded["random_forest"]["max_depth"], 12)
        self.assertIsNone(loaded["random_forest"]["max_features"])

    def test_unknown_model_in_params_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "params.json"
            path.write_text('{"_meta": {}, "xgboost": {"max_depth": 3}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                common.load_params(path)

    def test_plain_params(self):
        searched = {"model__regressor__alpha": np.float64(2.0), "model__max_depth": np.int64(7)}
        self.assertEqual(common.plain_params(searched), {"alpha": 2.0, "max_depth": 7})
        self.assertIsInstance(common.plain_params(searched)["max_depth"], int)

    def test_bundle_roundtrip(self):
        raw = common.standardise_raw(synthetic.make_raw_frame(days=40))
        data = common.clean_data(raw)
        pipeline = common.build_pipeline("ridge").fit(data[common.INPUT_COLUMNS], data[common.TARGET])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.bin"
            common.save_bundle(path, pipeline, {"model_name": "ridge"})
            bundle = common.load_bundle(path)
            self.assertEqual(bundle["metadata"], {"model_name": "ridge"})
            np.testing.assert_allclose(
                common.predict_counts(bundle["pipeline"], data[common.INPUT_COLUMNS].head(5)),
                common.predict_counts(pipeline, data[common.INPUT_COLUMNS].head(5)),
            )
            with self.assertRaises(FileNotFoundError):
                common.load_bundle(Path(folder) / "missing.bin")
            wrong = Path(folder) / "wrong.bin"
            wrong.write_bytes(b"\x80\x04N.")  # a pickled None
            with self.assertRaises(ValueError):
                common.load_bundle(wrong)


if __name__ == "__main__":
    unittest.main()
