"""Tests for download_data.py. No network is used: zips are built locally and served through file:// URLs."""
import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import common  # noqa: E402
import download_data  # noqa: E402
import synthetic  # noqa: E402


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.csv_source = self.folder / "source" / common.CSV_NAME
        self.csv_source.parent.mkdir()
        synthetic.write_raw_csv(self.csv_source, days=30)
        self.zip_path = self.folder / "dataset.zip"
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            archive.write(self.csv_source, arcname=common.CSV_NAME)
        self.dest_dir = self.folder / "data"

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = download_data.main(["--dest-dir", str(self.dest_dir), *args])
        return code, buffer.getvalue()

    def test_from_zip(self):
        code, output = self.run_main("--from-zip", str(self.zip_path))
        self.assertEqual(code, 0, output)
        self.assertEqual((self.dest_dir / common.CSV_NAME).read_bytes(), self.csv_source.read_bytes())
        self.assertIn(common.CSV_NAME, output)
        self.assertIn("720 rows", output)

    def test_output_never_contains_local_paths(self):
        for args in (("--from-zip", str(self.zip_path)), ("--from-csv", str(self.csv_source))):
            _, output = self.run_main(*args)
            self.assertNotIn(str(self.folder), output)

    def test_from_csv(self):
        code, output = self.run_main("--from-csv", str(self.csv_source))
        self.assertEqual(code, 0, output)
        self.assertTrue((self.dest_dir / common.CSV_NAME).exists())

    def test_download_through_a_url(self):
        code, output = self.run_main("--url", self.zip_path.as_uri())
        self.assertEqual(code, 0, output)
        self.assertIn("Saved", output)
        self.assertTrue((self.dest_dir / common.CSV_NAME).exists())

    def test_existing_file_is_kept_unless_forced(self):
        self.run_main("--from-csv", str(self.csv_source))
        target = self.dest_dir / common.CSV_NAME
        target.write_bytes(self.csv_source.read_bytes())  # unchanged copy
        code, output = self.run_main("--url", (self.folder / "does-not-exist.zip").as_uri())
        self.assertEqual(code, 0, output)
        self.assertIn("already exists", output)
        code, output = self.run_main("--url", self.zip_path.as_uri(), "--force")
        self.assertEqual(code, 0, output)
        self.assertIn("Saved", output)

    def test_unreachable_url_gives_manual_instructions(self):
        code, output = self.run_main("--url", (self.folder / "does-not-exist.zip").as_uri())
        self.assertEqual(code, 1)
        self.assertIn("--from-zip", output)

    def test_missing_local_files(self):
        self.assertEqual(self.run_main("--from-zip", str(self.folder / "nope.zip"))[0], 1)
        self.assertEqual(self.run_main("--from-csv", str(self.folder / "nope.csv"))[0], 1)

    def test_zip_without_csv(self):
        path = self.folder / "empty.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("readme.txt", "hello")
        code, output = self.run_main("--from-zip", str(path))
        self.assertEqual(code, 1)
        self.assertIn("ERROR", output)

    def test_zip_with_several_csv_files(self):
        path = self.folder / "many.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a.csv", "x\n1\n")
            archive.writestr("b.csv", "y\n2\n")
        self.assertEqual(self.run_main("--from-zip", str(path))[0], 1)

    def test_csv_with_wrong_schema_is_rejected(self):
        path = self.folder / "wrong.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(common.CSV_NAME, "a,b\n1,2\n")
        code, output = self.run_main("--from-zip", str(path))
        self.assertEqual(code, 1)
        self.assertIn("Missing column", output)

    def test_not_a_zip(self):
        path = self.folder / "fake.zip"
        path.write_bytes(b"<html>error page</html>")
        code, output = self.run_main("--from-zip", str(path))
        self.assertEqual(code, 1)
        self.assertIn("ERROR", output)

    def test_unexpected_row_count_is_flagged(self):
        _, output = self.run_main("--from-zip", str(self.zip_path))
        self.assertIn(f"expected {common.EXPECTED_ROWS} rows", output)

    def test_csv_from_zip_prefers_the_dataset_file(self):
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            archive.writestr("notes/other.csv", "x\n1\n")
            archive.write(self.csv_source, arcname="inner/" + common.CSV_NAME)
        data = download_data.csv_from_zip(self.zip_path.read_bytes())
        self.assertEqual(data, self.csv_source.read_bytes())


if __name__ == "__main__":
    unittest.main()
