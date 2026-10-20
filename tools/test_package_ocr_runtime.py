import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from package_ocr_runtime import load_base_runtime


class BaseRuntimeTests(unittest.TestCase):
    def archive(self, directory, *, damaged=False, extra=False):
        data = b"invented executable bytes"
        details = {"revision": "recorded-source", "files": {
            "legalocr.exe": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
        path = Path(directory) / "runtime.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("runtime-manifest.json", json.dumps(details))
            archive.writestr("README.txt", "Invented packaging fixture")
            archive.writestr("legalocr.exe", b"changed" if damaged else data)
            if extra:
                archive.writestr("unexpected.dll", b"unrecorded")
        return path

    def test_retains_binary_source_identity_and_records_base_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.archive(directory)
            files, details = load_base_runtime(path)
            self.assertEqual(files, {"legalocr.exe": b"invented executable bytes"})
            self.assertEqual(details["revision"], "recorded-source")
            self.assertEqual(details["base_runtime_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rejects_changed_or_unrecorded_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in [{"damaged": True}, {"extra": True}]:
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "changed"):
                    load_base_runtime(self.archive(directory, **options))


if __name__ == "__main__":
    unittest.main()
