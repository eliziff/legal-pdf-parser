import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from install_ocr_runtime import destination, install_models, released_models


class InstallationTests(unittest.TestCase):
    def test_latest_release_requires_a_named_digest_verified_owner_asset(self):
        url = "https://github.com/eliziff/legal-browser-ocr/releases/download/v9.8.7/legal-ocr-models.tar.gz"
        asset = {"name": "legal-ocr-models.tar.gz", "browser_download_url": url, "digest": "sha256:" + "a" * 64}
        self.assertEqual(released_models({"assets": [asset]}), (url, "a" * 64))
        for value in ({"assets": []}, {"assets": [{**asset, "digest": ""}]},
                      {"assets": [{**asset, "browser_download_url": "https://example.test/archive.tar.gz"}]}):
            with self.assertRaises(ValueError):
                released_models(value)

    def test_archive_must_match_inventory_and_checksums(self):
        asset = b"independently invented model bytes"
        manifest = {"recognizer": "indra", "input_height": 32, "assets": {
            "assets/model.onnx": {"bytes": len(asset), "sha256": hashlib.sha256(asset).hexdigest()}
        }}
        def package(value):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                for name, data in [("manifest.json", json.dumps(manifest).encode()), ("assets/model.onnx", value)]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            return stream.getvalue()
        with tempfile.TemporaryDirectory(prefix="legalocr-install-test-") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "checksum"):
                install_models(package(b"wrong"), root)
            self.assertFalse((root / "manifest.json").exists())
            install_models(package(asset), root)
            self.assertEqual((root / "assets/model.onnx").read_bytes(), asset)
            for name in ("../escape", "/outside", "C:/outside", "assets\\..\\escape"):
                with self.assertRaises(ValueError):
                    destination(root, name)


if __name__ == "__main__":
    unittest.main()
