import importlib.util
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location('native_ocr', Path(__file__).with_name('ocr.py'))
ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocr)


class CleanModelTextTests(unittest.TestCase):
    def test_not_sign_and_soft_hyphen(self):
        self.assertEqual(ocr.clean_model_text('juris\u00ac\ndiction'), 'jurisdiction')
        self.assertEqual(ocr.clean_model_text('juris\u00ad\ndiction'), 'jurisdiction')
        self.assertEqual(ocr.clean_model_text('answer\u00ac You'), 'answer You')

    def test_balanced_tier_is_the_measured_cascade(self):
        self.assertEqual(ocr.TIERS['balanced']['fallback_threshold'], .70)
        self.assertEqual(ocr.TIERS['balanced']['layout'], 'tesseract')
        self.assertEqual(ocr.TIERS['fidelity']['layout'], 'blla')

    def test_column_footnotes_follow_both_body_columns(self):
        box = lambda x, y, h=20: ocr.LineBox(x, y, x + 360, y + h)
        left_body = [box(75, 120 + i * 25) for i in range(12)]
        left_notes = [box(92, 500 + i * 20, 16) for i in range(4)]
        right_body = [box(465, 120 + i * 25) for i in range(12)]
        right_notes = [box(480, 500 + i * 20, 16) for i in range(4)]
        self.assertEqual(ocr.order_tesseract_boxes(left_body + left_notes + right_body + right_notes, 900, 700),
                         left_body + right_body + left_notes + right_notes)


if __name__ == '__main__':
    unittest.main()
