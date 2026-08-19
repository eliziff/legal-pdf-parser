import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('benchmark', Path(__file__).with_name('benchmark.py'))
benchmark = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(benchmark)


class GoldOrderTest(unittest.TestCase):
    def test_unindexed_lines_keep_document_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'page.xml'
            path.write_text('<PcGts><Page><TextRegion><TextLine><TextEquiv><Unicode>Z first</Unicode></TextEquiv></TextLine><TextLine><TextEquiv><Unicode>A second</Unicode></TextEquiv></TextLine></TextRegion></Page></PcGts>', encoding='utf-8')
            self.assertEqual(benchmark.gold(path), 'Z first A second')

    def test_soft_hyphen_and_not_sign_are_equivalent(self):
        self.assertEqual(benchmark.normalized('juris\u00ac\ndiction'), 'jurisdiction')
        self.assertEqual(benchmark.normalized('juris\u00ad\ndiction'), 'jurisdiction')
        self.assertEqual(benchmark.normalized('answer\u00ac You'), 'answer You')


if __name__ == '__main__': unittest.main()
