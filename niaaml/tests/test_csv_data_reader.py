from unittest import TestCase
from niaaml.data import CSVDataReader
import os


class CSVDataReaderTestCase(TestCase):
    def test_header_classes_works_fine(self):
        data_reader = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_header_classes.csv",
            has_header=True,
            contains_classes=True,
        )
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertEqual(y.shape, (100,))

    def test_no_header_classes_works_fine(self):
        data_reader = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_no_header_classes.csv",
            has_header=False,
            contains_classes=True,
        )
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertEqual(y.shape, (100,))

    def test_no_header_no_classes_works_fine(self):
        data_reader = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_no_header_no_classes.csv",
            has_header=False,
            contains_classes=False,
        )
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertIsNone(y)

    def test_header_no_classes_works_fine(self):
        data_reader = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_header_no_classes.csv",
            has_header=True,
            contains_classes=False,
        )
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertIsNone(y)
