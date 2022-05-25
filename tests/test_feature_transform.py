from unittest import TestCase
import niaaml.preprocessing.feature_transform as ft
from niaaml.data import CSVDataReader
import os


class FeatureTransformTestCase(TestCase):
    def setUp(self):
        self.__data = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_header_classes.csv",
            has_header=True,
            contains_classes=True,
        )

    def test_mas_works_fine(self):
        algo = ft.MaxAbsScaler()
        algo.fit(self.__data.get_x())
        transformed = algo.transform(self.__data.get_x())
        self.assertEqual(transformed.shape, self.__data.get_x().shape)

    def test_norm_works_fine(self):
        algo = ft.Normalizer()
        algo.fit(self.__data.get_x())
        transformed = algo.transform(self.__data.get_x())
        self.assertEqual(transformed.shape, self.__data.get_x().shape)

    def test_qt_works_fine(self):
        algo = ft.QuantileTransformer()
        algo.fit(self.__data.get_x())
        transformed = algo.transform(self.__data.get_x())
        self.assertEqual(transformed.shape, self.__data.get_x().shape)

    def test_rs_works_fine(self):
        algo = ft.RobustScaler()
        algo.fit(self.__data.get_x())
        transformed = algo.transform(self.__data.get_x())
        self.assertEqual(transformed.shape, self.__data.get_x().shape)

    def test_ss_works_fine(self):
        algo = ft.StandardScaler()
        algo.fit(self.__data.get_x())
        transformed = algo.transform(self.__data.get_x())
        self.assertEqual(transformed.shape, self.__data.get_x().shape)
