from unittest import TestCase
import niaaml.preprocessing.feature_selection as fs
from niaaml.data import CSVDataReader
import os


class FeatureSelectionTestCase(TestCase):
    def setUp(self):
        self.__data = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_header_classes.csv",
            has_header=True,
            contains_classes=True,
        )

    def test_PSO_works_fine(self):
        algo = fs.ParticleSwarmOptimization()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))

    def test_select_k_best_works_fine(self):
        algo = fs.SelectKBest()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))

    def test_select_percentile_works_fine(self):
        algo = fs.SelectPercentile()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))

    def test_bat_algorithm_works_fine(self):
        algo = fs.BatAlgorithm()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))

    def test_de_works_fine(self):
        algo = fs.DifferentialEvolution()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))

    def test_gwo_works_fine(self):
        algo = fs.GreyWolfOptimizer()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))

    def test_jdefsth_works_fine(self):
        algo = fs.jDEFSTH()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))

    def test_vt_works_fine(self):
        algo = fs.VarianceThreshold()
        selected_features_mask = algo.select_features(
            self.__data.get_x(), self.__data.get_y()
        )
        self.assertEqual(self.__data.get_x().shape[1], len(selected_features_mask))
