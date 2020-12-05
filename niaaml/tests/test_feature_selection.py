from unittest import TestCase
from niaaml.preprocessing.feature_selection import ParticleSwarmOptimization, SelectKBest
from niaaml.data import CSVDataReader
import os

class FeatureSelectionTestCase(TestCase):
    def setUp(self):
        self.__algo1 = ParticleSwarmOptimization()
        self.__algo2 = SelectKBest()
    
    def test_select_features_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_classes.csv', has_header=True, contains_classes=True)

        selected_features_mask = self.__algo1.select_features(data_reader.get_x(), data_reader.get_y())
        self.assertEqual(data_reader.get_x().shape[1], len(selected_features_mask))

        selected_features_mask = self.__algo2.select_features(data_reader.get_x(), data_reader.get_y())
        self.assertEqual(data_reader.get_x().shape[1], len(selected_features_mask))