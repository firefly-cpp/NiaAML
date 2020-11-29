from unittest import TestCase
from niaaml.preprocessing.feature_selection import FeatureSelectionAlgorithmFactory, FeatureSelectionAlgorithm

class FeatureSelectionAlgorithmFactoryTestCase(TestCase):
    def setUp(self):
        self.__factory = FeatureSelectionAlgorithmFactory()
    
    def test_get_result_works_fine(self):
        for entry in self.__factory._entities:
            instance = self.__factory.get_result(entry)
            self.assertIsNotNone(instance)
            self.assertIsInstance(instance, FeatureSelectionAlgorithm)
        
        with self.assertRaises(TypeError):
            self.__factory.get_result('non_existent_name')