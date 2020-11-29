from unittest import TestCase
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithmFactory, FeatureTransformAlgorithm

class FeatureTransformAlgorithmFactoryTestCase(TestCase):
    def setUp(self):
        self.__factory = FeatureTransformAlgorithmFactory()
    
    def test_get_result_works_fine(self):
        for entry in self.__factory._entities:
            instance = self.__factory.get_result(entry)
            self.assertIsNotNone(instance)
            self.assertIsInstance(instance, FeatureTransformAlgorithm)
        
        with self.assertRaises(TypeError):
            self.__factory.get_result('non_existent_name')