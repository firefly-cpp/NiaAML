from unittest import TestCase
from niaaml.preprocessing.feature_transform import (
    FeatureTransformAlgorithmFactory,
    FeatureTransformAlgorithm,
)


class FeatureTransformAlgorithmFactoryTestCase(TestCase):
    def setUp(self):
        self.__factory = FeatureTransformAlgorithmFactory()

    def test_get_result_works_fine(self):
        for entry in self.__factory._entities:
            instance = self.__factory.get_result(entry)
            self.assertIsNotNone(instance)
            self.assertIsInstance(instance, FeatureTransformAlgorithm)

        with self.assertRaises(TypeError):
            self.__factory.get_result("non_existent_name")

    def test_get_dictionary_works_fine(self):
        d = self.__factory.get_name_to_classname_mapping()
        d_keys = d.keys()
        e_keys = self.__factory._entities.keys()

        self.assertEqual(len(e_keys), len(d_keys))

        for k in d:
            self.assertIsNotNone(d[k])
