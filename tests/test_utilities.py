from unittest import TestCase
from niaaml import ParameterDefinition, MinMax, OptimizationStats, get_bin_index
import numpy as np
import tempfile


class UtilitiesTestCase(TestCase):
    def test_get_bin_index_works_fine(self):
        self.assertEqual(get_bin_index(0.0, 4), 0)
        self.assertEqual(get_bin_index(0.24, 4), 0)
        self.assertEqual(get_bin_index(0.25, 4), 1)
        self.assertEqual(get_bin_index(0.49, 4), 1)
        self.assertEqual(get_bin_index(0.5, 4), 2)
        self.assertEqual(get_bin_index(0.74, 4), 2)
        self.assertEqual(get_bin_index(0.75, 4), 3)
        self.assertEqual(get_bin_index(1.0, 4), 3)


class ParameterDefinitionTestCase(TestCase):
    def test_works_fine(self):
        parameter_definition = ParameterDefinition(MinMax(0.0, 5.9), float)

        self.assertIsInstance(parameter_definition.value, MinMax)
        self.assertEqual(parameter_definition.param_type, float)


class OptimizationStatsTestCase(TestCase):
    def setUp(self):
        y = np.array(
            [
                "Class 1",
                "Class 1",
                "Class 1",
                "Class 2",
                "Class 1",
                "Class 2",
                "Class 2",
                "Class 2",
                "Class 2",
                "Class 1",
                "Class 1",
                "Class 2",
                "Class 1",
                "Class 2",
                "Class 1",
                "Class 1",
                "Class 1",
                "Class 1",
                "Class 2",
                "Class 1",
            ]
        )
        predicted = np.array(
            [
                "Class 1",
                "Class 1",
                "Class 1",
                "Class 2",
                "Class 2",
                "Class 2",
                "Class 1",
                "Class 1",
                "Class 1",
                "Class 2",
                "Class 1",
                "Class 1",
                "Class 2",
                "Class 2",
                "Class 1",
                "Class 2",
                "Class 1",
                "Class 2",
                "Class 2",
                "Class 2",
            ]
        )

        self.__stats = OptimizationStats(predicted, y)

    def test_works_fine(self):
        self.assertEqual(self.__stats._accuracy, 0.5)
        self.assertEqual(self.__stats._precision, 0.5199999999999999)
        self.assertEqual(self.__stats._cohen_kappa, 0.0)
        self.assertEqual(self.__stats._f1_score, 0.505050505050505)


class MinMaxTestCase(TestCase):
    def test_works_fine(self):
        minmax = MinMax(0.0, 5.9)

        self.assertEqual(minmax.min, 0.0)
        self.assertEqual(minmax.max, 5.9)
