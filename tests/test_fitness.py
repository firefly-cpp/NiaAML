import numpy as np
from unittest import TestCase
import niaaml.fitness as f


class FitnessTestCase(TestCase):
    def setUp(self):
        self.__y = np.array(
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
        self.__predicted = np.array(
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

    def test_accuracy_works_fine(self):
        ff = f.Accuracy()
        val = ff.get_fitness(self.__predicted, self.__y)
        self.assertEqual(val, 0.5)

    def test_precision_works_fine(self):
        ff = f.Precision()
        val = ff.get_fitness(self.__predicted, self.__y)
        self.assertEqual(val, 0.5199999999999999)

    def test_cohen_kappa_works_fine(self):
        ff = f.CohenKappa()
        val = ff.get_fitness(self.__predicted, self.__y)
        self.assertEqual(val, 0.0)

    def test_f1_works_fine(self):
        ff = f.F1()
        val = ff.get_fitness(self.__predicted, self.__y)
        self.assertEqual(val, 0.505050505050505)


class RegressionFitnessTestCase(TestCase):
    def setUp(self):
        self.__y = np.array([3, -0.5, 2, 7])
        self.__predicted = np.array([2.5, 0.0, 2, 8])

    def test_r2_works_fine(self):
        ff = f.R2()
        val = ff.get_fitness(self.__predicted, self.__y)
        self.assertEqual(val, 0.9486081370449679)
