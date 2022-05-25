from unittest import TestCase
from niaaml.preprocessing.imputation import SimpleImputer, impute_features
from niaaml.data import BasicDataReader
import numpy
import pandas


class ImputerTestCase(TestCase):
    def setUp(self):
        x = numpy.concatenate(
            (
                numpy.random.uniform(low=0.0, high=15.0, size=(100, 6)),
                numpy.array([numpy.random.choice(["a", "b"], size=(100,))]).T,
            ),
            axis=1,
        )
        x[50, 6] = numpy.nan
        x[30, 2] = numpy.nan
        y = numpy.random.choice(["Class 1", "Class 2"], size=100)
        self.__data_reader = BasicDataReader(x=x, y=y)

    def test_simple_imputer_works_fine(self):
        features = self.__data_reader.get_x()
        imputer1 = SimpleImputer()
        imputer1.fit(features[[2]])
        f = pandas.DataFrame(imputer1.transform(features[[2]]))
        self.assertFalse(f[0].isnull().any())

        imputer2 = SimpleImputer()
        imputer2.fit(features[[6]])
        f = pandas.DataFrame(imputer2.transform(features[[6]]))
        self.assertFalse(f[0].isnull().any())

    def test_utility_method_works_fine(self):
        features = self.__data_reader.get_x().astype(
            {
                0: "float64",
                1: "float64",
                2: "float64",
                3: "float64",
                4: "float64",
                5: "float64",
            }
        )
        features.iloc[50, 6] = numpy.nan
        features, imputers = impute_features(features, "SimpleImputer")
        self.assertEqual(len(imputers), 2)
        self.assertEqual(features.shape[1], 7)
