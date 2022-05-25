from unittest import TestCase
from niaaml.preprocessing.encoding import OneHotEncoder, encode_categorical_features
from niaaml.data import BasicDataReader
import numpy


class FeatureEncoderTestCase(TestCase):
    def setUp(self):
        x = numpy.concatenate(
            (
                numpy.random.uniform(low=0.0, high=15.0, size=(100, 6)),
                numpy.array([numpy.random.choice(["a", "b"], size=(100,))]).T,
            ),
            axis=1,
        )
        y = numpy.random.choice(["Class 1", "Class 2"], size=100)
        self.__data_reader = BasicDataReader(x=x, y=y)

    def test_one_hot_encoder_works_fine(self):
        encoder = OneHotEncoder()
        features = self.__data_reader.get_x()
        encoder.fit(features[[6]])
        f = encoder.transform(features[[6]])

        ind = 0
        for i in features[6]:
            if i == "a":
                self.assertTrue(f.loc[ind, 0] == 1.0)
                self.assertTrue(f.loc[ind, 1] == 0.0)
            else:
                self.assertTrue(f.loc[ind, 0] == 0.0)
                self.assertTrue(f.loc[ind, 1] == 1.0)
            ind += 1

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
        features, encoders = encode_categorical_features(features, "OneHotEncoder")
        self.assertEqual(len(encoders), 1)
        self.assertEqual(features.shape[1], 8)
