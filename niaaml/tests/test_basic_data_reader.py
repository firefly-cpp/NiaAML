from unittest import TestCase
from niaaml.data import BasicDataReader
import numpy


class BasicDataReaderTestCase(TestCase):
    def setUp(self):
        self.__x = numpy.random.uniform(low=0.0, high=15.0, size=(100, 6))
        self.__y = numpy.random.choice(["Class 1", "Class 2"], size=100)

    def test_x_y_works_fine(self):
        data_reader = BasicDataReader(x=self.__x, y=self.__y)
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertEqual(y.shape, (100,))

        self.assertTrue(numpy.all(self.__x == x))
        self.assertTrue(numpy.all(self.__y == y))

    def test_no_y_works_fine(self):
        data_reader = BasicDataReader(x=self.__x)
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertIsNone(y)

        self.assertTrue(numpy.all(self.__x == x))
