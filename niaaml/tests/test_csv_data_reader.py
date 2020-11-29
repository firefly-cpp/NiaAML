from unittest import TestCase
from niaaml.data import CSVDataReader
import os
import numpy
import csv

class CSVDataReaderTestCase(TestCase):
    def __read_csv(self, src):
        arr = []
        with open(src) as f:
            reader = csv.reader(f)
            for row in reader:
                arr1 = []
                for el in row:
                    try:
                        arr1.append(float(el))
                    except ValueError:
                        arr1.append(el)
                arr.append(arr1)
        return numpy.array(arr, dtype=object)
    
    def test_header_classes_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_classes.csv', has_header=True, contains_classes=True)
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertEqual(y.shape, (100, ))

        csv_content = self.__read_csv(os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_classes.csv')
        data_reader_content = numpy.concatenate((numpy.array(x, dtype=object), numpy.array(y.reshape((100,1)), dtype=object)), axis=1)
        self.assertTrue(numpy.all(csv_content[1:] == data_reader_content))

    def test_no_header_classes_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_no_header_classes.csv', has_header=False, contains_classes=True)
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertEqual(y.shape, (100, ))

        csv_content = self.__read_csv(os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_no_header_classes.csv')
        data_reader_content = numpy.concatenate((numpy.array(x, dtype=object), numpy.array(y.reshape((100,1)), dtype=object)), axis=1)
        self.assertTrue(numpy.all(csv_content == data_reader_content))

    def test_no_header_no_classes_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_no_header_no_classes.csv', has_header=False, contains_classes=False)
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertIsNone(y)

        csv_content = self.__read_csv(os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_no_header_no_classes.csv')
        self.assertTrue(numpy.all(csv_content == numpy.array(x, dtype=object)))

    def test_header_no_classes_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_no_classes.csv', has_header=True, contains_classes=False)
        x = data_reader.get_x()
        y = data_reader.get_y()
        self.assertEqual(x.shape, (100, 6))
        self.assertIsNone(y)

        csv_content = self.__read_csv(os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_no_classes.csv')
        self.assertTrue(numpy.all(csv_content[1:] == numpy.array(x, dtype=object)))