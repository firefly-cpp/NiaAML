from unittest import TestCase
from niaaml import Pipeline, OptimizationStats
from niaaml.classifiers import Bagging, AdaBoost
from niaaml.preprocessing.feature_selection import SelectKBest, SelectPercentile
from niaaml.preprocessing.feature_transform import StandardScaler, Normalizer
from niaaml.data import CSVDataReader
import os
import numpy
import tempfile

class PipelineTestCase(TestCase):
    def setUp(self):
        self.__pipeline = Pipeline(
            feature_selection_algorithm=SelectKBest(),
            feature_transform_algorithm=Normalizer(),
            classifier=Bagging()
        )

    def test_pipeline_optimize_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_classes.csv', has_header=True, contains_classes=True)
        
        self.assertIsInstance(self.__pipeline.get_classifier(), Bagging)
        self.assertIsInstance(self.__pipeline.get_feature_selection_algorithm(), SelectKBest)
        self.assertIsInstance(self.__pipeline.get_feature_transform_algorithm(), Normalizer)

        accuracy = self.__pipeline.optimize(data_reader.get_x(), data_reader.get_y(), 10, 30, 'ParticleSwarmAlgorithm', 'Accuracy')

        self.assertGreaterEqual(accuracy, -1.0)
        self.assertLessEqual(accuracy, 0.0)

        self.assertIsInstance(self.__pipeline.get_classifier(), Bagging)
        self.assertIsInstance(self.__pipeline.get_feature_selection_algorithm(), SelectKBest)
        self.assertIsInstance(self.__pipeline.get_feature_transform_algorithm(), Normalizer)

    def test_pipeline_run_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_classes.csv', has_header=True, contains_classes=True)
        self.__pipeline.optimize(data_reader.get_x(), data_reader.get_y(), 10, 30, 'ParticleSwarmAlgorithm', 'Accuracy')
        predicted = self.__pipeline.run(numpy.random.uniform(low=0.0, high=15.0, size=(30, data_reader.get_x().shape[1])))

        self.assertEqual(predicted.shape, (30, ))

        s1 = set(data_reader.get_y())
        s2 = set(predicted)
        self.assertTrue(s2.issubset(s1))
        self.assertTrue(len(s2) > 0 and len(s2) <= 2)
    
    def test_pipeline_export_works_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.__pipeline.export(os.path.join(tmp, 'pipeline'))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'pipeline.ppln')))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))

            self.__pipeline.export(os.path.join(tmp, 'pipeline.ppln'))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'pipeline.ppln')))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))

    def test_pipeline_export_text_works_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.__pipeline.export_text(os.path.join(tmp, 'pipeline'))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'pipeline.txt')))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))

            self.__pipeline.export_text(os.path.join(tmp, 'pipeline.txt'))
            self.assertTrue(os.path.exists(os.path.join(tmp, 'pipeline.txt')))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))
        
        self.assertIsNotNone(self.__pipeline.to_string())
        self.assertGreater(len(self.__pipeline.to_string()), 0)

    def test_pipeline_setters_work_fine(self):
        self.__pipeline.set_classifier(AdaBoost())
        self.__pipeline.set_feature_selection_algorithm(SelectPercentile())
        self.__pipeline.set_feature_transform_algorithm(StandardScaler())
        self.__pipeline.set_selected_features_mask(numpy.ones([1, 1, 0, 0], dtype=bool))

        self.__y = numpy.array(['Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 1', 'Class 2',
       'Class 2', 'Class 2', 'Class 2', 'Class 1', 'Class 1', 'Class 2',
       'Class 1', 'Class 2', 'Class 1', 'Class 1', 'Class 1', 'Class 1',
       'Class 2', 'Class 1'])
        self.__predicted = numpy.array(['Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 2', 'Class 2',
       'Class 1', 'Class 1', 'Class 1', 'Class 2', 'Class 1', 'Class 1',
       'Class 2', 'Class 2', 'Class 1', 'Class 2', 'Class 1', 'Class 2',
       'Class 2', 'Class 2'])
        self.__pipeline.set_stats(OptimizationStats(self.__predicted, self.__y))

        self.assertIsInstance(self.__pipeline.get_classifier(), AdaBoost)
        self.assertIsInstance(self.__pipeline.get_feature_selection_algorithm(), SelectPercentile)
        self.assertIsInstance(self.__pipeline.get_feature_transform_algorithm(), StandardScaler)
        self.assertIsInstance(self.__pipeline.get_stats(), OptimizationStats)