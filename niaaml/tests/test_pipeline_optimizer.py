from unittest import TestCase
from niaaml import PipelineOptimizer, Pipeline
from niaaml.classifiers import AdaBoost, Bagging
from niaaml.preprocessing.feature_selection import SelectKBest, SelectPercentile
from niaaml.preprocessing.feature_transform import Normalizer, StandardScaler
from niaaml.data import CSVDataReader
import os
import numpy

class PipelineOptimizerTestCase(TestCase):

    def setUp(self):
        self.__data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_classes.csv', has_header=True, contains_classes=True)
        self.__pipeline_optimizer = PipelineOptimizer(
            data=self.__data_reader,
            feature_selection_algorithms=['SelectKBest', 'SelectPercentile'],
            feature_transform_algorithms=['Normalizer', 'StandardScaler'],
            classifiers=['AdaBoost', 'Bagging']
        )

    def test_pipeline_optimizeer_run_works_fine(self):
        pipeline = self.__pipeline_optimizer.run('Accuracy', 10, 10, 20, 20, 'ParticleSwarmAlgorithm')
        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(isinstance(pipeline.get_classifier(), AdaBoost) or isinstance(pipeline.get_classifier(), Bagging))
        self.assertTrue(isinstance(pipeline.get_feature_selection_algorithm(), SelectKBest) or isinstance(pipeline.get_feature_selection_algorithm(), SelectPercentile))
        self.assertTrue(pipeline.get_feature_transform_algorithm() is None or isinstance(pipeline.get_feature_transform_algorithm(), Normalizer) or isinstance(pipeline.get_feature_transform_algorithm(), StandardScaler))

    def test_pipeline_optimizer_getters_work_fine(self):
        fsas = self.__pipeline_optimizer.get_feature_selection_algorithms()
        ftas = self.__pipeline_optimizer.get_feature_transform_algorithms()
        classifiers = self.__pipeline_optimizer.get_classifiers()

        self.assertEqual(self.__pipeline_optimizer.get_data(), self.__data_reader)
        self.assertTrue((numpy.array(['AdaBoost', 'Bagging']) == numpy.array(classifiers)).all())
        self.assertTrue((numpy.array(['SelectKBest', 'SelectPercentile']) == numpy.array(fsas)).all())

        self.assertTrue((numpy.array([None, 'Normalizer', 'StandardScaler'] == numpy.array(ftas))).all())

    def test_pipeline_optimizeer_missing_values_categorical_attributes_run_works_fine(self):
        data_reader = CSVDataReader(src=os.path.dirname(os.path.abspath(__file__)) + '/tests_files/dataset_header_classes_cat_miss.csv', has_header=True, contains_classes=True)
        pipeline_optimizer = PipelineOptimizer(
            data=self.__data_reader,
            feature_selection_algorithms=['SelectKBest', 'SelectPercentile'],
            feature_transform_algorithms=['Normalizer', 'StandardScaler'],
            classifiers=['AdaBoost', 'Bagging'],
            categorical_features_encoder='OneHotEncoder',
            imputer='SimpleImputer'
        )

        pipeline = self.__pipeline_optimizer.run('Accuracy', 10, 10, 20, 20, 'ParticleSwarmAlgorithm')
        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(isinstance(pipeline.get_classifier(), AdaBoost) or isinstance(pipeline.get_classifier(), Bagging))
        self.assertTrue(isinstance(pipeline.get_feature_selection_algorithm(), SelectKBest) or isinstance(pipeline.get_feature_selection_algorithm(), SelectPercentile))
        self.assertTrue(pipeline.get_feature_transform_algorithm() is None or isinstance(pipeline.get_feature_transform_algorithm(), Normalizer) or isinstance(pipeline.get_feature_transform_algorithm(), StandardScaler))