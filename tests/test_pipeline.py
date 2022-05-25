from unittest import TestCase
from niaaml import Pipeline, OptimizationStats
from niaaml.classifiers import RandomForest, AdaBoost
from niaaml.preprocessing.feature_selection import SelectKBest, SelectPercentile
from niaaml.preprocessing.feature_transform import StandardScaler, Normalizer
from niaaml.data import CSVDataReader
import os
import numpy
import tempfile
import pandas


class PipelineTestCase(TestCase):
    def test_pipeline_optimize_works_fine(self):
        pipeline = Pipeline(
            feature_selection_algorithm=SelectKBest(),
            feature_transform_algorithm=Normalizer(),
            classifier=RandomForest(),
        )

        data_reader = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_header_classes.csv",
            has_header=True,
            contains_classes=True,
        )

        self.assertIsInstance(pipeline.get_classifier(), RandomForest)
        self.assertIsInstance(pipeline.get_feature_selection_algorithm(), SelectKBest)
        self.assertIsInstance(pipeline.get_feature_transform_algorithm(), Normalizer)

        accuracy = pipeline.optimize(
            data_reader.get_x(),
            data_reader.get_y(),
            20,
            40,
            "ParticleSwarmAlgorithm",
            "Accuracy",
        )

        if accuracy != float("inf"):
            self.assertGreaterEqual(accuracy, -1.0)
            self.assertLessEqual(accuracy, 0.0)

        self.assertIsInstance(pipeline.get_classifier(), RandomForest)
        self.assertIsInstance(pipeline.get_feature_selection_algorithm(), SelectKBest)
        self.assertIsInstance(pipeline.get_feature_transform_algorithm(), Normalizer)

    def test_pipeline_run_works_fine(self):
        pipeline = Pipeline(
            feature_selection_algorithm=SelectKBest(),
            feature_transform_algorithm=Normalizer(),
            classifier=RandomForest(),
        )

        data_reader = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_header_classes.csv",
            has_header=True,
            contains_classes=True,
        )
        pipeline.optimize(
            data_reader.get_x(),
            data_reader.get_y(),
            20,
            40,
            "ParticleSwarmAlgorithm",
            "Accuracy",
        )
        predicted = pipeline.run(
            pandas.DataFrame(
                numpy.random.uniform(
                    low=0.0, high=15.0, size=(30, data_reader.get_x().shape[1])
                )
            )
        )

        self.assertEqual(predicted.shape, (30,))

        s1 = set(data_reader.get_y())
        s2 = set(predicted)
        self.assertTrue(s2.issubset(s1))
        self.assertTrue(len(s2) > 0 and len(s2) <= 2)

    def test_pipeline_export_works_fine(self):
        pipeline = Pipeline(
            feature_selection_algorithm=SelectKBest(),
            feature_transform_algorithm=Normalizer(),
            classifier=RandomForest(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            pipeline.export(os.path.join(tmp, "pipeline"))
            self.assertTrue(os.path.exists(os.path.join(tmp, "pipeline.ppln")))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))

            pipeline.export(os.path.join(tmp, "pipeline.ppln"))
            self.assertTrue(os.path.exists(os.path.join(tmp, "pipeline.ppln")))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))

    def test_pipeline_export_text_works_fine(self):
        pipeline = Pipeline(
            feature_selection_algorithm=SelectKBest(),
            feature_transform_algorithm=Normalizer(),
            classifier=RandomForest(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            pipeline.export_text(os.path.join(tmp, "pipeline"))
            self.assertTrue(os.path.exists(os.path.join(tmp, "pipeline.txt")))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))

            pipeline.export_text(os.path.join(tmp, "pipeline.txt"))
            self.assertTrue(os.path.exists(os.path.join(tmp, "pipeline.txt")))
            self.assertEqual(1, len([name for name in os.listdir(tmp)]))

        self.assertIsNotNone(pipeline.to_string())
        self.assertGreater(len(pipeline.to_string()), 0)

    def test_pipeline_setters_work_fine(self):
        pipeline = Pipeline(
            feature_selection_algorithm=SelectKBest(),
            feature_transform_algorithm=Normalizer(),
            classifier=RandomForest(),
        )

        pipeline.set_classifier(AdaBoost())
        pipeline.set_feature_selection_algorithm(SelectPercentile())
        pipeline.set_feature_transform_algorithm(StandardScaler())
        pipeline.set_selected_features_mask(numpy.ones([1, 1, 0, 0], dtype=bool))

        self.__y = numpy.array(
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
        self.__predicted = numpy.array(
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
        pipeline.set_stats(OptimizationStats(self.__predicted, self.__y))

        self.assertIsInstance(pipeline.get_classifier(), AdaBoost)
        self.assertIsInstance(
            pipeline.get_feature_selection_algorithm(), SelectPercentile
        )
        self.assertIsInstance(
            pipeline.get_feature_transform_algorithm(), StandardScaler
        )
        self.assertIsInstance(pipeline.get_stats(), OptimizationStats)
