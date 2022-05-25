from unittest import TestCase
import niaaml.classifiers as c
from niaaml.data import CSVDataReader
import os
from sklearn.model_selection import train_test_split


class ClassifierTestCase(TestCase):
    def setUp(self):
        self.__data = CSVDataReader(
            src=os.path.dirname(os.path.abspath(__file__))
            + "/tests_files/dataset_header_classes.csv",
            has_header=True,
            contains_classes=True,
        )
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(
            self.__data.get_x(), self.__data.get_y(), test_size=0.2
        )

    def test_adaboost_works_fine(self):
        algo = c.AdaBoost()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_bagging_works_fine(self):
        algo = c.Bagging()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_ert_works_fine(self):
        algo = c.ExtremelyRandomizedTrees()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_lsvc_works_fine(self):
        algo = c.LinearSVC()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_mlp_works_fine(self):
        algo = c.MultiLayerPerceptron()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_rf_works_fine(self):
        algo = c.RandomForest()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_dt_works_fine(self):
        algo = c.DecisionTree()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_kn_works_fine(self):
        algo = c.KNeighbors()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_gp_works_fine(self):
        algo = c.GaussianProcess()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_gnb_works_fine(self):
        algo = c.GaussianNB()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)

    def test_qda_works_fine(self):
        algo = c.QuadraticDiscriminantAnalysis()
        algo.fit(self.__x_train, self.__y_train)
        predictions = algo.predict(self.__x_test)
        self.assertEqual(predictions.shape, self.__y_test.shape)
