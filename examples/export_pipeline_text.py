from niaaml import Pipeline
from niaaml.classifiers import AdaBoost
from niaaml.preprocessing.feature_selection import SelectKBest
from niaaml.preprocessing.feature_transform import Normalizer

pipeline = Pipeline(
    data=None,
    feature_selection_algorithm=SelectKBest(),
    feature_transform_algorithm=Normalizer(),
    classifier=AdaBoost()
)
pipeline.export_text('exported_pipeline.txt')