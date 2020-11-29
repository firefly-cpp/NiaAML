from niaaml.classifiers import ClassifierFactory
from niaaml.preprocessing.feature_selection import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithmFactory
from niaaml.fitness import FitnessFactory

classifier_factory = ClassifierFactory()
fsa_factory = FeatureSelectionAlgorithmFactory()
fta_factory = FeatureTransformAlgorithmFactory()
f = FitnessFactory()

mlp = classifier_factory.get_result('MultiLayerPerceptron')
pso = fsa_factory.get_result('ParticleSwarmOptimization')
normalizer = fta_factory.get_result('Normalizer')
precision = f.get_result('Precision')

# variables mlp, pso, normalizer and precision contain instances of the classes with the passed names