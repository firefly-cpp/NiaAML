from niaaml.classifiers import ClassifierFactory
from niaaml.preprocessing.feature_selection import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithmFactory
from niaaml.fitness import FitnessFactory

"""
In this example, we show how to use all of the implemented factories to create new object instances using their class names. You may also
import and instantiate objects directly, but it more convenient to use factories in some cases.
"""

# instantiate all possible factories
classifier_factory = ClassifierFactory()
fsa_factory = FeatureSelectionAlgorithmFactory()
fta_factory = FeatureTransformAlgorithmFactory()
f = FitnessFactory()

# get an instance of the MultiLayerPerceptron class
mlp = classifier_factory.get_result('MultiLayerPerceptron')

# get an instance of the ParticleSwarmOptimization class
pso = fsa_factory.get_result('ParticleSwarmOptimization')

# get an instance of the Normalizer class
normalizer = fta_factory.get_result('Normalizer')

# get an instace of the Precision class
precision = f.get_result('Precision')

# variables mlp, pso, normalizer and precision contain instances of the classes with the passed names