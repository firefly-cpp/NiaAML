from niaaml.classifiers import ClassifierFactory
from niaaml.preprocessing.feature_selection import FeatureSelectionAlgorithmFactory
from niaaml.preprocessing.feature_transform import FeatureTransformAlgorithmFactory
from niaaml.fitness import FitnessFactory
from niaaml.preprocessing.encoding import EncoderFactory
from niaaml.preprocessing.imputation import ImputerFactory

"""
This example presents how to use all of the implemented factories to create new object instances using their class names. You may also
import and instantiate objects directly, but it is more convenient to use factories in some cases.
"""

# instantiate all possible factories
classifier_factory = ClassifierFactory()
fsa_factory = FeatureSelectionAlgorithmFactory()
fta_factory = FeatureTransformAlgorithmFactory()
f_factory = FitnessFactory()
e_factory = EncoderFactory()
i_factory = ImputerFactory()

# get an instance of the MultiLayerPerceptron class
mlp = classifier_factory.get_result("MultiLayerPerceptron")

# get an instance of the ParticleSwarmOptimization class
pso = fsa_factory.get_result("ParticleSwarmOptimization")

# get an instance of the Normalizer class
normalizer = fta_factory.get_result("Normalizer")

# get an instance of the Precision class
precision = f_factory.get_result("Precision")

# get an instance of the OneHotEncoder class
ohe = e_factory.get_result("OneHotEncoder")

# get an instance of the SimpleImputer class
imp = i_factory.get_result("SimpleImputer")

# variables mlp, pso, normalizer, precision, ohe and imp contain instances of the classes with the passed names
