import cmd
import sys

from niaaml.data.csv_data_reader import CSVDataReader
from niaaml.classifiers.random_forest import RandomForest
from niaaml.pipeline import Pipeline
from niaaml.pipeline_optimizer import PipelineOptimizer

class NiaAMLShell(cmd.Cmd):
    intro = 'Welcome to the NiaAML shell. Type help or ? to list commands.\n'
    prompt = '(niaaml) '

    # Initialize components
    def __init__(self):
        super(NiaAMLShell, self).__init__()
        self.data = None
        self.classifier = None
        self.pipeline = None

    def do_load_data(self, arg):
        'Load data for an experiment: load_data <file_path> <contains_classes> <has_header>'
        args = arg.split()
        if len(args) != 3:
            print("Usage: load_data <file_path> <contains_classes> <has_header>")
            return

        file_path, contains_classes, has_header = args
        contains_classes = contains_classes.lower() == 'true'
        has_header = has_header.lower() == 'true'

        try:
            data_reader = CSVDataReader(src=file_path, contains_classes=contains_classes, has_header=has_header)
            self.data = data_reader
            print(f"Data loaded from {file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def do_setup_classifier(self, arg):
        'Set up a classifier: setup_classifier <classifier_name> [<param1>=<value1> <param2>=<value2> ...]'
        args = arg.split()
        if len(args) < 1:
            print("Usage: setup_classifier <classifier_name> [<param1>=<value1> <param2>=<value2> ...]")
            return

        classifier_name = args[0].lower()
        params = dict(arg.split('=') for arg in args[1:])

        try:
            if classifier_name == 'randomforest':
                self.classifier = RandomForest()
                self.classifier.set_parameters(**params)
                print("RandomForest classifier set up with parameters:", params)
            else:
                print(f"Classifier '{classifier_name}' is not supported.")
        except Exception as e:
            print(f"Error setting up classifier: {e}")

    def do_optimize_pipeline(self, arg):
        'Optimize a pipeline: optimize_pipeline'
        if self.data is None or self.classifier is None:
            print("Error: Please load data and set up a classifier before running an experiment.")
            return

        try:
            # Create a pipeline with the loaded data and classifier
            self.pipeline = Pipeline(classifier=self.classifier, data=self.data)
            
            # Fit the pipeline with the data
            self.pipeline.optimize(
                    self.data.get_x(),
                    self.data.get_y(),
                    20,
                    40,
                    "ParticleSwarmAlgorithm",
                    "Accuracy",
            )
            
            print("Experiment run successfully.")
        except Exception as e:
            print(f"Error running experiment: {e}")

    def do_exit(self, arg):
        'Exit the NiaAML shell.'
        print("Goodbye")
        return True

    # Helper function to parse parameters - this would need to be implemented based on NiaAML's requirements
    def parse_parameters(self, arg):
        # Parse and return parameters
        pass

if __name__ == '__main__':
    NiaAMLShell().cmdloop()

