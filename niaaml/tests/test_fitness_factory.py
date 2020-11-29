from unittest import TestCase
from niaaml.fitness import FitnessFactory, FitnessFunction

class FitnessFactoryTestCase(TestCase):
    def setUp(self):
        self.__factory = FitnessFactory()
    
    def test_get_result_works_fine(self):
        for entry in self.__factory._entities:
            instance = self.__factory.get_result(entry)
            self.assertIsNotNone(instance)
            self.assertIsInstance(instance, FitnessFunction)
        
        with self.assertRaises(TypeError):
            self.__factory.get_result('non_existent_name')