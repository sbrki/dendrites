import sys
import unittest
sys.path.append("..")


import dendrites



class TestNeuralNetworkObjectInitialization(unittest.TestCase):
    def test_create_neural_network_object(self):

        neural_network = dendrites.NeuralNetwork()

        self.assertIsNotNone(neural_network)

    def test_create_neural_network_object_with_dimensions_param(self):

        test_dimensions = ( 2, 10, 13, 15 )

        neural_network = dendrites.NeuralNetwork( dimensions = test_dimensions )

        self.assertEqual( test_dimensions, neural_network.dimensions)

    def test_create_neural_network_object_with_inputs_outputs_params(self):

        test_inputs = 10
        test_outputs = 10

        neural_network = dendrites.NeuralNetwork( inputs = test_inputs,
                                                  outputs = test_outputs )

        self.assertEqual(test_inputs, neural_network.dimensions[0])
        self.assertEqual(test_outputs, neural_network.dimensions[-1])



class TestAddingSupervisedDataset(unittest.TestCase):
    def setUp(self):
        self.neural_network = dendrites.NeuralNetwork( dimensions = (3,4,2) )


    def test_adding_valid_dataset(self):
        self.neural_network.add( input = [1, 1, 1], output = [1, 1])

    def test_adding_invalid_dataset_with_bad_number_of_inputs(self):
        with self.assertRaises(Exception) as context:
         self.neural_network.add( input = [1, 1, 1, 1], output = [1, 1])


    def test_adding_invalid_dataset_with_bad_number_of_outputs(self):
        with self.assertRaises(Exception) as context:
            self.neural_network.add( input = [1, 1, 1], output = [1, 1, 1])



class TestTraining(unittest.TestCase):
    def setUp(self):
        self.neural_network = dendrites.NeuralNetwork( dimensions = (3,4,2) )
        self.neural_network.add( input = [1, 1, 1], output = [1, 1] )

    def test_train_method(self):
        self.neural_network.train()


class TestRunning(unittest.TestCase):
    def setUp(self):
        self.test_inputs = [1, 0, 1]
        self.test_outputs = [1, 0]

        self.neural_network = dendrites.NeuralNetwork( dimensions = (3,4,2) )
        self.neural_network.add( input = self.test_inputs, output = self.test_outputs )
        self.neural_network.train()

    def test_run(self):
        network_output = self.neural_network.run(input = self.test_inputs)

        for index, expected_result in enumerate(self.test_outputs):
            self.assertEqual(expected_result, round(network_output[index]))





if __name__ == "__main__":
    unittest.main()