# This code is partly inspired by Ryan Harris-es code
# on back propagation neural networks.


import numpy as np
import json
import os


class NeuralNetwork:
    """Creates a Neural Network"""

    def __init__(self, **kwargs):
        """Initializes the network"""
        # Total number of layers, including input layer, output layer
        # and the "hidden layers"
        self.number_of_layers = None

        # Dimensions of the net, as a touple. Each element of a touple
        # tells how many neurons are in his "layer".
        # e.g. neural net with 2 inputs, 1 hidden layer with 5 neurons
        # and 3 outpus has Dimensions of (2,5,3)

        self.dimensions = tuple()

        # Synapse is an array of matrices that represent weights.
        # I prefer the term synapse because weights don't really have
        # all that much to do with neurons.

        self.synapse = list()

        # LayerInputs is an array containing matrices with inputs to layers

        self.layer_inputs = list()

        # LayerInputs is an array containing matrices with output of layers

        self.layer_outputs = list()

        # Supervised inputs and outputs is the dataset used to train the network

        self.supervised_inputs = None

        self.supervised_outputs = None

        # # # # # # # # # # # # # # # # # # #
        # Get the network (matrix) dimensions
        # # # # # # # # # # # # # # # # # # #
        print("Getting network dimensions...")

        # User can define the dimensions of the net in two ways.
        # First, simpler, is by defining number of inputs and outputs.
        if "inputs" in kwargs.keys():
            inputs = kwargs["inputs"]
            outputs = kwargs["outputs"]
            hidden_layers = kwargs.get('hidden_layers', 1)
            scale = kwargs.get("initial_weight_scale", 0.2)

            self.number_of_layers = hidden_layers + 2

            self.dimensions += (inputs,)
            for i in range(hidden_layers):
                self.dimensions += (max(inputs, outputs),)
            self.dimensions += (outputs,)

            # Second one is by defining the Dimensions themself.
        elif "dimensions" in kwargs.keys():
            self.dimensions = kwargs["dimensions"]

        print("Created a network with dimensions {}".format(self.dimensions))

        # # # # # # # # # # # # # # # # # # #
        # Create the synapse.
        # # # # # # # # # # # # # # # # # # #
        print("Creating the synapse...")

        for (out, inp) in zip(self.dimensions[:-1], self.dimensions[1:]):
            self.synapse.append(
                np.random.normal(
                    scale=0.2, size=(
                        inp, out + 1)))
        print("Created the synapse with {} elements.".format(len(self.synapse)))

    # # # # # # # # # # # # # # # # # # #
    # Transfer functions
    # # # # # # # # # # # # # # # # # # #

    def sigmoid(self, x, derivative=False):
        if derivative:
            result = self.sigmoid(x)
            return result * (1 - result)
        else:
            return 1 / (1 + np.exp(-x))

    # # # # # # # # # # # # # # # # # # #
    # Run method
    # # # # # # # # # # # # # # # # # # #

    def _run(self, input):
        """Runs the network with the providen inputs"""

        # Check if the number of inputs equals to the size of the input layer.
        if len(input) != self.dimensions[0]:
            raise ValueError(
                "The number of providen inputs is not compliant with the network you specified.")

        # Handling the first layer
        # Turn the input touple into a column matrice
        input = np.array(input)

        self.layer_outputs.append(input)
        self.layer_inputs.append(np.array(None))

        for i in range(1, len(self.dimensions)):

            previous_layer = np.vstack(
                (self.layer_outputs[-1], np.ones([1, self.layer_outputs[-1].shape[1]])))

            synapse = self.synapse[i - 1]
            layer_input = np.dot(synapse, previous_layer)
            self.layer_inputs.append(layer_input)

            layer_output = self.sigmoid(layer_input)
            self.layer_outputs.append(layer_output)

        return self.layer_outputs[-1]

    def run(self, input):
        result = self._run(input=np.array([input]).T)
        human_result = [e[0] for e in result]
        return human_result

    # # # # # # # # # # # # # # # # # # #
    # BackPropagationStep method
    # # # # # # # # # # # # # # # # # # #
    # Updates the weights for a single step.

    def backpropagation_step(self, input, target, rate=0.2):
        """Trains the network for one step."""
        target = np.array(target)

        # Run the network
        self._run(input=input)

        deltas = list()

        # Compute deltas for the final neurons
        delta_matrice = self.layer_outputs[-1] - target

        error = np.sum(delta_matrice**2)
        deltas.append(delta_matrice *
                      self.sigmoid(self.layer_inputs[-1], derivative=True))

        # Compute deltas for the "hidden layer(s)"
        for i in reversed(range(1, len(self.dimensions) - 1)):
            delta_pullback_matrice = np.dot(self.synapse[i].T, deltas[-1])
            deltas.append(
                delta_pullback_matrice[
                    :-
                    1:] *
                self.sigmoid(
                    self.layer_inputs[i],
                    derivative=True))

        # Compute deltas for the last layer, the "input" layer (neutrons)
        delta_pullback_matrice = np.dot(self.synapse[0].T, deltas[-1])
        deltas.append(
            delta_pullback_matrice[
                :-
                1:] *
            self.sigmoid(
                self.layer_outputs[0],
                derivative=True))

        # # # # # # # # # # # # # # # # # # #
        # Calculate synapse (weight) deltas
        # # # # # # # # # # # # # # # # # # #

        deltas = deltas[::-1]
        for i in range(0, len(self.dimensions) - 1):
            layer_output = np.vstack(
                (self.layer_outputs[i], np.ones([1, self.layer_outputs[i].shape[1]])))

            weight_delta = sum(layer_output[None, :, :].transpose(
                2, 0, 1) * deltas[i + 1][None, :, :].transpose(2, 1, 0))
            self.synapse[i] -= weight_delta * rate

        return error

    # # # # # # # # # # # # # # # # # # #
    # Train method
    # # # # # # # # # # # # # # # # # # #
    # method that trains the network
    # based on the provided
    # supervised dataset.

    def train(
            self,
            rate=0.02,
            margin=0.01,
            max_times=10000,
            force_convergence=False):
        """Trains the network based on the supervised dataset
        that user provided with the "Add" method."""

        if force_convergence:
            margin = 0.0
        error = margin + 1  # just so we enter the while loop down below
        count = -1

        input = self.supervised_inputs.T
        target = self.supervised_outputs.T

        while (error > margin) and ((count < max_times) or force_convergence):
            count += 1
            old_error = error
            error = self.backpropagation_step(
                input=input, target=target, rate=rate)
            # if error>old_error:
            #	print("Done/Abort - error started increasing.")
            #	break
            if count % (max_times / 1000) == 0:
                print("\rGeneration = {}, error = {}".format(count, error))
        print("Done training.")

    # # # # # # # # # # # # # # # # # # #
    # Add method
    # # # # # # # # # # # # # # # # # # #
    # method that adds user dataset as a
    # supervised dataset.

    def add(self, input, output):
        """Adds a user dataset as a supervised dataset."""

        if len(input) != self.dimensions[0]:
            raise ValueError("Number of provided inputs is not the same as the"
                             "number of inputs of the network.")
        if len(output) != self.dimensions[-1]:
            raise ValueError("Number of provided outputs is not the same as the"
                             "number of outputs of the network.")

        input = np.array([input])
        output = np.array([output])

        if self.supervised_inputs is None:
            self.supervised_inputs = input
        else:
            self.supervised_inputs = np.vstack([self.supervised_inputs, input])

        if self.supervised_outputs is None:
            self.supervised_outputs = output
        else:
            self.supervised_outputs = np.vstack(
                [self.supervised_outputs, output])

    def save(self, location):
        """Saves the synapse, NumPy array by array."""
        out_obj = dict()
        out_obj["Dimensions"] = self.dimensions
        out_obj["Synapse"] = list()
        for i in range(len(self.synapse)):
            np.savetxt("layer.out", self.synapse[i])
            layer_data = open("layer.out").read()
            layer_data = layer_data.replace("\n", " <break> ")
            out_obj["Synapse"].append(layer_data)
        os.remove("layer.out")
        file = open(location, "w")
        file.write(json.dumps(out_obj))
        file.close()

        print("Saved neural network to file <{}>.".format(location))

    def load(self, location):
        """Reads the synapse from a saved file."""
        file = open(location, "r")
        in_obj = json.loads(file.read())
        file.close()

        self.dimensions = in_obj["Dimensions"]

        for string in in_obj["Synapse"]:
            string = string.replace(" <break> ", "\n")
            temp_file = open("layer.out", "w")
            temp_file.write(string)
            temp_file.close()
            temp_file = open("layer.out", "r")

            self.synapse.append(np.loadtxt(temp_file))

        os.remove("layer.out")

        print("Loaded neural network from file <{}>.".format(location))
