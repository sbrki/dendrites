# This code is partly inspired by Ryan Harris-es code
# on back propagation neural networks.


import numpy as np
import json
import os


class NeuralNetwork:
    """Creates a Neural Network"""

    # Total number of layers, including input layer, output layer
    # and the "hidden layers"
    NumberOfLayers = None

    # Dimensions of the net, as a touple. Each element of a touple
    # tells how many neurons are in his "layer".
    # e.g. neural net with 2 inputs, 1 hidden layer with 5 neurons
    # and 3 outpus has Dimensions of (2,5,3)

    Dimensions = tuple()

    # Synapse is an array of matrices that represent weights.
    # I prefer the term synapse because weights don't really have
    # all that much to do with neurons.

    Synapse = list()

    # LayerInputs is an array containing matrices with inputs to layers

    LayerInputs = list()

    # LayerInputs is an array containing matrices with output of layers

    LayerOutputs = list()

    # Supervised inputs and outputs is the dataset used to train the network

    SupervisedInputs = None

    SupervisedOutputs = None

    def __init__(self, **kwargs):
        """Initializes the network"""

        # # # # # # # # # # # # # # # # # # #
        # Get the network (matrix) dimensions
        # # # # # # # # # # # # # # # # # # #
        print("Getting network dimensions...")

        # User can define the dimensions of the net in two ways.
        # First, simpler, is by defining number of inputs and outputs.
        if "inputs" in kwargs.keys():
            inputs = kwargs["inputs"]
            outputs = kwargs["outputs"]
            hiddenLayers = kwargs.get('hiddenLayers', 1)
            scale = kwargs.get("initial_weight_scale", 0.2)

            self.NumberOfLayers = hiddenLayers + 2

            self.Dimensions += (inputs,)
            for i in range(hiddenLayers):
                self.Dimensions += (max(inputs, outputs),)
            self.Dimensions += (outputs,)

            # Second one is by defining the Dimensions themself.
        elif "dimensions" in kwargs.keys():
            self.Dimensions = kwargs["dimensions"]

        print("Created a network with dimensions {}".format(self.Dimensions))

        # # # # # # # # # # # # # # # # # # #
        # Create the synapse.
        # # # # # # # # # # # # # # # # # # #
        print("Creating the synapse...")

        for (out, inp) in zip(self.Dimensions[:-1], self.Dimensions[1:]):
            self.Synapse.append(
                np.random.normal(
                    scale=0.2, size=(
                        inp, out + 1)))
        print("Created the synapse with {} elements.".format(len(self.Synapse)))

    # # # # # # # # # # # # # # # # # # #
    # Transfer functions
    # # # # # # # # # # # # # # # # # # #

    def Sigmoid(self, x, derivative=False):
        if derivative:
            result = self.Sigmoid(x)
            return result * (1 - result)
        else:
            return 1 / (1 + np.exp(-x))

    # # # # # # # # # # # # # # # # # # #
    # Run method
    # # # # # # # # # # # # # # # # # # #

    def _Run(self, input):
        """Runs the network with the providen inputs"""

        # Check if the number of inputs equals to the size of the input layer.
        if len(input) != self.Dimensions[0]:
            raise ValueError(
                "The number of providen inputs is not compliant with the network you specified.")

        # Handling the first layer
        # Turn the input touple into a column matrice
        input = np.array(input)

        self.LayerOutputs.append(input)
        self.LayerInputs.append(np.array(None))

        for i in range(1, len(self.Dimensions)):

            PreviousLayer = np.vstack(
                (self.LayerOutputs[-1], np.ones([1, self.LayerOutputs[-1].shape[1]])))

            Synapse = self.Synapse[i - 1]
            LayerInput = np.dot(Synapse, PreviousLayer)
            self.LayerInputs.append(LayerInput)

            LayerOutput = self.Sigmoid(LayerInput)
            self.LayerOutputs.append(LayerOutput)

        return self.LayerOutputs[-1]

    def Run(self, input):
        result = self._Run(input=np.array([input]).T)
        human_result = [e[0] for e in result]
        return human_result

    # # # # # # # # # # # # # # # # # # #
    # BackPropagationStep method
    # # # # # # # # # # # # # # # # # # #
    # Updates the weights for a single step.

    def BackPropagationStep(self, input, target, rate=0.2):
        """Trains the network for one step."""
        target = np.array(target)

        # Run the network
        self._Run(input=input)

        deltas = list()

        # Compute deltas for the final neurons
        delta_matrice = self.LayerOutputs[-1] - target

        error = np.sum(delta_matrice**2)
        deltas.append(delta_matrice *
                      self.Sigmoid(self.LayerInputs[-1], derivative=True))

        # Compute deltas for the "hidden layer(s)"
        for i in reversed(range(1, len(self.Dimensions) - 1)):
            delta_pullback_matrice = np.dot(self.Synapse[i].T, deltas[-1])
            deltas.append(
                delta_pullback_matrice[
                    :-
                    1:] *
                self.Sigmoid(
                    self.LayerInputs[i],
                    derivative=True))

        # Compute deltas for the last layer, the "input" layer (neutrons)
        delta_pullback_matrice = np.dot(self.Synapse[0].T, deltas[-1])
        deltas.append(
            delta_pullback_matrice[
                :-
                1:] *
            self.Sigmoid(
                self.LayerOutputs[0],
                derivative=True))

        # # # # # # # # # # # # # # # # # # #
        # Calculate synapse (weight) deltas
        # # # # # # # # # # # # # # # # # # #

        deltas = deltas[::-1]
        for i in range(0, len(self.Dimensions) - 1):
            LayerOutput = np.vstack(
                (self.LayerOutputs[i], np.ones([1, self.LayerOutputs[i].shape[1]])))

            weightDelta = sum(LayerOutput[None, :, :].transpose(
                2, 0, 1) * deltas[i + 1][None, :, :].transpose(2, 1, 0))
            self.Synapse[i] -= weightDelta * rate

        return error

    # # # # # # # # # # # # # # # # # # #
    # Train method
    # # # # # # # # # # # # # # # # # # #
    # method that trains the network
    # based on the provided
    # supervised dataset.

    def Train(
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

        input = self.SupervisedInputs.T
        target = self.SupervisedOutputs.T

        while (error > margin) and ((count < max_times) or force_convergence):
            count += 1
            old_error = error
            error = self.BackPropagationStep(
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

    def Add(self, input, output):
        """Adds a user dataset as a supervised dataset."""

        if len(input) != self.Dimensions[0]:
            raise ValueError("Number of provided inputs is not the same as the\
							 number of inputs of the network.")
        if len(output) != self.Dimensions[-1]:
            raise ValueError("Number of provided outputs is not the same as the\
							 number of outputs of the network.")

        input = np.array([input])
        output = np.array([output])

        if self.SupervisedInputs is None:
            self.SupervisedInputs = input
        else:
            self.SupervisedInputs = np.vstack([self.SupervisedInputs, input])

        if self.SupervisedOutputs is None:
            self.SupervisedOutputs = output
        else:
            self.SupervisedOutputs = np.vstack(
                [self.SupervisedOutputs, output])

    def Save(self, location):
        """Saves the synapse, NumPy array by array."""
        out_obj = dict()
        out_obj["Dimensions"] = self.Dimensions
        out_obj["Synapse"] = list()
        for i in range(len(self.Synapse)):
            np.savetxt("layer.out", self.Synapse[i])
            layer_data = open("layer.out").read()
            layer_data = layer_data.replace("\n", " <break> ")
            out_obj["Synapse"].append(layer_data)
        os.remove("layer.out")
        file = open(location, "w")
        file.write(json.dumps(out_obj))
        file.close()

        print("Saved neural network to file <{}>.".format(location))

    def Load(self, location):
        """Reads the synapse from a saved file."""
        file = open(location, "r")
        in_obj = json.loads(file.read())
        file.close()

        self.Dimensions = in_obj["Dimensions"]

        for string in in_obj["Synapse"]:
            string = string.replace(" <break> ", "\n")
            temp_file = open("layer.out", "w")
            temp_file.write(string)
            temp_file.close()
            temp_file = open("layer.out", "r")

            self.Synapse.append(np.loadtxt(temp_file))

        os.remove("layer.out")

        print("Loaded neural network from file <{}>.".format(location))
