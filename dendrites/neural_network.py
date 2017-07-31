# This code is partly inspired by Ryan Harris-es code
# on back propagation neural networks.


import numpy as np
import io
import json
import logging
from .log_format import LogFormat


# Initialize logging
logging.getLogger(__name__).addHandler(
    logging.NullHandler
)


class NeuralNetwork:
    """Creates a Neural Network"""

    def __init__(self, inputs=None, outputs=None, hidden_layers=None, dimensions=None):
        """Initializes the network"""

        logging.debug(LogFormat.INIT_START)

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

        # Get the network (matrix) dimensions
        logging.debug("{}Getting network dimensions".format( LogFormat.INIT))

        # User can define the dimensions of the net in two ways.
        # First, simpler, is by defining number of inputs and outputs.
        if inputs is not None:
            if hidden_layers is None:
                hidden_layers = 1

            self.number_of_layers = hidden_layers + 2

            self.number_of_layers = hidden_layers + 2

            self.dimensions += (inputs,)
            for i in range(hidden_layers):
                self.dimensions += (max(inputs, outputs),)
            self.dimensions += (outputs,)

            # Second one is by defining the Dimensions themself.

        elif dimensions is not None:
            self.dimensions = dimensions

        logging.debug(
            "{}Created a network with dimensions: {}".format(
                LogFormat.INIT,
                self.dimensions
            )
        )

        # Create the synapse.
        logging.debug(
            "{}Creating the synapse...".format(LogFormat.INIT)
        )

        for (out, inp) in zip(self.dimensions[:-1], self.dimensions[1:]):
            self.synapse.append(np.random.normal(scale=0.2, size=(inp, out + 1)))

        logging.debug(
            "{}Created the synapse with {} elements.".format(
                LogFormat.INIT,
                len(self.synapse)
            )
        )
        logging.info("{}".format(
            LogFormat.INIT_DONE
            )
        )

    def sigmoid(self, x, derivative=False):
        """
        Sigmoid function.

        :param x: input
        :type x: float
        :param derivative: Return derivative instead if true.
        :type derivative: bool
        :return: result
        :rtype: float
        """
        if derivative:
            result = self.sigmoid(x)
            return result * (1 - result)
        else:
            return 1 / (1 + np.exp(-x))

    def _run(self, input):
        """
        Runs the network against the provided input

        :param input: Input to run the network against
        :type input: numpy.ndarray
        :return: Result of running the network against the input
        :rtype: numpy.ndarray
        """

        # Check if the number of inputs equals to the size of the input layer.
        if len(input) != self.dimensions[0]:
            raise ValueError("{} inputs were provided, but the network requires {}.".format(len(input),
                                                                                            self.dimensions[0]))

        # Handling the first layer
        # Turn the input tuple into a numpy array
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
        """
        Runs the network using human-readable input and generates human-readable output.

        :param input: Input to provide to the network.
        :type input: list
        :return: Result of running the network against the input
        :rtype: list
        """
        result = self._run(input=np.array([input]).T)
        human_result = [e[0] for e in result]
        return human_result

    def backpropagation_step(self, input, target, rate=0.2):
        """
        Trains the network.

        :param input: supervised input
        :type input: numpy.ndarray
        :param target: targeted output
        :type target: numpy.ndarray
        :param rate: rate for calculating synapse (weight) deltas
        :type rate: float
        :return: Degree of error after training.
        :rtype: float
        """
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
            deltas.append(delta_pullback_matrice[:-1:] *
                          self.sigmoid(self.layer_inputs[i], derivative=True))

        # Compute deltas for the last layer, the "input" layer (neutrons)
        delta_pullback_matrice = np.dot(self.synapse[0].T, deltas[-1])
        deltas.append(delta_pullback_matrice[:-1:] *
                      self.sigmoid(self.layer_outputs[0], derivative=True))

        # Calculate synapse (weight) deltas
        deltas = deltas[::-1]
        for i in range(0, len(self.dimensions) - 1):
            layer_output = np.vstack(
                (self.layer_outputs[i], np.ones([1, self.layer_outputs[i].shape[1]])))

            weight_delta = sum(layer_output[None, :, :].transpose(
                2, 0, 1) * deltas[i + 1][None, :, :].transpose(2, 1, 0))
            self.synapse[i] -= weight_delta * rate

        return error

    def train(self, rate=0.02, margin=0.01, max_times=10000, force_convergence=False):
        """
        Trains the network until the specified margin is reached, or after a specified number of attempts.

        :param rate: Rate for calculating synapse (weight) deltas
        :type rate: float
        :param margin: Desired degree of error for the network
        :type margin: float
        :param max_times: Maximum number of times the network will be trained
        :type max_times: int
        :param force_convergence: Require network to train until margin is 0.0;  Overrides margin if specified
        :type force_convergence: bool
        """

        logging.info("{}".format(LogFormat.TRAINING_START))

        if force_convergence:
            margin = 0.0
        error = margin + 1  # just so we enter the while loop down below
        count = -1

        input = self.supervised_inputs.T
        target = self.supervised_outputs.T

        while (error > margin) and ((count < max_times) or force_convergence):
            count += 1
            error = self.backpropagation_step(
                input=input, target=target, rate=rate)
            if count % (max_times / 1000) == 0:
                logging.debug("{}Generation = {}{}{}, error = {}{}{}".format(
                    LogFormat.TRAINING,
                    LogFormat.OKBLUE,count,LogFormat.ENDC,
                    LogFormat.WARNING, error, LogFormat.ENDC
                    )
                )

        logging.info("{}, in {} generations, with a final error of {}".format(LogFormat.TRAINING_DONE, count, error))

    def add(self, input, output):
        """
        Adds a user dataset as a supervised dataset.

        :param input: Input of dataset.
        :type input: list
        :param output: Output of dataset.
        :type output: list
        """

        if len(input) != self.dimensions[0]:
            raise ValueError("{} inputs were provided, but the network requires {}.".format(len(input),
                                                                                            self.dimensions[0]))
        if len(output) != self.dimensions[-1]:
            raise ValueError("{} outputs were provided, but the network requires {}.".format(len(output),
                                                                                             self.dimensions[-1]))

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
        """
        Saves the synapse, NumPy array by array.

        :param location: File name to save the synapse to
        :type location: str

        """
        out_obj = dict()
        out_obj["Dimensions"] = self.dimensions
        out_obj["Synapse"] = list()

        for s in self.synapse:
            temp = io.BytesIO()
            np.savetxt(temp, s, newline=" <break> ")
            temp.seek(0)
            out_obj["Synapse"].append(temp.read()[:-9].decode())

        with open(location, "w") as out_file:
            out_file.write(json.dumps(out_obj))

        logging.info("{}Saved neural network to file <{}>.".format(LogFormat.SAVE, location))

    def load(self, location):
        """
        Read the network from a file.

        :param location: The path to the file.
        :type location: str
        """
        with open(location, "r") as dat_file:
            in_obj = json.loads(dat_file.read())

        self.dimensions = in_obj["Dimensions"]

        for string in in_obj["Synapse"]:
            temp_file = io.StringIO(string.replace(" <break> ", "\n"))
            temp_file.seek(0)
            self.synapse.append(np.loadtxt(temp_file))

        logging.info("{}Loaded neural network from file <{}>".format(
            LogFormat.LOAD,
            location
            )
        )
