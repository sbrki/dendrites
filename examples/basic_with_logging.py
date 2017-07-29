import dendrites
import logging, sys

# Create the logger
logger = logging.getLogger("examples")
# Tell the logger to "print" on stdout
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


# Create the neural network with 2 layers, input and output layer.
# The Input layer has 2 inputs, and the output layer has 3 outputs.

# Logging - IMPORTANT! logger_string must be either "dendrites", or
# "something.dendrites", see python logging docs for more info.
# (default logger_string = "dendrites")
MyNeuralNet = dendrites.NeuralNetwork(dimensions = (2,3), logger_string="examples.dendrites" )

# Add a supervised dataset -- tell the network how to behave
MyNeuralNet.add(input = [0, 1], output = [0, 1, 0])


# Train the network
MyNeuralNet.train()

# Run the network on the dataset we provided earlier, and see the results.
#print(MyNeuralNet.run(input = [0, 1]))



# Save the MyNeuralNet in a net.dat file, so we can use it later (without Training it again)
MyNeuralNet.save(location="net.dat")


# Delete MyNeuralNet from memory
del(MyNeuralNet)

# Now lets load the network from the net.dat file we created earlier.
MyLoadedNeuralNet = dendrites.NeuralNetwork()
MyLoadedNeuralNet.load(location="net.dat")


# Lets run the network one more time on the dataset we provided earlier,
# so we can be sure that it is the same network we saved earlier.
#print(MyLoadedNeuralNet.run(input = [0, 1]))

