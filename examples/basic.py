import dendrites

# Create the neural network with 2 layers, input and output layer.
# The Input layer has 2 inputs, and the output layer has 3 outputs.

MyNeuralNet = dendrites.NeuralNetwork(dimensions = (2,3) )


# Add a supervised dataset -- tell the network how to behave
MyNeuralNet.Add(input = [0,1], output = [0,1,0])


# Train the network
MyNeuralNet.Train()

# Run the network on the dataset we provided earlier, and see the results.
print(MyNeuralNet.Run(input = [0,1]))



# Save the MyNeuralNet in a net.dat file, so we can use it later (without Training it again)
MyNeuralNet.Save(location="net.dat")


# Delete MyNeuralNet from memory
del(MyNeuralNet)

# Now lets load the network from the net.dat file we created earlier.
MyLoadedNeuralNet = dendrites.NeuralNetwork()
MyLoadedNeuralNet.Load(location="net.dat")


# Lets run the network one more time on the dataset we provided earlier,
# so we can be sure that it is the same network we saved earlier.
print(MyLoadedNeuralNet.Run(input = [0,1]))


