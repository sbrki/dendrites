# dendrites

### Neural networks for humans
---
### Concept:
Dendrites aims to be a straightfoward and
very easy to use neural network tool in/for Python3.

With dendrites, **performance is not in the main focus** (yet), but it is not 
being ignored either.

The main purpose of dendrites is to be a **ultra easy to use** neural network.

**Note: This repo is heavy work in progress, but it fundamentally
works. Every help is welcome**

---

### Requirements:

Dendrites depends on only one non standard library, **numpy**.

If you don't have numpy already installed, run: `sudo pip3 install numpy`.

---

## Manual
### Creating a NeuralNetwork object

```python
import dendrites

my_neural_net = dendrites.NeuralNetwork( dimensions = (2,3) )
```

This code creates a NeuralNetwork instance.
With the `dimensions` argument, we instruct our network to have
2 layers.

The first layer has 2 inputs, and the second layer has 3 outputs.

Therefore, our network has 2 inputs and 3 outputs.


If we wanted a third, *"middle"* layer with 5 nodes, we would write:
```python
import dendrites

my_neural_net = dendrites.NeuralNetwork( dimensions = (2,5,3) )
```

--

### Creating a NeuralNetwork object with only the number of inputs and outputs

You can also create a NeralNetwork instance only by telling it how many 
inputs and outputs to have. This is even more simple than the previous method.

```python
import dendrites

my_neural_net = dendrites.NeuralNetwork( inputs = 2, outputs = 3 )
```

The above code creates a NeuralNetwork instance with 2 inputs and 3 outputs.

The neural network tries to predict how many hidden layers should it have. (*Heavy work in progress*)

-- 

###Adding a supervised dataset

In order to tell our neural network how to behave when a certain input is brought to the input layer, we add what we call *a supervised dataset*.

In this example, lets use the NeuralNetwork instance with 2 inputs and 3 outputs we created earlier.

Lets instruct it to bring `0 , 1 , 0` to output when `0 , 1` is brought to input.

```python
my_neural_net.add( input = [0,1], output = [0,1,0] )
```

--

###Training the network

Lets say we added a supervised dataset to our network. But we can't run our network yet,

we have to train it first. Training is the part in which the network *learns*.

```python
my_neural_net.train()
```

The above code instructs our network to train itself.

**By default** the network trains until the difference between the way
it behaves and the way we specified it to behave (with providing supervised dataset)
is less than **1%**.


If we want, for some reason, for our network to train itself until the margin (error) is
under **10%**, we would write:
```python
my_neural_net.train( margin = 0.1 )
```


Also, we could also want for out network to train until the margin (error) is virtually zero.
Then, we would write:

```python
my_neural_net.train( force_convergence = True )
```

This is not recommended, as it can take a really long time for the network to train this way.

--

###Running the network

By running:
```python
output = my_neural_net.run( input = [0,1] )
print( output )
```

we bring `0 , 1` to the input of our network. The network output is also a list, with the dimensions we specified earlier.


*Note*: you can Run your network before you have trained it. However, the network will give out a random result, because every time you create a network, the synapse nodes are initiated with random weights.

--

###Saving the network to a file

When we trained our network, we can save it to a file. This way if we wanted to use our
network again, can read it from the file and we wont need to Train it again.

```python
my_neural_net.save( location = "net.dat" )
```

Saves the neural network MyNeuralNet to a local file `net.dat`.


--


###Reading the network from a file

```python
my_loaded_neural_net = dendrites.NeuralNetwork()
my_loaded_neural_net.load( location = "net.dat" )
```

Creates a new neural network MyLoadedNeuralNet and loads it from the file `net.dat`. We can now Run MyLoadedNeuralNet and it will behave the same way the previous network (we saved to `net.dat` file) did.
