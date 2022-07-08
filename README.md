# genetic-neural-network
This is a Python Library that can be used to Create and Train Neural Networks without the use of BackPropagation. This Library uses only Genetic Algorithms to train Neural Networks and thus is Easier to understand how it works

To use this library you will need:
- A custom fitness or Error Function
- An structure for your Neural Network

# Structure

The GenericNeuralNetwork() class needs 2 things to initialize
- Structure = Tuple
- Number_of_Agents = Integer

The structure is a tuple that each integer is the amount of neurons in that layer
You can have as many layers as you want, but only one Input layer and one Output Layer

Structure = (30,20,20,3)
This means that the neural network will take a list or numpy array of 30 elements each one being a number, as the input
Then it will have 2 hidden layers with 20 neurons each
And an output layer with 3 neurons

The way the neural networks created with this library train is using genetic algorithms.
So you will also have to specify the amount of "agents" that will be tested throughout the Training process.

Each agent will be a neural Network with the Structure specified. 
The minimun recommended amount of agents is 100, but depending on the problem this can vary
There is no maximun limit on how many agents you can have, but the more agents you create more memory will use

The number of agents has to be an integer number, not float

Each agent is a simple python list with the length of your Structure tuple -1, in our example (30,20,20,3)
That means that each agent will be a list with 3 elements, lets call the index of each element "i"

Each element at index i in a neural network being a numpy array of size (Structure[i]*Structure[i+1]), so the first numpy array will have 600 values between 0 and 1

# Activation Functions
The GenericNeuralNetwork() class has some activations functions ready to use:
- feedforward_relu()
- feedforward_sigmoid()
- feedforward_sigmoid_relu()
- feedforward_relu_sigmoid()

1) feedforward_relu():
  Will apply relu to each layer in the network and return the output of the network
  
2) feedforward_sigmoid():
  Will apply sigmoid to each layer in the network and return the output of the network

3) feedforward_sigmoid_relu():
  Will apply sigmoid to the input and hidden layers
  and apply relu to the output layer and return the output of the network

4) feedforward_relu_sigmoid():
  Will apply relu to the input layer and hidden layers
  and apply sigmoid to the output layer and return the output of the network

# Fitness Funciton
In order to properly train using this library you will have to create a Fitness Function that can be used to know how good or bad a neural network is
in you task

To do this you will have to create a list to contain the fitness of each neural netwok

First apply the any of the feedforward functions to each neural network and use it's predicction in task that you are trying to learn or optimize.

Then taking output of each neural network apply the fitness function that you have created.

And lastly order each neural network in decesnding or ascending order, in a way that puts the best performing neural network first in your list.






