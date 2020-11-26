import numpy as np

class NeuralNetwork():

    def __init__(self): # our initialize function for our NN class, always written first, takes in the standard variable self
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1 # initialize our synaptic weights between 0 and 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs) #think is another function we define later

            error = training_outputs - output # all explained in nn_framework.py
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output)) # matrix multiplication
            self.synaptic_weights += adjustments # back propagation

    def think(self, inputs): 

        inputs = inputs.astype(float) # since inputs (IN THIS CASE) are integers and synaptic_weights are floats, you must convert int to float to take the dot product correctly
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights)) # takes the sigmoid of the dot product of the inputs multiplied by the weights

        return output

# next, we must make this class useable in the command line
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
    
    training_outputs = np.array([[0,1,1,0]]).T #same inputs and outputs from our previous framework, will be replaced by picking from database later on

    training_iterations = 10000

    neural_network.train(training_inputs, training_outputs, training_iterations)

    print("Synaptic Weights After Training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: ")) #These inputs are for testing, we will replace this with our testing data

    print("New situation: input data= ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))