import numpy  as np

# https://www.youtube.com/watch?v=kft1AJ9WVDk

# Our neural network will have:
# x input layers
# 1 hidden neuron
# outputs that are any number


def sigmoid(x): #this is our normalizing function we do on each neuron to get a value between 0 and 1. All neurons need to contain an activation between 0 and 1
    return 1 / (1 + np.exp(-x)) # sigmoid function = 1/(1+e^-x)

def sigmoid_derivative(x): #sigmoid derivative is used for error calculation and correction for our training process
    return x * (1 - x) # sigmoid derivative = x(1-x)

#next, we define our training inputs. This will be a matrix of all the input values that we have collected
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
# Training outputs are in order with the inputs. (in this training example, a first  value of 1 means an output of 1)
training_outputs = np.array([[0,1,1,0]]).T # The .T transposes the row into a 4x1 matrix. In our case, these would be the individual closing stock price values

# seed random numbers for our weights
np.random.seed(1)

# create a 3x1 matrix because in this case we have 3 inputs and 1 output.
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random Starting synaptic weights: ")
print(synaptic_weights)

# main loop
for iteration in range(20000): #20000 iterations to change our weights
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights)) # finds the dot product (multiplying matrices input_layr and our synaptic_weights) then we put a sigmoid function to gget a value between 0 and 1

    # TRAINING PROCESS:
    # 1. take the inputs from the training example and put them through our formula to get the neuron's output
    # 2. Calculate the error, which is difference between the output we got and the actual output
    # 3. Depending on the severeness of the error, adjust the weights accordingly
    # 4. repeat the process 20000 times

    # To know which values to adjust the weights, we use the error weighted derivative:
    # Adjust weights by = error * input * gradient of the sigmoid(output)
    # gradient of sigmoid = x * (1-x)
    # Error = output - actual output
    # input = either 1 or 0 (or between), so if the input is 0, the weights aren't adjusted 
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments) # dot product of the matrices input_layer and adjustments. the .T transposes input_layer so that it multiplies correctly.

print("Synaptic Weights after training")
print(synaptic_weights)
print("Outputs after training: ")
print(outputs) #ideally, the outputs should be close to the training_outputs, but we haven't done any training yet.
 

