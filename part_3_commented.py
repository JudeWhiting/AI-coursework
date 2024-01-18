#This NN can accurately guess the sum of two numbers, both in the range 0 to 0.5

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

# Define the neural network architecture
X = 2  # Number of input features
HL = 10  # Number of neurons in the hidden layer
Y = 1  # Number of output neurons

# Initialize weights for the vertices attached to the hidden layer and the output layer
HL_weights = np.random.rand(X, HL)
Y_weights = np.random.rand(HL, Y)
#initializes two 2D arrays of zeros with 1 row and multiple columns. The zeroes will change values as the network learns.
HL_biases = np.zeros((1, HL)) 
Y_biases = np.zeros((1, Y))

training_data = np.array([[np.random.rand()/2 for _ in range(2)] for _ in range(1000)])  #creates training set of inputs
target_outputs = np.array([[i[0] + i[1]] for i in training_data]) #creates training set of outputs

# this part trains the NN
learning_rate = 0.1
epochs = 100
for i in range(epochs):
    # Forward pass
    for j, inputs in enumerate (training_data):
        inputs = inputs.reshape((1,2)) #converts inputs form a 1d array to a 2d array
        HL_input = np.dot(inputs, HL_weights) + HL_biases #multiplies the inputs by the first set of weights and adds the biases
        HL_output = sigmoid(HL_input) #applies the sigmoid function to the result of the line above
        

        Y_input = np.dot(HL_output, Y_weights) + Y_biases #multiplies the hidden layer outputs by the second set of weights and adds the second set of biases
        predicted_outputs = sigmoid(Y_input) #applies the sigmoid function to the result of the line above, giving the final output
        # Calculate the error
        error = target_outputs[j] - predicted_outputs #calculates the error by finding the difference between the predicted output & target output

        # Backpropagation
        output_error = error * sigmoid_der(predicted_outputs) #calculates the error of the output layer
        hidden_layer_error = np.dot(output_error, Y_weights.T) * sigmoid_der(HL_output) #calculates the error of the hidden layer

        # Update weights and biases
        Y_weights += np.dot(HL_output.T, output_error) * learning_rate #updates the second set of weights
        Y_biases += np.sum(output_error) * learning_rate #updates the second set of biases
        
        HL_weights += np.dot(inputs.T, hidden_layer_error) * learning_rate #updates the first set of weights
        HL_biases += np.sum(hidden_layer_error) * learning_rate #updates the first set of biases

# Test the trained neural network
for k in [[0.1, 0.1], [0.3, 0.2], [0.4, 0.35], [0.3, 0.3]]: #test data goes here
    new_input_data = np.array(k)
    #Forward pass
    HL_input = np.dot(new_input_data, HL_weights) + HL_biases
    HL_output = sigmoid(HL_input)
    Y_input = np.dot(HL_output, Y_weights) + Y_biases
    predicted_outputs = sigmoid(Y_input)

    print("Input:", new_input_data)
    print("Predicted Output:", predicted_outputs)


