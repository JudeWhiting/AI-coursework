#This NN can predict the sum of 2 numbers between 0 and 1
import numpy as np

def sigmoid(x): #sigmoid activation function, used for the forward pass
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x): #sigmoid derivative, used for backpropagation
    return x * (1 - x)

X = 2  #number of input nodes
HL = 10  #number of nodes in the hidden layer
Y = 1  #number of output nodes

#initialize weights for the vertices attached to the hidden layer and the output layer
HL_weights = np.random.rand(X, HL)
Y_weights = np.random.rand(HL, Y)
#initializes two 2D arrays of zeros with 1 row and multiple columns. The zeroes will change values as the network learns.
HL_biases = np.zeros((1, HL))
Y_biases = np.zeros((1, Y))

#prints the starting values of the weights and biases
print(HL_weights, '\n')
print(HL_biases, '\n')
print(Y_weights, '\n')
print(Y_biases, '\n')
print('NEURAL NETWORK CURRENTLY TRAINING, PLEASE WAIT FOR LESS THAN TEN SECONDS\n')

#creates the set of training data and the target outputs
training_data = np.array([[np.random.rand() for _ in range(2)] for _ in range(1000)])  
target_outputs = np.array([[i[0] + i[1]] for i in training_data])

#this part trains the NN
learning_rate = 0.05 #dictates how much the weights and biases will change each time
epochs = 100 #number of times the network will run through the training data

for i in range(epochs):
    for j, inputs in enumerate(training_data):
        #forward pass
        inputs = inputs.reshape((1, 2)) #converts inputs form a 1d array to a 2d array
        HL_input = np.dot(inputs, HL_weights) + HL_biases #multiplies the inputs by the first set of weights and adds the biases
        HL_output = sigmoid(HL_input) #applies the sigmoid function to the result of the line above
        Y_input = np.dot(HL_output, Y_weights) + Y_biases #multiplies the hidden layer outputs by the second set of weights and adds the second set of biases
        predicted_outputs = Y_input


        error = target_outputs[j] - predicted_outputs #calculates the error


        #backpropagation/backward pass
        output_error = error
        HL_error = np.dot(output_error, Y_weights.T) * sigmoid_der(HL_output) #calculates the error of the hidden layer

        Y_weights += np.dot(HL_output.T, output_error) * learning_rate #updates the second set of weights
        Y_biases += np.sum(output_error) * learning_rate #updates the second set of biases
        HL_weights += np.dot(inputs.T, HL_error) * learning_rate #updates the first set of weights
        HL_biases += np.sum(HL_error) * learning_rate #updates the first set of biases

#prints the final values of the weights and biases
print(HL_weights, '\n')
print(HL_biases, '\n')
print(Y_weights, '\n')
print(Y_biases, '\n')

#this part tests the trained NN
for k in [[0.1, 0.1], [0.3, 0.2], [0.4, 0.35], [0.3, 0.3], [0.8, 0.6], [0.5, 0.5]]: #test data goes here
    test_data = np.array(k)

    #forward pass
    HL_input = np.dot(test_data, HL_weights) + HL_biases
    HL_output = sigmoid(HL_input)
    Y_input = np.dot(HL_output, Y_weights) + Y_biases
    predicted_outputs = Y_input

    print('\nInput:', test_data)
    print('Predicted Output:', predicted_outputs)
    print('Target Output:', test_data[0] + test_data[1])
