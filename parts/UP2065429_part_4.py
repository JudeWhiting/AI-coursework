#This NN can predict the sum of 2 numbers between 0 and 1 with great accuracy
import numpy as np

def relu(x): #relu activation function, used for the forward pass
    return np.maximum(0, x)

def relu_der(x): #relu derivative, used for backpropagation
    return np.where(x > 0, 1, 0)

X = 2  #number of input nodes
HL1 = 4  #number of nodes in the first hidden layer
HL2 = 4  #number of nodes in the second hidden layer
Y = 1  #number of output nodes

#initialize weights for the vertices attached to the hidden layers and the output layer
HL1_weights = np.random.rand(X, HL1)
HL2_weights = np.random.rand(HL1, HL2)
Y_weights = np.random.rand(HL2, Y)
#initializes three 2D arrays of zeros with 1 row and multiple columns. The zeroes will change values as the network learns.
HL1_biases = np.zeros((1, HL1))
HL2_biases = np.zeros((1, HL2))
Y_biases = np.zeros((1, Y))

#prints the starting values of the weights and biases
print(HL1_weights, '\n')
print(HL1_biases, '\n')
print(HL2_weights, '\n')
print(HL2_biases, '\n')
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
        HL1_input = np.dot(inputs, HL1_weights) + HL1_biases #multiplies the inputs by the first set of weights and adds the biases
        HL1_output = relu(HL1_input) #applies the reLU function to the result of the line above

        HL2_input = np.dot(HL1_output, HL2_weights) + HL2_biases #multiplies the inputs by the second set of weights and adds the biases
        HL2_output = relu(HL2_input) #applies the reLU function to the result of the line above

        Y_input = np.dot(HL2_output, Y_weights) + Y_biases #multiplies the hidden layer outputs by the third set of weights and adds the third set of biases
        predicted_outputs = Y_input 
        
        
        error = target_outputs[j] - predicted_outputs #calculates the error


        #backpropagation/backward pass
        output_error = error
        HL2_error = np.dot(output_error, Y_weights.T) * relu_der(HL2_output) #calculates the error of the second hidden layer
        HL1_error = np.dot(HL2_error, HL2_weights.T) * relu_der(HL1_output) #calculates the error of the first hidden layer

        Y_weights += np.dot(HL2_output.T, output_error) * learning_rate #updates the third set of weights
        Y_biases += np.sum(output_error) * learning_rate #updates the third set of biases
        HL2_weights += np.dot(HL1_output.T, HL2_error) * learning_rate #updates the second set of weights
        HL2_biases += np.sum(HL2_error) * learning_rate #updates the second set of biases
        HL1_weights += np.dot(inputs.T, HL1_error) * learning_rate #updates the first set of weights
        HL1_biases += np.sum(HL1_error) * learning_rate #updates the first set of biases

#prints the final values of the weights and biases
print(HL1_weights, '\n')
print(HL1_biases, '\n')
print(HL2_weights, '\n')
print(HL2_biases, '\n')
print(Y_weights, '\n')
print(Y_biases, '\n')

#this part tests the trained NN
for k in [[0.1, 0.1], [0.3, 0.2], [0.4, 0.35], [0.3, 0.3], [0.8, 0.6], [0.5, 0.5]]: #test data goes here
    test_data = np.array(k)

    #forward pass
    HL1_input = np.dot(test_data, HL1_weights) + HL1_biases
    HL1_output = relu(HL1_input)
    HL2_input = np.dot(HL1_output, HL2_weights) + HL2_biases
    HL2_output = relu(HL2_input)
    Y_input = np.dot(HL2_output, Y_weights) + Y_biases
    predicted_outputs = Y_input

    print('\nInput:', test_data)
    print('Predicted Output:', predicted_outputs)
    print('Target Output:', test_data[0] + test_data[1])
