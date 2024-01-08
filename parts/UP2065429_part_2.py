import numpy as np


def sigmoid(x): #sigmoid activation function
    return 1.0/(1+np.exp(-x))
    

def OR():

    X = np.array([[0,0],[0,1],[1,0],[1,1]]) #training data
    Y = np.array([0,1,1,1])  #truth table
    weights = np.random.rand(2) - 0.5  #initialize random weights, between -0.5 and 0.5
    bias = np.random.rand(1) - 0.5  #initialize a random bias, between -0.5 and 0.5
    learning_rate = 0.1

    print('starting weights: ',weights) #print starting weights and bias
    print('starting bias: ',bias)
    
    for count in range(1000): #adjusts weights and bias 1000 times

        for i in range(len(X)):

            output = sigmoid(np.dot(X[i], weights) + bias) #calculates weighted sum and applies the sigmoid function

            #update weights and the bias based on the error
            error = Y[i] - output
            weights += learning_rate * error * X[i]
            bias += learning_rate * error

    print('trained weights: ',weights) #print trained weights and bias
    print('trained bias: ',bias)

    predictions = [sigmoid(np.dot(x, weights) + bias) for x in X] #predict results

    print('inputs: ',[list(b) for b in X]) #print inputs, predictions corresponding to those inputs, and the truth table for those inputs
    print('predictions:',[list(c) for c in predictions]) 
    print('truth table: ',list(Y))



def XOR():
    X = np.array([[0,0],[0,1],[1,0],[1,1]]) #training data
    Y = np.array([0,1,1,0])  #truth table
    weights = np.random.rand(2) - 0.5  #initialize random weights, between -0.5 and 0.5
    bias = np.random.rand(1) - 0.5  #initialize a random bias, between -0.5 and 0.5
    learning_rate = 0.1

    print('starting weights: ',weights) #print starting weights and bias
    print('starting bias: ',bias)
    
    for count in range(1000): #adjusts weights and bias 1000 times

        for i in range(len(X)):

            output = sigmoid(np.dot(X[i], weights) + bias) #calculates weighted sum and applies the sigmoid function

            #update weights and the bias based on the error
            error = Y[i] - output
            weights += learning_rate * error * X[i]
            bias += learning_rate * error

    print('trained weights: ',weights) #print trained weights and bias
    print('trained bias: ',bias)

    predictions = [sigmoid(np.dot(x, weights) + bias) for x in X] #predict results

    print('inputs: ',[list(b) for b in X]) #print inputs, predictions corresponding to those inputs, and the truth table for those inputs
    print('predictions:',[list(c) for c in predictions]) 
    print('truth table: ',list(Y))


print('Input and results of a perceptron trained to solve the OR problem:')
OR()
print('\nInput and results of a perceptron trained to solve the XOR problem:')
XOR()