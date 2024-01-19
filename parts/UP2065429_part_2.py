import numpy as np


def sigmoid(x): #sigmoid activation function
    return 1.0/(1+np.exp(-x))
    

def perceptron(X,Y):

    weights = np.random.rand(2) #initializes 2 random weights between zero and 1
    bias = np.zeros(1) #initializes the bias as zero
    learning_rate = 0.05

    print('starting weights: ',weights) #print starting weights and bias
    print('starting bias: ',bias)
    
    for count in range(10000): #adjusts the perceptrons weights and values 10000 times

        for i in range(len(X)):

            output = sigmoid(np.dot(X[i], weights) + bias) #calculates weighted sum and applies the sigmoid function

            error = Y[i] - output #update weights and the bias based on the error
            weights += learning_rate * error * X[i]
            bias += learning_rate * error

    print('trained weights: ',weights) #print trained weights and bias
    print('trained bias: ',bias)

    predictions = [sigmoid(np.dot(x, weights) + bias) for x in X] #predict results

    print('inputs: ',[list(b) for b in X]) #print inputs, predictions corresponding to those inputs, and the truth table for those inputs
    print('predictions:',[list(c) for c in predictions]) 
    print('truth table: ',list(Y))

data = np.array([[0,0],[0,1],[1,0],[1,1]]) #training data
OR_truth_table = np.array([0,1,1,1]) #truth table for OR
XOR_truth_table = np.array([0,1,1,0]) #truth table for XOR
print('Input and results of a perceptron trained to solve the OR problem:')
perceptron(data,OR_truth_table)
print('\nInput and results of a perceptron trained to solve the XOR problem:')
perceptron(data,XOR_truth_table)
print('\n based on the predicted outputs of each perceptron, we can see that the perceptron trained to solve the OR problem is very confident with which output node should be activated, whereas the perceptron trained to solve the XOR problem has no idea. This is because the XOR problem is not linearly separable, whereas the OR problem is.')