import numpy as np

def neuron(inputs, weights, bias):
    return 1.0/(1+np.exp(-(np.dot(inputs, weights) + bias)))

X = 2
HL = [8]
Y = 1
L = [X] + HL + [Y]

weights = np.random.rand(HL[0] , X) - 0.5
biases = np.random.rand(HL[0]) - 0.5

def train_neuron(inputs, target_outputs, W, B, epochs=100, learning_rate = 0.1):
    for epoch in range(epochs):
        for i in range(len(inputs)):
            error = target_outputs[i] - neuron(inputs[i], W, B)
            W += learning_rate * error * inputs[i]
            B += learning_rate * error
    return W, B



training_inputs = np.array([[np.random.rand()/2 for _ in range(2)] for _ in range(1000)])  #this creates a training set of inputs
targets = np.array([[i[0] * i[1]] for i in training_inputs])     #this creates a training set of outputs

data = np.array([[0,0],[0,1],[1,0],[1,1]]) #training data
OR_truth_table = np.array([0,1,1,1]) #truth table for OR
for i in range(HL[0]):   
    weights[i], biases[i] = train_neuron(data, OR_truth_table, weights[i], biases[i])


