import numpy as np

def perceptron(input_data, weights, bias):
    # Calculate the weighted sum
    weighted_sum = np.dot(input_data, weights) + bias
    
    # Apply the step function as the activation function
    output = 1 if weighted_sum > 0 else 0
    
    return output

def test_perceptron_logic_gate(logic_gate):
    # Define training data for OR and XOR problems
    input_data_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_or = np.array([0, 1, 1, 1])  # OR gate truth table
    
    input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_xor = np.array([0, 1, 1, 0])  # XOR gate truth table
    
    # Initialize weights and bias
    weights = np.array([1, 1])
    bias = -0.5
    
    # Test the perceptron on the specified logic gate problem
    print(f"Testing perceptron for {logic_gate} problem:")
    for i in range(len(input_data_or)):
        result = perceptron(input_data_or[i], weights, bias) if logic_gate == 'OR' else perceptron(input_data_xor[i], weights, bias)
        print(f"Input: {input_data_or[i]}, Output: {result}")
    
# Test the perceptron for OR problem
test_perceptron_logic_gate('OR')

# Test the perceptron for XOR problem
test_perceptron_logic_gate('XOR')