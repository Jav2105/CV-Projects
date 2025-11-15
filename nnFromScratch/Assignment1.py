
import math, random
random.seed(0)

# All vectors with math operations are 1xN matrices so that we can multiply vectors and matrices together

def sigmoid(z):
    return [[1/(1 + math.exp(-z[0][i])) for i in range(len(z[0]))]]

def sigmoid_derivative(z):
    return [[math.exp(-z[0][i]) / (1 + math.exp(-z[0][i]))**2 for i in range(len(z[0]))]]

def matrix_sum(A, B):
    # Also serves for array summation
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_substract(A, B):
    # Also serves for array substraction
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_multiplication(A, B):
    C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_scalar(A, scalar):
    return [[scalar * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def hadamard_product(A, B):
    # Useful for backpropagation
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

class Layer:
    def __init__(self, n_neurons, n_inputs, n_layer):
        self.n_layer = n_layer
        # The variance changes depending on the input size so that activations do not become saturated
        # Xavier / Glorot initialization
        self.weights = [[random.gauss(0, math.sqrt(2.0 / (n_inputs + n_neurons))) for _ in range(n_inputs)] for _ in range(n_neurons)]
        self.biases = [[0.0 for _ in range(n_neurons)]]
        self.prev_activations = None
        self.z = None
        self.activations = None

    def output(self, input):
        self.prev_activations = input
        self.z = matrix_sum(matrix_multiplication(input, transpose(self.weights)), self.biases)
        self.activations = sigmoid(self.z)
        return(self.activations)
    
class Network:
    def __init__(self, structure, n_inputs):
        self.n_layers = len(structure)
        self.layers = []
        layer_inputs = n_inputs
        for i in range(self.n_layers):
            self.layers.append(Layer(structure[i], layer_inputs, i))
            layer_inputs = structure[i]

        self.structure = structure
        self.loss_derivative = None

    def output(self, input):
        input_aux = input
        for layer in self.layers:
            layer_output = layer.output(input_aux)
            input_aux = layer_output
        return layer_output
    
    def forward_prop(self, input, expected_out):
        aux = matrix_substract(self.output(input), expected_out)
        self.loss_derivative = [[2*aux[0][i] for i in range(len(aux[0]))]]

    def backpropagate(self, x_train, y_train, learning_rate, epochs):
        next_error_term = None
        next_weights = None
        for _ in range(epochs):
            # Epochs: number of times that all of the instances are used for backpropagation
            for i in range(len(x_train)):
                self.forward_prop([x_train[i]], [y_train[i]])
                for layer in reversed(self.layers):
                    if (layer.n_layer == self.n_layers-1):
                        error_term = hadamard_product(self.loss_derivative, sigmoid_derivative(layer.z))
                    else:
                        product = matrix_multiplication (next_error_term, next_weights)
                        error_term = hadamard_product(product, sigmoid_derivative(layer.z))
                    next_error_term = error_term
                    next_weights = layer.weights
                    grad_weights = matrix_multiplication(transpose(layer.prev_activations), error_term)
                    layer.weights = matrix_substract(layer.weights, matrix_scalar(transpose(grad_weights), learning_rate))
                    layer.biases = matrix_substract(layer.biases, matrix_scalar(error_term, learning_rate))

    def accuracy(self, x, y):
        # We determine the accuracy using linear regression, depending on if the output is closer to
        # zero or to one.
        correct = 0
        for i in range(len(x)):
            prediction = self.output([x[i]])[0]
            for j in range(len(y[0])):
                if ((prediction[j] >= 0.5 and y[i][j] == 1) or (prediction[j] < 0.5 and y[i][j] == 0)):
                    correct += 1
        return correct / (len(x) * len(y[0]))

    def learn(self, x_train, y_train, x_test, y_test, learning_rate, epochs):
        print('''We are going to explore how the following affect our model:
              - Learning rate
              - Structure
              - Number of epochs
              - Training - test split''')

# Four instances turned into sixteen via the addition of noise
xor = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [0.1, 0, 0],
    [0, 0.1, 0],
    [0.001, 0.009, 0],
    [0.05, 0.99, 1],
    [0.04, 0.95, 1],
    [0.05, 0.95, 1],
    [0.9, 0.01, 1],
    [1, 0.08, 1],
    [0.96, 0.04, 1],
    [0.95, 1, 0],
    [0.97, 0.9, 0],
    [0.99, 0.98, 0]
]

# Thirty two instances
binary_adder = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,1,0],
    [0,0,0,1,0,0,1,0],
    [0,0,0,1,1,0,0,1],
    [0,0,1,0,0,1,0,0],
    [0,0,1,0,1,1,1,0],
    [0,0,1,1,0,1,1,0],
    [0,0,1,1,1,1,0,1],
    [0,1,0,0,0,1,0,0],
    [0,1,0,0,1,1,1,0],
    [0,1,0,1,0,1,1,0],
    [0,1,0,1,1,1,0,1],
    [0,1,1,0,0,0,1,0],
    [0,1,1,0,1,0,0,1],
    [0,1,1,1,0,0,0,1],
    [0,1,1,1,1,0,1,1],
    [1,0,0,0,0,1,0,0],
    [1,0,0,0,1,1,1,0],
    [1,0,0,1,0,1,1,0],
    [1,0,0,1,1,1,0,1],
    [1,0,1,0,0,0,1,0],
    [1,0,1,0,1,0,0,1],
    [1,0,1,1,0,0,0,1],
    [1,0,1,1,1,0,1,1],
    [1,1,0,0,0,0,1,0],
    [1,1,0,0,1,0,0,1],
    [1,1,0,1,0,0,0,1],
    [1,1,0,1,1,0,1,1],
    [1,1,1,0,0,1,1,0],
    [1,1,1,0,1,1,0,1],
    [1,1,1,1,0,1,0,1],
    [1,1,1,1,1,1,1,1]
]

print('''We are going to explore how the following affect our model:
        - Learning rate
        - Structure
        - Number of epochs
        - Training - test split
        32 total models: 16 XOR models and 16 binary addition\n''')
    
def test_train_split(split, dataset):
    aux = dataset[:]
    random.shuffle(aux)
    n_test = int((1 - split) * len(dataset))
    test = aux[:n_test]
    train = aux[n_test:]
    return test, train

xor_learning_rates = [0.01, 0.04]
binary_learning_rates = [0.01, 0.06]
xor_structures = [[3], [3, 1]]
binary_structures = [[8, 8, 3], [8, 10, 10, 3]]
xor_n_epochs = [200, 2000]
binary_n_epochs = [400, 2500]
xor_training_test_split = [0.625, 0.75]
binary_training_test_split = [0.75, 0.875]

print("Results for the xor models:")
for structure in xor_structures:
    for rate in xor_learning_rates:
        for epoch in xor_n_epochs:
            for split in xor_training_test_split:
                model = Network(structure, 2)
                test, train = test_train_split(split, xor)
                x_test = [row[:2] for row in test]
                y_test = [row[2:] for row in test]
                x_train = [row[:2] for row in train]
                y_train = [row[2:] for row in train]
                model.backpropagate(x_train, y_train, rate, epoch)
                training_accuracy = model.accuracy(x_train, y_train)
                testing_accuracy = model.accuracy(x_test, y_test)
                print(f'''Structure: {structure}, Learning rate: {rate}, Epochs: {epoch}, Split: {split} => 
                        Training accuracy: {training_accuracy}, Testing accuracy: {testing_accuracy}''')
        
print(''' Best xor model:
    - Stucture: [3, 1]
    - Learning rate: 0.04
    - Epochs: 2000
    - Split: 0.75
    Training accuracy: 1.0
    Testing accuracy: 1.0
    ''')

print("Results for the binary adder models:")
for structure in binary_structures:
    for rate in binary_learning_rates:
        for epoch in binary_n_epochs:
            for split in binary_training_test_split:
                model = Network(structure, 5)
                test, train = test_train_split(split, binary_adder)
                x_test = [row[:5] for row in test]
                y_test = [row[5:] for row in test]
                x_train = [row[:5] for row in train]
                y_train = [row[5:] for row in train]
                model.backpropagate(x_train, y_train, rate, epoch)
                training_accuracy = model.accuracy(x_train, y_train)
                testing_accuracy = model.accuracy(x_test, y_test)
                print(f'''Structure: {structure}, Learning rate: {rate}, Epochs: {epoch}, Split: {split} => 
                        Training accuracy: {training_accuracy},\tTesting accuracy: {testing_accuracy}''')
                
print(''' Best binary model:
    - Stucture: [8, 10, 10, 3]
    - Learning rate: 0.06
    - Epochs: 2500
    - Split: 0.875
    Training accuracy: 0.9762
    Testing accuracy: 1.0
    ''')
