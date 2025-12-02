import numpy as np
import matplotlib.pyplot as plt

# Define functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Generate x values
x = np.linspace(-5, 5, 400)

# Compute function values
y_sigmoid = sigmoid(x)
y_sigmoid_deriv = sigmoid_derivative(x)
y_relu = relu(x)
y_relu_deriv = relu_derivative(x)

# Plot all on one figure
plt.figure()
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_sigmoid_deriv, label='Sigmoid Derivative')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_relu_deriv, label='ReLU Derivative')
plt.legend()
plt.title("Sigmoid, ReLU and their Derivatives")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
