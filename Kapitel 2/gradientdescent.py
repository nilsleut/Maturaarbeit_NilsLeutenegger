import numpy as np
import matplotlib.pyplot as plt

# Define function and its gradient
def f(x):
    return x**2

def grad_f(x):
    return 2*x

# Gradient descent parameters
x0 = 8           # starting point
lr = 0.1       # learning rate
steps = 20       # number of iterations

# Store values
x_values = [x0]
for i in range(steps):
    x_new = x_values[-1] - lr * grad_f(x_values[-1])
    x_values.append(x_new)

x = np.linspace(-10, 10, 400)
y = f(x)

# Plot function
plt.figure()
plt.plot(x, y, label=r'$f(x) = x^2$')
plt.title("Gradient Descent Visualization")
plt.xlabel("x")
plt.ylabel("f(x)")

# Plot descent steps
x_steps = np.array(x_values)
y_steps = f(x_steps)

plt.plot(x_steps, y_steps, 'o-', color='red', label='Gradient Descent Steps')
for i in range(len(x_steps)):
    plt.annotate(f'{i}', (x_steps[i], y_steps[i]), textcoords="offset points", xytext=(5,5))

plt.legend()
plt.grid(True)
plt.show()
