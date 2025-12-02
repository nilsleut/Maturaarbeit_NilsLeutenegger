import numpy as np
import matplotlib.pyplot as plt

# Fehlerachse
x = np.linspace(-1, 1, 400)

# Parameter für Huber-Loss
delta = 1

# Lossfunktionen
mae = np.abs(x)
mse = x**2

# Huber-Loss
huber = np.where(np.abs(x) <= delta,
                 0.5 * x**2,
                 delta * (np.abs(x) - 0.5 * delta))

# Plot
plt.figure()
plt.plot(x, mae, label='MAE')
plt.plot(x, mse, label='MSE')
plt.plot(x, huber, label=f'Huber Loss (δ={delta})')

plt.legend()
plt.title("Vergleich von MAE, MSE und Huber Loss")
plt.xlabel("Fehler (e)")
plt.ylabel("Wert")
plt.grid(True)
plt.show()
