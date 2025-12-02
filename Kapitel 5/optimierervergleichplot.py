import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Daten in DataFrame schreiben
# ---------------------------------------------------------
data = {
    "Optimierer": ["Sdg", "Sdg", "Adagrad", "Adagrad", "RMSProp", "RMSProp", "Adam", "Adam", "cg", "cg"],
    "Datenset": ["klein", "gross", "klein", "gross", "klein", "gross", "klein", "gross", "klein", "gross"],
    "R2": [0.0778, 0.2452, 0.0526, 0.2762, 0.0817, 0.2874, 0.0447, 0.3175, -0.0169, -0.0008],
    "Abweichung": [0.103, 0.0224, 0.0637, 0.0277, 0.0868, 0.0356, 0.0853, 0.0275, 0.0972, 0.0814]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Plot vorbereiten
# ---------------------------------------------------------
optimizers = df["Optimierer"].unique()
x = np.arange(len(optimizers))

# Werte extrahieren
r2_small = df[df["Datenset"] == "klein"]["R2"].values
r2_large = df[df["Datenset"] == "gross"]["R2"].values

err_small = df[df["Datenset"] == "klein"]["Abweichung"].values
err_large = df[df["Datenset"] == "gross"]["Abweichung"].values

width = 0.35  # Abstand zwischen den Balken

# ---------------------------------------------------------
# Barplot
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.bar(x - width/2, r2_small, width, yerr=err_small, capsize=5, label="kleines Datenset")
plt.bar(x + width/2, r2_large, width, yerr=err_large, capsize=5, label="grosses Datenset")

plt.xticks(x, optimizers)
plt.ylabel("R²-Wert")
plt.title("Vergleich der Optimierer anhand von R² (mit Abweichung)")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()
