import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Daten in DataFrame schreiben
# ---------------------------------------------------------
data = {
    "Modell": ["xgb", "svr", "rf", "lr", "eigenes MLP"],
    "R2": [0.329766814, 0.248124127, 0.278458307, 0.240061638, 0.3175],
    "Abweichung": [0.015373823, 0.008806901, 0.016273272, 0.024310748, 0.0275]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

x = np.arange(len(df))
plt.bar(x, df["R2"], yerr=df["Abweichung"], capsize=6)

plt.xticks(x, df["Modell"])
plt.ylabel("R²-Wert")
plt.title("Vergleich verschiedener Trainingsmodelle (mit Abweichung)")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
