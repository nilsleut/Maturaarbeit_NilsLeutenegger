import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# ----------------------------------------
# Outlier-Entfernung (IQR-Methode)
# ----------------------------------------
def remove_outliers_iqr(df, columns):
    clean_df = df.copy()
    for col in columns:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    return clean_df

# ----------------------------------------
# Daten laden
# ----------------------------------------
df = pd.read_csv(
    "C:\\Users\\nilsl\\Desktop\\Matura\\Auswertungen\\SGD_Regularisierung_Architekturen.csv",
    sep=';',
    on_bad_lines='skip'
)

df = df[["momentum", "R2"]]
df.rename(columns={"momentum": "Momentum"}, inplace=True)

# ----------------------------------------
# Outlier entfernen
# ----------------------------------------
df_clean = remove_outliers_iqr(df, ["Momentum", "R2"])

# ----------------------------------------
# Durchschnittlicher R²-Wert pro Momentum
# ----------------------------------------
df_mean = df_clean.groupby("Momentum", as_index=False)["R2"].mean()

# ----------------------------------------
# Korrelation basierend auf Mittelwerten
# ----------------------------------------
corr, p_value = pearsonr(df_mean["Momentum"], df_mean["R2"])

# ----------------------------------------
# Plot
# ----------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df_mean["Momentum"], df_mean["R2"], label="Durchschnittlicher R² pro Momentum")

# Trendlinie
z = np.polyfit(df_mean["Momentum"], df_mean["R2"], 1)
p = np.poly1d(z)
plt.plot(df_mean["Momentum"], p(df_mean["Momentum"]), "r--", label=f"Trendlinie (r={corr:.2f})")

plt.title("Durchschnittlicher R²-Wert pro Momentum (Outlier entfernt)")
plt.xlabel("Momentum")
plt.ylabel("mittlerer R²-Wert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Korrelationskoeffizient (Pearson r): {corr:.3f}")
print(f"P-Wert: {p_value:.3e}")
