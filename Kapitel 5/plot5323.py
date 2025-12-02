import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

# -------------------------
# Funktion zur Outlier-Entfernung (IQR-Methode)
# -------------------------
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


# -------------------------
# Daten laden
# -------------------------
df = pd.read_csv(
    "C:\\Users\\nilsl\\Desktop\\Matura\\Auswertungen\\gridsearch_Adam_results.csv",
    sep=';',
    on_bad_lines='skip'
)

# Relevante Spalten
df = df[["eta", "batchsize", "R2"]]

# -------------------------
# Analyse für eta
# -------------------------
df_eta = df[["eta", "R2"]]

# Outlier entfernen
df_eta_clean = remove_outliers_iqr(df_eta, ["eta", "R2"])

# Mittelwert von R² pro eta berechnen
df_eta_mean = df_eta_clean.groupby("eta", as_index=False)["R2"].mean()

# Korrelation berechnen
corr_eta, p_eta = pearsonr(df_eta_mean["eta"], df_eta_mean["R2"])

# Plot
plt.figure(figsize=(8,5))
plt.scatter(df_eta_mean["eta"], df_eta_mean["R2"], label="Durchschnittlicher R² pro eta")

# Trendlinie
z = np.polyfit(df_eta_mean["eta"], df_eta_mean["R2"], 1)
p = np.poly1d(z)
plt.plot(df_eta_mean["eta"], p(df_eta_mean["eta"]), "r--", label=f"Trendlinie (r={corr_eta:.2f})")

plt.title("Durchschnittlicher R²-Wert pro eta (Outlier entfernt)")
plt.xlabel("eta")
plt.ylabel("mittlerer R²-Wert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("ETA-Auswertung:")
print(f"Korrelationskoeffizient r: {corr_eta:.3f}")
print(f"P-Wert: {p_eta:.3e}")
print("-" * 50)


# -------------------------
# Analyse für batchsize
# -------------------------
df_batch = df[["batchsize", "R2"]]

# Outlier entfernen
df_batch_clean = remove_outliers_iqr(df_batch, ["batchsize", "R2"])

# Mittelwert von R² pro batchsize berechnen
df_batch_mean = df_batch_clean.groupby("batchsize", as_index=False)["R2"].mean()

# Korrelation berechnen
corr_batch, p_batch = pearsonr(df_batch_mean["batchsize"], df_batch_mean["R2"])

# Plot
plt.figure(figsize=(8,5))
plt.scatter(df_batch_mean["batchsize"], df_batch_mean["R2"], label="Durchschnittlicher R² pro batchsize")

# Trendlinie
z = np.polyfit(df_batch_mean["batchsize"], df_batch_mean["R2"], 1)
p = np.poly1d(z)
plt.plot(df_batch_mean["batchsize"], p(df_batch_mean["batchsize"]), "r--", label=f"Trendlinie (r={corr_batch:.2f})")

plt.title("Durchschnittlicher R²-Wert pro batchsize (Outlier entfernt)")
plt.xlabel("batchsize")
plt.ylabel("mittlerer R²-Wert")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Batchsize-Auswertung:")
print(f"Korrelationskoeffizient r: {corr_batch:.3f}")
print(f"P-Wert: {p_batch:.3e}")
