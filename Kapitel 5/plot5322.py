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
df_raw = pd.read_csv(
    "C:\\Users\\nilsl\\Desktop\\Matura\\Auswertungen\\gridsearch_RMSProp_results.csv",
    sep=';',
    on_bad_lines='skip'
)

# Relevante Spalten
df = df_raw[["regulrate", "RMSPropconst", "RMSProprate", "R2"]]


# ----------------------------------------
# Funktion für Plot + Korrelation
# ----------------------------------------
def analyze_and_plot(df, xcol):

    # Outlier entfernen
    df_clean = remove_outliers_iqr(df[[xcol, "R2"]], [xcol, "R2"])

    # Durchschnittlicher R2 pro Parameterwert
    df_mean = df_clean.groupby(xcol, as_index=False)["R2"].mean()

    # Korrelation basierend auf Mittelwerten
    corr, p_value = pearsonr(df_mean[xcol], df_mean["R2"])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df_mean[xcol], df_mean["R2"], label=f"Durchschnittlicher R² pro {xcol}")

    # Trendlinie
    z = np.polyfit(df_mean[xcol], df_mean["R2"], 1)
    p = np.poly1d(z)
    plt.plot(df_mean[xcol], p(df_mean[xcol]), "r--", label=f"Trendlinie (r={corr:.2f})")

    plt.title(f"Durchschnittlicher R²-Wert pro {xcol} (Outlier entfernt)")
    plt.xlabel(xcol)
    plt.ylabel("mittlerer R²-Wert")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"{xcol} – Korrelationskoeffizient r: {corr:.3f}")
    print(f"{xcol} – P-Wert: {p_value:.3e}")
    print("-" * 70)


# ----------------------------------------
# ANALYSEN
# ----------------------------------------

analyze_and_plot(df, "regulrate")
analyze_and_plot(df, "RMSPropconst")
analyze_and_plot(df, "RMSProprate")
