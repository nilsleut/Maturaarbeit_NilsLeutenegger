import sys
sys.path.append("Desktop\\Matura\\MLP_version5_biglist")
import numpy as np
from KNN import MLP
import matplotlib.pyplot as plt

# ============================
# Daten laden und normalisieren
# ============================
with open("daten_encoded.csv", "r") as f:
    X = np.loadtxt(f, delimiter=";")
xMin = X.min(axis=0); xMax = X.max(axis=0)
denom = xMax - xMin
denom = np.where(denom == 0, 1, denom)
X = (X - xMin) / denom

with open("daten_encoded_output.csv", "r") as f:
    Y = np.loadtxt(f, delimiter=";")
yMin = Y.min(axis=0); yMax = Y.max(axis=0)
denom = yMax - yMin
denom = np.where(denom == 0, 1, denom)
Y = (Y - yMin) / denom

# ============================
# Hilfsfunktionen
# ============================

def feature_importance(model, X, Y, metric=np.mean):
    """
    Berechnet die Wichtigkeit der Features mittels Permutation Importance.
    metric: Fehlerfunktion, z. B. np.mean für MSE.
    """
    baseline_error = metric((Y - model.predict(X).ravel()) ** 2)
    importances = []

    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        perm_error = metric((Y - model.predict(X_permuted).ravel()) ** 2)
        importance = perm_error - baseline_error
        importances.append(importance)

    return np.array(importances)


def train_and_evaluate(X, Y, runs=10):
    """
    Trainiert das Modell mehrfach und mittelt die Feature-Importances.
    """
    n_features = X.shape[1]
    all_importances = np.zeros((runs, n_features))

    for r in range(runs):
        print(f"\n=== Run {r+1}/{runs} ===")

        # Train/Test Split
        TrainSet = np.random.choice(X.shape[0], int(X.shape[0] * 0.8), replace=False)
        XTrain = X[TrainSet, :]
        YTrain = Y[TrainSet]
        TestSet = np.delete(np.arange(0, len(Y)), TrainSet)
        XTest = X[TestSet, :]
        YTest = Y[TestSet]

        # Modell erstellen
        model = MLP(
            akt="relu",
            hiddenlayer=(50, 50),
            dropout=0,
            eta=0.03,
            maxIter=1000,
            vareps=0.01,
            regul="l2",
            regulrate=0.001,
            patience=1000,
            batchsize=32,
            sgdmaxiterations=1000,
            momentum=0.9,
            sgdupdate=True,
            nestrov=False,
            adagrad=False,
            RMSProp=False,
            Adam=False,
            conjugateGradient=False
        )

        # Training
        model.fit(XTrain, YTrain)
        Y_pred = model.predict(XTest)
        mse = np.mean((YTest.ravel() - Y_pred.ravel()) ** 2)
        print(f"Run {r+1}: MSE = {mse:.6f}, Modell = {model.modelused}")

        # Feature Importances
        importances = feature_importance(model, XTest, YTest)
        all_importances[r, :] = importances

    # Mittelwert über alle Runs
    mean_importance = np.mean(all_importances, axis=0)
    return mean_importance


# ============================
# Hauptprogramm
# ============================
mean_importance = train_and_evaluate(X, Y, runs=10)

# Sortieren und Top/Bottom 10 anzeigen
sorted_idx = np.argsort(mean_importance)[::-1]

print("\n===============================")
print("Top 10 wichtigste Features:")
print("===============================")
for i in range(min(10, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"Feature {idx:3d}: Importance = {mean_importance[idx]:.6f}")

print("\n===============================")
print("Bottom 10 unwichtigste Features:")
print("===============================")
for i in range(1, min(11, len(sorted_idx) + 1)):
    idx = sorted_idx[-i]
    print(f"Feature {idx:3d}: Importance = {mean_importance[idx]:.6f}")

# Plot der mittleren Wichtigkeiten
plt.figure(figsize=(10, 6))
plt.bar(range(len(mean_importance)), mean_importance[sorted_idx])
plt.xlabel("Feature-Index (sortiert)")
plt.ylabel("Mittlere Importance (Δ MSE)")
plt.title("Mittlere Feature Importance über 10 Runs")
plt.tight_layout()
plt.show()
