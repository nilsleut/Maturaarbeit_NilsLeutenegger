import sys
sys.path.append("Desktop\\Matura\\MLP_version5_biglist")
import numpy as np
import itertools
import pandas as pd
from KNN import MLP
import time

# --- Daten laden & normalisieren ---
with open("daten_encoded.csv", "r") as f:
    X = np.loadtxt(f, delimiter=";")
xMin, xMax = X.min(axis=0), X.max(axis=0)
X = (X - xMin) / np.where(xMax - xMin == 0, 1, xMax - xMin)

with open("daten_encoded_output.csv", "r") as f:
    Y = np.loadtxt(f, delimiter=";")
yMin, yMax = Y.min(axis=0), Y.max(axis=0)
Y = (Y - yMin) / np.where(yMax - yMin == 0, 1, yMax - yMin)

# --- Train/Test Split ---
TrainSet = np.random.choice(X.shape[0], int(X.shape[0]*0.8), replace=False)
TestSet = np.delete(np.arange(0, len(Y)), TrainSet)
XTrain, YTrain = X[TrainSet, :], Y[TrainSet]
XTest, YTest = X[TestSet, :], Y[TestSet]

# --- Parameter-Grid für Conjugate Gradient (reduziert für Laufzeit) ---
param_grid = {
    "akt": ["relu", "sigmoid"],
    "hiddenlayer": [
        (50, 50, 50),
        (100, 50, 25)
    ],
    "dropout": [0, 0.2],
    "eta": [0.001, 0.01],
    "regul": ["none", "l2"],
    "regulrate": [0.0001, 0.001],
    "batchsize": [32]
}

# --- Alle Kombinationen generieren ---
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Gesamtanzahl der CG-Durchläufe: {len(combinations)}")

# --- Ergebnisse speichern ---
results = []

for i, params in enumerate(combinations, 1):
    print(f"\n🔹 Lauf {i}/{len(combinations)} mit Parametern: {params}")
    start_time = time.time()
    
    exe = MLP(
        akt=params["akt"],
        hiddenlayer=params["hiddenlayer"],
        dropout=params["dropout"],
        eta=params["eta"],
        maxIter=50,
        vareps=0.01,
        regul=params["regul"],
        regulrate=params["regulrate"],
        patience=200,
        batchsize=params["batchsize"],
        sgdmaxiterations=500,
        momentum=0,
        sgdupdate=False,
        nestrov=False,
        adagrad=False,
        adagradconst=1e-7,
        RMSProp=False,
        Adam=False,
        conjugateGradient=True
    )

    exe.fit(XTrain, YTrain)
    Y_pred = exe.predict(XTest).ravel()
    Y_true = YTest.ravel()

    mse = np.mean((Y_true - Y_pred) ** 2)
    sum_res = np.sum((Y_true - Y_pred) ** 2)
    sum_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
    r2 = 1 - sum_res / sum_tot
    duration = time.time() - start_time

    results.append({
        **params,
        "MSE": mse,
        "R2": r2,
        "TrainSteps": exe.trainingsteps,
        "Time_s": round(duration, 2)
    })

# --- Ergebnisse speichern ---
df = pd.DataFrame(results)
df.to_csv("gridsearch_CG_results.csv", sep=";", index=False)
print("\n✅ Ergebnisse gespeichert in 'gridsearch_CG_results.csv'")
