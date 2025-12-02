import sys
sys.path.append("Desktop\\Matura\\MLP_version5_biglist")
import numpy as np
import itertools
import pandas as pd
import time
from KNN import MLP

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
XTest,  YTest  = X[TestSet, :],  Y[TestSet]

# --- Parameter-Grid für Adam ---
param_grid = {
    "akt": ["relu"],
    "hiddenlayer": [
        (50, 50, 50),
        (100, 50, 25),
        (100, 100, 100, 50),
    ],
    "dropout": [0, 0.1, 0.3],
    "eta": [0.001, 0.01, 0.03],
    "regul": ["none", "l1", "l2"],
    "regulrate": [1e-5, 1e-4, 1e-2],
    "batchsize": [16, 32, 64],
    "firstmomentrate": [0.8, 0.9, 0.99],   # β1
    "secmomentrate":  [0.9, 0.99, 0.999],  # β2
    "Adamconst":      [1e-8, 1e-7, 1e-6]   # ε
}

# --- Alle Kombinationen erzeugen ---
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Gesamtanzahl der Adam-Durchläufe: {len(combinations)}")

results = []

# --- Grid Search ---
for i, params in enumerate(combinations, 1):
    print(f"\n🔹 Lauf {i}/{len(combinations)} mit Parametern: {params}")
    start = time.time()

    exe = MLP(
        akt=params["akt"],
        hiddenlayer=params["hiddenlayer"],
        dropout=params["dropout"],
        eta=params["eta"],
        maxIter=500,
        vareps=0.01,
        regul=params["regul"],
        regulrate=params["regulrate"],
        patience=250,
        batchsize=params["batchsize"],
        sgdmaxiterations=1000,
        momentum=0,
        sgdupdate=True,
        nestrov=False,
        adagrad=False,
        RMSProp=False,
        Adam=True,
        firstmomentrate=params["firstmomentrate"],
        secmomentrate=params["secmomentrate"],
        Adamconst=params["Adamconst"],
        conjugateGradient=False
    )

    exe.fit(XTrain, YTrain)
    Y_pred = exe.predict(XTest).ravel()
    Y_true = YTest.ravel()

    mse = np.mean((Y_true - Y_pred) ** 2)
    r2 = 1 - np.sum((Y_true - Y_pred) ** 2) / np.sum((Y_true - np.mean(Y_true)) ** 2)
    duration = time.time() - start

    results.append({
        **params,
        "MSE": mse,
        "R2": r2,
        "TrainSteps": exe.trainingsteps,
        "Time_s": round(duration, 2)
    })

# --- Ergebnisse speichern ---
df = pd.DataFrame(results)
df.to_csv("gridsearch_Adam_results.csv", sep=";", index=False)
print("\n✅ Ergebnisse gespeichert in 'gridsearch_Adam_results.csv'")
