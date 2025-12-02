import sys
import itertools
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

sys.path.append("Desktop/Matura/MLP_version5_biglist")
from KNN import MLP

# -------------------------------
# 🔧 Daten laden & normalisieren
# -------------------------------
with open("daten_encoded.csv", "r") as f:
    X = np.loadtxt(f, delimiter=";")
xMin = X.min(axis=0)
xMax = X.max(axis=0)
denom = np.where(xMax - xMin == 0, 1, xMax - xMin)
X = (X - xMin) / denom

with open("daten_encoded_output.csv", "r") as f:
    Y = np.loadtxt(f, delimiter=";")
yMin = Y.min(axis=0)
yMax = Y.max(axis=0)
denom = np.where(yMax - yMin == 0, 1, yMax - yMin)
Y = (Y - yMin) / denom

# -------------------------------
# 🔀 Train/Test Split
# -------------------------------
TrainSet = np.random.choice(X.shape[0], int(X.shape[0]*0.80), replace=False)
XTrain = X[TrainSet, :]
YTrain = Y[TrainSet]
TestSet = np.delete(np.arange(0, len(Y)), TrainSet)
XTest = X[TestSet, :]
YTest = Y[TestSet]

# -------------------------------
# 🧩 Parametergrid für Momentum-Test
# -------------------------------
param_grid = {
    "hiddenlayer": [(50,50), (100,50,25), (100,100,50,25)],
    "momentum": [0, 0.3, 0.6, 0.9],
    "nestrov": [False, True],
    "regul": ["none", "l2"],
    "regulrate": [0.0001, 0.001],
    "dropout": [0, 0.3],
    "early_stopping": [False, True]
}

keys = list(param_grid.keys())
combinations = list(itertools.product(*param_grid.values()))
print(f"Starte Momentum-Test mit {len(combinations)} Kombinationen...\n")

# -------------------------------
# ⚙️ Basisparameter
# -------------------------------
base_params = {
    "akt": "relu",
    "eta": 0.03,
    "maxIter": 1000,
    "vareps": 0.01,
    "batchsize": 32,
    "sgdupdate": True,
    "adagrad": False,
    "RMSProp": False,
    "Adam": False,
    "conjugateGradient": False
}

results = []

# -------------------------------
# 🚀 Grid Search Schleife
# -------------------------------
for i, combo in enumerate(combinations):
    params = dict(zip(keys, combo))
    patience = 250 if params["early_stopping"] else 0
    print(f"[{i+1}/{len(combinations)}] Momentum={params['momentum']}, Nestrov={params['nestrov']}, Reg={params['regul']}, Dropout={params['dropout']}, Arch={params['hiddenlayer']}")

    start_time = time.time()
    try:
        exe = MLP(
            akt=base_params["akt"],
            hiddenlayer=params["hiddenlayer"],
            dropout=params["dropout"],
            eta=base_params["eta"],
            maxIter=base_params["maxIter"],
            vareps=base_params["vareps"],
            regul=params["regul"],
            regulrate=params["regulrate"],
            patience=patience,
            batchsize=base_params["batchsize"],
            sgdmaxiterations=1000,
            momentum=params["momentum"],
            sgdupdate=base_params["sgdupdate"],
            nestrov=params["nestrov"],
            adagrad=base_params["adagrad"],
            adagradconst=1e-7,
            RMSProp=base_params["RMSProp"],
            RMSPropconst=1e-6,
            RMSProprate=0.9,
            Adam=base_params["Adam"],
            firstmomentrate=0.9,
            secmomentrate=0.999,
            Adamconst=1e-8,
            conjugateGradient=base_params["conjugateGradient"],
            alpha=1,
            c1=1e-4,
            c2=0.9,
            rho=0.25,
            maxlinesearch=20
        )

        exe.fit(XTrain, YTrain)
        Y_pred = exe.predict(XTest)

        # Fehlerberechnung
        YTest_flat = YTest.ravel()
        Y_pred_flat = Y_pred.ravel()
        mse = np.mean((YTest_flat - Y_pred_flat)**2)
        sum_res = np.sum((YTest_flat - Y_pred_flat)**2)
        sum_tot = np.sum((YTest_flat - np.mean(YTest_flat))**2)
        r2 = 1 - sum_res / sum_tot
        elapsed = time.time() - start_time

        results.append({
            **params,
            "MSE": mse,
            "R2": r2,
            "train_steps": getattr(exe, "trainingsteps", None),
            "epochs": max(getattr(exe, "epoch", [0])),
            "time_sec": round(elapsed, 2)
        })

        print(f"✅ Fertig: MSE={mse:.4f}, R²={r2:.4f}, Zeit={elapsed:.1f}s")

    except Exception as e:
        print(f"❌ Fehler bei Kombination {i+1}: {e}")
        results.append({**params, "MSE": None, "R2": None, "Error": str(e)})

# -------------------------------
# 💾 Ergebnisse speichern
#
