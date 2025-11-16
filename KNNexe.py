"""
Dieses Skript lädt vorbereitete, numerisch kodierte Datensätze,
trainiert das selbst implementierte MLP (siehe KNN.py) und
wertet die Modellleistung über verschiedene Kennzahlen (MSE, R²)
sowie grafisch aus.
"""


import sys
sys.path.append("Desktop\Matura\MLP_version6")
import numpy as np
from KNN import MLP 
import matplotlib.pyplot as plt

#---Daten laden und normalisieren---

with open("daten_encoded.csv", "r") as f:
    X = np.loadtxt(f, delimiter=";")

#Min-Max Normalisierung 
xMin = X.min(axis=0); xMax = X.max(axis=0)
denom = xMax - xMin
denom = np.where(denom == 0, 1, denom)
X = (X - xMin) / denom

with open("daten_encoded_output.csv", "r") as f:
    Y = np.loadtxt(f, delimiter=";")
 
#Min-Max Normalisierung
yMin = Y.min(axis=0); yMax = Y.max(axis=0)
denom = yMax - yMin
denom = np.where(denom == 0, 1, denom)
Y = (Y - yMin) / denom

#---Daten in Trainings- und Testset aufteilen---

TrainSet = np.random.choice(X.shape[0],int(X.shape[0]*0.80),replace=False)
XTrain = X[TrainSet,:]
YTrain = Y[TrainSet]

TestSet = np.delete(np.arange(0,len(Y)),TrainSet)
XTest = X[TestSet,:]
YTest = Y[TestSet]

#---MLP Initalisieren und Trainieren---

exe = MLP(
    akt="relu", 
    hiddenlayer=(50, 50, 50, 50, 50), 
    dropout=0, eta=0.03, 
    maxIter=1000, 
    regul="l2", 
    regulrate = 0.0001, 
    patience = 250, 
    batchsize = 32, 
    sgdmaxiterations = 1000, 
    momentum = 0, 
    sgdupdate = False, 
    nestrov = False, 
    adagrad = False, 
    adagradconst = 10e-7, 
    RMSProp = False, 
    RMSPropconst = 10e-6, 
    RMSProprate = 0.9, 
    Adam = True, 
    firstmomentrate = 0.9, 
    secmomentrate = 0.999, 
    Adamconst = 10e-8, 
    conjugateGradient = False, 
    alpha = 1, 
    c1 = 1e-4, 
    c2 = 0.9, 
    rho = 0.25, 
    maxlinesearch = 20
    )

#Modelltraining
exe.fit(XTrain,YTrain)

#Vorhersagen auf Testset berechnen
Y_pred = exe.predict(XTest)

#---Modellleistung berechnen und ausgeben---

YTest = YTest.ravel()
Y_pred = Y_pred.ravel()

print("genutztes Modell: ", exe.modelused)
if exe.dropoutused:
    print("Dropout was used")
print("Mean Square Error:", np.mean((YTest - Y_pred)**2))
print("Var",np.var(YTest))

#R^2-Wert berechnen
sum_res = np.sum((YTest - Y_pred)**2)         
sum_tot = np.sum((YTest - np.mean(YTest))**2) 
r2 = 1 - sum_res / sum_tot 
print("R^2 Score:", r2)

#Statische Kennzahlen ausgeben
print("YTest.min =", YTest.min(), "YTest.max =", YTest.max())
print("Y_pred.min =", Y_pred.min(), "Y_pred.max =", Y_pred.max())
print("best amount of training steps:", exe.trainingsteps)
print("Predicted values: ", np.round(Y_pred[:3], 2))
print("Real values:", YTest[:3])
print("Mean Pred", np.mean(Y_pred))
print("Mean Test", np.mean(YTest))
print("benötigte Schritte", max(exe.epoch))

#---Fehlerverlauf visualisieren---
epoch = exe.epoch
error = exe.meanE
plt.plot(epoch, error)
plt.xlabel('epoch')
plt.ylabel('error')
plt.yscale('log')   #Log-Skalierung der Fehlerachse
plt.show()

#---Validierungssatz visualisieren---

plt.figure(figsize=(10, 6))
plt.plot(YTest, label="YTest (True Values)", color='black', linewidth=1)
plt.plot(Y_pred, label="Y_pred (Predicted)", color='red', linewidth=1)
plt.axhline(y=np.mean(YTest), color='blue', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(YTest):.2f}')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Wert")
plt.tight_layout()
plt.show()


