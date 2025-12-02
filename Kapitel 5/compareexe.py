from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer
import sys
sys.path.append("Desktop\Matura\MLP_version5_biglist")
import numpy as np
from KNN import MLP 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#np.random.seed(42)  


MLP = MLP(akt="relu", hiddenlayer=(50, 50, 50, 50, 50), dropout=0, eta=0.01, maxIter=1000, regul="l2", regulrate = 0.001, patience = 250, batchsize = 32, sgdmaxiterations = 1000, momentum = 0, sgdupdate = np.False_, nestrov = False, adagrad = False, adagradconst = 10e-7, RMSProp = False, RMSPropconst = 10e-6, RMSProprate = 0.9, Adam = True, firstmomentrate = 0.9, secmomentrate = 0.999, Adamconst = 10e-8, conjugateGradient = False, alpha = 1, c1 = 1e-4, c2 = 0.9, rho = 0.25, maxlinesearch = 20)

lr = LinearRegression()

rf = RandomForestRegressor(n_estimators=100, random_state=42)

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)



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
y = (Y - yMin) / denom
print("Loaded X shape:", X.shape, "Y shape:", y.shape)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
print("Scaled X shape:", X_scaled.shape, "y_scaled shape:", y_scaled.shape)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Cross-validation
scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
cv_results_mlp = {'r2': [], 'mse': []}
for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(X_scaled):
    lr.fit(X_scaled[train_idx], y_scaled[train_idx])
    y_pred = lr.predict(X_scaled[val_idx])
    cv_results_mlp['r2'].append(r2_score(y_scaled[val_idx], y_pred))
    cv_results_mlp['mse'].append(mean_squared_error(y_scaled[val_idx], y_pred))
print("lr CV R²:", np.mean(cv_results_mlp['r2']), "±", np.std(cv_results_mlp['r2']))
print("lr CV MSE:", np.mean(cv_results_mlp['mse']), "±", np.std(cv_results_mlp['mse']))

#MLP.fit(X_train, y_train)
#y_pred_mlp = MLP.predict(X_test)
#plt.scatter(y_test, y_test - y_pred_mlp, label="MLP")
#plt.axhline(0, color='red', linestyle='--')
#plt.xlabel("True Values")
#plt.ylabel("Residuals")
#plt.legend()
#plt.show()