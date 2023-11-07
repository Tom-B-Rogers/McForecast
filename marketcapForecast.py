import cleanFinancialData as cDA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np
import glob
import os

import matplotlib.pyplot as plt
import openpyxl
import datetime as dt
from collections import Counter

pd.set_option('display.max_columns', 25)

d = cDA.df_forecast_dropped.copy()

d = d[d["Company"] != 'WES']
d = d[d["Company"] != 'ALL']
d = d[d["Company"] != 'TLC']

random_state = 42
test_size = 0.25

rf_param_grid = {
    'n_estimators': [900, 950, 1000, 1050, 1100],
    'max_depth': [5, None],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [2, 3, 4],
    'bootstrap': [True, False],
    'criterion': ['friedman_mse']
}

reg_cols_to_drop = ["Market_Cap_Forecast", "Company",
                    'Quartile_Q2', 'Quartile_Q3', 'Quartile_Q4', 'Ticker',
                    "Market_Cap_Lag", "Market_Cap_Change"]

X = d.drop(columns=reg_cols_to_drop)
y = d['Market_Cap_Forecast']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=random_state),
                              param_grid=rf_param_grid, cv=4, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print(
    f"Random Forest GridSearch RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print("Best parameters found:", rf_grid_search.best_params_)

importances = best_rf.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)


sorted_idx = np.argsort(importances)
pos = np.arange(sorted_idx.shape[0])

fig, ax = plt.subplots(figsize=(15, 15))
ax.barh(pos, importances[sorted_idx], align='center',
        color='slateblue', edgecolor='black')
ax.set_yticks(pos)
ax.set_yticklabels(feature_names[sorted_idx], fontsize=12)
ax.set_xlabel('Feature Importance Index', fontsize=14)
ax.set_title('Features', fontsize=16, fontweight='bold')

ax.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
ax.set_axisbelow(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_facecolor('whitesmoke')
plt.tight_layout()
plt.show()

data_to_forecast = cDA.data_forecast.loc["2023-06-30"]
dat = data_to_forecast.drop(reg_cols_to_drop, axis=1)
data_to_forecast["Market_Cap_Forecast"] = best_rf.predict(dat)

plotting_data = data_to_forecast[[
    "Company", "Market_Cap_Forecast", "Market Cap."]]
plotting_data = plotting_data[plotting_data["Company"] != "TLC"]
plotting_data = plotting_data[plotting_data["Company"] != "WES"]
plotting_data = plotting_data[plotting_data["Company"] != "ALL"]

plotting_data.to_excel("stock_forecasts.xlsx")
