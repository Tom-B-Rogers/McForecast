from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import cleanFinancialData as cDA
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
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


data = cDA.df_forecast_dropped.copy()
data_lag = cDA.df_forecast_dropped_MCLAG
random_state = 42
test_size = 0.25

rf_param_grid = {
    'n_estimators': [1400, 1500, 1600],
    'max_depth': [None],
    'min_samples_split': [3, 4, 5],
    'min_samples_leaf': [2]
}

lasso_param_grid = {'alpha': [0.1, 0.01, 0.001, 0.0001]}

reg_cols_to_drop = ["Market_Cap_Forecast", "Company",
                    'Quartile_Q2', 'Quartile_Q3', 'Quartile_Q4', 'Ticker']

X = data_lag.drop(columns=reg_cols_to_drop)
y = data_lag['Market_Cap_Forecast']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

# xgb_model = xgb.XGBRegressor(
#     objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
#     max_depth=20, colsample_bytree=0.5, subsample=0.5, gamma=1,
#     random_state=random_state
# )
# xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_test, y_test)], verbose=True)
# y_pred = xgb_model.predict(X_test)
# print(f"XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Random forest with scaling

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_Y.transform(y_test.values.reshape(-1, 1))

rf_model = RandomForestRegressor(n_estimators=1500, random_state=random_state)
rf_model.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = rf_model.predict(X_test_scaled)

# Reshape the 1D array to 2D
y_pred_scaled_2D = y_pred_scaled.reshape(-1, 1)

y_pred = scaler_Y.inverse_transform(y_pred_scaled_2D)
print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

rf_model = RandomForestRegressor(n_estimators=1500, random_state=random_state)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")


importances = rf_model.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots(figsize=(12, 6))
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
plt.show()

# rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid=rf_param_grid, cv=4, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
# rf_grid_search.fit(X_train, y_train)
# best_rf = rf_grid_search.best_estimator_
# y_pred = best_rf.predict(X_test)
# print(f"Random Forest GridSearch RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
# print("Best parameters found:", rf_grid_search.best_params_)

# # Data Scaling for Neural Network
# scaler_X = MinMaxScaler()
# scaler_Y = MinMaxScaler()
# X_train_scaled = scaler_X.fit_transform(X_train)
# X_test_scaled = scaler_X.transform(X_test)
# y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))
# y_test_scaled = scaler_Y.transform(y_test.values.reshape(-1, 1))

# # Neural Network Model
# nn_model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(1)
# ])
# nn_model.compile(optimizer='adam', loss='mean_squared_error')
# nn_model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test_scaled), verbose=1)
# loss = nn_model.evaluate(X_test_scaled, y_test_scaled)
# y_pred_scaled = nn_model.predict(X_test_scaled)
# y_pred = scaler_Y.inverse_transform(y_pred_scaled)
# print(f"Neural Network Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# lasso_param_grid = {'alpha': [0.1, 0.01, 0.001, 0.0001]}
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
# lasso_grid_search = GridSearchCV(Lasso(max_iter=10000), param_grid=lasso_param_grid, cv=5)
# lasso_grid_search.fit(X_train, y_train)
# y_pred = lasso_grid_search.predict(X_test)
# print(f"Lasso RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
# print("Best Lasso hyperparameters:", lasso_grid_search.best_params_)


# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.1)
# elastic_net.fit(X_train, y_train)
# y_pred = elastic_net.predict(X_test)
# print(f"Elastic Net RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

data_to_forecast = cDA.data_forecast.loc["2023-06-30"]
dat = data_to_forecast.drop(reg_cols_to_drop, axis=1)
data_to_forecast["Market_Cap_Forecast"] = rf_model.predict(dat)

plotting_data = data_to_forecast[[
    "Company", "Market_Cap_Forecast", "Market Cap."]]
# plotting_data["Shares_Outstanding"] = cDA.forecasting_df["Weighted Average Number Of Shares"].loc["2023-06-30"].values
# plotting_data["Share_Price_Forecasting"] = plotting_data["Market_Cap_Forecast"] / plotting_data["Shares_Outstanding"]
plotting_data.to_excel("stock_forecasts.xlsx")

data_to_forecast = cDA.data_forecast.loc["2023-06-30"]
dat = data_to_forecast.drop(reg_cols_to_drop, axis=1)
dat_scaled = scaler_X.transform(dat)
predictions_scaled = rf_model.predict(dat_scaled)
forecasted = scaler_Y.inverse_transform(predictions_scaled.reshape(-1, 1))
data_to_forecast["Market_Cap_Forecast"] = forecasted

# data_check = cDA.data_forecast.copy()

# data_check = data_check[data_check["Company"].isin(
#     ["PBH", "CKF", "CCX", "SGR", "APE"])]

# summary_stats = data_check.groupby(
#     "Company").describe().loc[:, (slice(None), ['mean'])]

# clusterCols = forest_importances[forest_importances > 0.005].index.to_list()
# clusterCols.append("Company")

# cd = cDA.data_extra_lag
# reg_cols_to_drop = ["Market_Cap_Forecast", 'Quartile_Q2',
#                     'Quartile_Q3', 'Quartile_Q4', 'Ticker']
# cd = cd.drop(columns=reg_cols_to_drop)
# cd = cd[clusterCols]

# last_observations = cd.groupby('Company').last().reset_index()

# # Drop non-numeric columns
# data_numeric = last_observations.drop(columns=['Company'])

# # Handle missing values (if any)
# data_numeric = data_numeric.fillna(method='ffill')

# # Standardize the data
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data_numeric)

# n_clusters = 5  # Example number of clusters
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# clusters = kmeans.fit_predict(data_scaled)
# last_observations['Cluster'] = clusters


# # Calculate similarity matrix
# similarity_matrix = cosine_similarity(data_scaled)

# # Create a DataFrame for better readability
# similarity_df = pd.DataFrame(
#     similarity_matrix, index=last_observations['Company'], columns=last_observations['Company'])


# styled_similarity = similarity_df.style.background_gradient(
#     cmap='RdPu', axis=None).format("{:.2f}")
