from pybats.plot import *
from pybats.point_forecast import median
from pybats.analysis import analysis
from pybats.shared import load_sales_example
import statsmodels.api as sm
import pandas as pd
import numpy as np
import glob
import os
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path_to_files = "/Users/tomrogers/Desktop/Python/CD_Forecasting/Economic_Indicators.xlsx"

df = pd.read_excel(path_to_files, parse_dates=True)
series_names = df.columns[1::2]

data = [pd.DataFrame(index=df.iloc[:, i], data=df.iloc[:, i+1].values)
        for i in range(0, df.shape[1], 2)]

data = [data.dropna() for data in data]

for i in range(len(data)):
    data[i].index = pd.to_datetime(data[i].index)
    data[i] = data[i].loc['2000-01-01':'2023-07-01']

for df in data:
    df.index = df.index.to_period('M').to_timestamp(how='end')

for i, df in enumerate(data):
    df.columns = [f'col_{i}']

result = pd.concat(data, axis=1)

result["col_1"] = result["col_1"].shift(-1)

result.index = result.index.normalize()

result = result.resample('M').last()

result = result.fillna(method='ffill')
result = result[result.index.month.isin([3, 6, 9, 12])]

result.columns = series_names

result = result.drop(['Australia Consumer Prices, All Items, Total, Index, 2011-2012=100',
                      "Australia Refinitiv / Ipsos Primary Consumer Sentiment Index (CSI), Index (diffusion)",
                      "Australia BUILDING APPROVALS: NEW HOUSES, Chg Y/Y",
                      "Australia COMMONWEALTH GOVERNMENT BOND YIELD 10 YEAR (EP), Not SA",
                      "Australia Estimated resident population, total"
                      ], axis=1)

cols = ["Cash Rate", "CPI YoY", "Savings Ratio", "Consumer Spending", "Unemployment Rate",
        "Avg. Weekly Earnings", "Mortgage Lending Rates", "Unit Labour Cost"]

result.columns = cols
result = result.dropna()

fig, axes = plt.subplots(len(result.columns), 1, figsize=(6, 14))

for ax, col in zip(axes, result.columns):
    result[col].plot(ax=ax, title=col, legend=False)
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()

correlation_matrix = result.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)

start_date = '2020-03-01'
end_date = '2021-12-31'
result['Covid_dummy'] = 0
result.loc[start_date:end_date, 'Covid_dummy'] = 1


pred_col = "Cash Rate"
lagged_df = result.copy()
forecast_length = 2
dates_extended = pd.date_range(start=lagged_df.index[0], periods=len(
    lagged_df) + forecast_length, freq='3M')
lagged_df = lagged_df.reindex(dates_extended)
lagged_cols = []
for col in lagged_df.columns:
    if col not in ['Covid_dummy']:
        lagged_df[f'{col}_lag2'] = lagged_df[col].shift(2)
        # lagged_df[f'{col}_lag3'] = lagged_df[col].shift(3)
        lagged_cols.append(col)
lagged_cols.remove(pred_col)
lagged_df.drop(lagged_cols, axis=1, inplace=True)
# lagged_df = lagged_df.apply(lambda x: x.shift(2) if x.name not in ['Covid_dummy', "Cash Rate"] else x)
lagged_df.loc[pd.to_datetime("2023-09-30"):, "Covid_dummy"] = 0
lagged_df = lagged_df[3:]
pd.concat([lagged_df.head(3), lagged_df.tail(3)])
print(lagged_df.shape)

forecast_df_res = pd.DataFrame(columns=["sept", "dec"])

forecast_length = 2

for col in result.columns:

    if col not in ['Covid_dummy']:

        pred_col = col

        print(pred_col)

        lagged_df = result.copy()

        dates_extended = pd.date_range(start=lagged_df.index[0], periods=len(
            lagged_df) + forecast_length, freq='3M')

        lagged_df = lagged_df.reindex(dates_extended)

        lagged_cols = []
        for col in lagged_df.columns:
            if col not in ['Covid_dummy']:
                lagged_df[f'{col}_lag2'] = lagged_df[col].shift(2)
                # lagged_df[f'{col}_lag3'] = lagged_df[col].shift(3)
                lagged_cols.append(col)

        lagged_cols.remove(pred_col)
        lagged_df.drop(lagged_cols, axis=1, inplace=True)
        lagged_df.loc[pd.to_datetime("2023-09-30"):, "Covid_dummy"] = 0
        lagged_df = lagged_df[3:]

        grid_df = pd.DataFrame(columns=["prior_length", "rho", "mae", "rmse"])

        for p_l in np.ceil(np.linspace(lagged_df.shape[1], 50, 10)):
            p_l = int(p_l)
            for r in (np.linspace(0, 1, 10)):
                Y = lagged_df[pred_col].values
                X = lagged_df.drop(pred_col, axis=1)
                prior_length = p_l
                k = forecast_length
                rho = r
                s = 1000
                fs = pd.to_datetime("2022-09-30")
                fe = pd.to_datetime("2023-06-30")

                mod, samples, model_coef = analysis(Y, X.values,
                                                    k, fs, fe, nsamps=s,
                                                    family='normal',
                                                    prior_length=prior_length, dates=lagged_df.index,
                                                    rho=rho,
                                                    ret=['model', 'forecast', 'model_coef'])

                pred = median(samples[:, :, 0])
                act = lagged_df[pred_col].loc[fs:fe]

                errors = pred - act
                mae = np.abs(errors).mean()
                rmse = np.sqrt((errors**2).mean())

                grid_df.loc[len(grid_df)] = [p_l, r, mae, rmse]

        opt_length = grid_df.sort_values(
            "rmse").iloc[0]["prior_length"].astype(int)
        opt_rho = grid_df.sort_values("rmse").iloc[0]["rho"]

        mod, samples, model_coef = analysis(Y, X.values,
                                            k, fs, fe, nsamps=s,
                                            family='normal',
                                            prior_length=opt_length, dates=lagged_df.index,
                                            rho=opt_rho,
                                            ret=['model', 'forecast', 'model_coef'])

        x_future = X.iloc[-2:].values
        two_step_forecast_samples = mod.forecast_path(
            k=2, X=x_future, nsamps=1000)
        median_forecast = np.median(two_step_forecast_samples, axis=0)

        forecast_df_res.loc[pred_col] = [
            median_forecast[0], median_forecast[1]]

    else:

        print(col)
        continue

forecast_df_res = forecast_df_res.T

forecast_df_res.index = pd.date_range("2023-09-30", freq='3M', periods=2)
opt_length = grid_df.sort_values("rmse").loc[0]["prior_length"].astype(int)
opt_rho = grid_df.sort_values("rmse").loc[0]["rho"]

mod, samples, model_coef = analysis(Y, X.values,
                                    k, fs, fe, nsamps=s,
                                    family='normal',
                                    prior_length=opt_length, dates=lagged_df.index,
                                    rho=opt_rho,
                                    ret=['model', 'forecast', 'model_coef'])


def plot_eco_forecast(col_name):

    fig, ax = plt.subplots(1, 1)

    ax.plot(result.index, result[col_name], label="Actual", color="black")
    ax.plot(forecast_df_res.index, forecast_df_res[col_name],
            label="Forecast", color="#0074FF", ls="-", lw=1.5, ms=5, marker="s")
    std_res = result[col_name].std()
    upper = forecast_df_res[col_name] + std_res
    lower = forecast_df_res[col_name] - std_res
    # lower_bound = np.percentile(two_step_forecast_samples, 2.5, axis=0)
    # upper_bound = np.percentile(two_step_forecast_samples, 97.5, axis=0)
    ax.fill_between(forecast_df_res.index, lower, upper, color='#ADE6FF')

    print(forecast_df_res[col_name])

    ax.legend()
    return plt.show()


plot_eco_forecast("Unit Labour Cost")
full_forecast_df = pd.concat([result, forecast_df_res])
full_forecast_df.to_csv("full_macro_forecast.csv")
