import datetime
import yfinance as yf
from collections import Counter
import datetime as dt
import openpyxl
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
import warnings


pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option('display.max_columns', 25)

file_path = "/Users/tomrogers/Desktop/Python/CD_Forecasting/Financial_Data/"
sheets = ["Profit Loss", "Balance Sheet", "Cash Flow", "Ratio Analysis"]

comps = []
data = {}
i = 0
for file in os.listdir(file_path):
    if file.endswith(".xlsx"):
        for sheet in sheets:
            i = i+1
            df = pd.read_excel(f"Financial_Data/{file}", sheet_name=sheet)
            name = df["ASX Code"][0]
            df = df.T
            df["Company"] = df.iloc[0, 0]
            df.columns = df.iloc[2]
            df.columns = df.columns.str.title()
            df = df.iloc[3:]
            df.columns = [*df.columns[:-1], 'Company']
            df.rename(columns={"Payment For Purchase Of Subsidiaries": "Payments For Purchase Of Subsidiaries",
                               "Loan Granted": "Loans Granted",
                               "Loan Repaid": "Loans Repaid",
                               "Cl - Total Assets": "Total Assets",
                               "Ebit Margin (%": "Ebit Margin (%)",
                               "Working Cap Turnover": "Working Cap Turnover (%)",
                               "Year End Share Price": "Share Price",
                               "Interim Share Price": "Share Price"
                               }, inplace=True)
            start_date = pd.to_datetime(df.index[0], format="%m/%y")
            end_date = pd.to_datetime(df.index[-1], format="%m/%y")
            df.index = pd.date_range(
                start=start_date, end=end_date, periods=len(df.index))
            data[f"{name}_{sheet}_{i}"] = df


def extract_info(key):
    parts = key.split('_')
    return parts[0], parts[1]


grouped_keys = {}
for key in data.keys():
    ticker, dataset_type = extract_info(key)
    if (ticker, dataset_type) not in grouped_keys:
        grouped_keys[(ticker, dataset_type)] = []
    grouped_keys[(ticker, dataset_type)].append(key)

result = {}
for (ticker, dataset_type), keys in grouped_keys.items():
    concatenated_df = pd.concat([data[key] for key in keys])

    result[(ticker, dataset_type)] = concatenated_df.sort_index()

for key, value in result.items():
    result[key].index = [date.replace(month=6, day=30) if date.month in [5, 6] else date.replace(
        month=12, day=31) if date.month in [11, 12] else date for date in value.index]
    # result[key].index = value.index.date

for key, value in result.items():
    p = len(value.index)
    date_index = pd.date_range(end="30-06-2023", freq="6M", periods=p)
    result[key].index = date_index

grouped_data = {}
for (ticker, report_type), df in result.items():
    if ticker not in grouped_data:
        grouped_data[ticker] = []
    grouped_data[ticker].append(df)

result = {}
for ticker, dfs in grouped_data.items():
    result[ticker] = pd.concat(dfs, axis=1)

df_full = pd.concat(result, axis=1)
forecasting_df = df_full.loc[:, ~df_full.columns.duplicated()].stack(level=0)
forecasting_df = forecasting_df[forecasting_df["Market Cap."] != 0]
na_sums = forecasting_df.isna().sum().sort_index()
cols_to_exclude = ['Abnormals', 'Abnormals Tax', 'Asset Turnover', 'Bv Per Share ($)', 'Ca - Prepaid Expenses', 'Capex/Operating Rev. (%)', 'Cash At Beginning Of Period', 'Cash At End Of Period',
                   'Creditors/Op. Rev. (%)', 'Days Inventory', 'Days Payables', 'Days Receivables', 'Depreciation/Capex (%)', 'Depreciation/Pp&E (%)', 'Depreciation/Revenue (%)', 'Ebit Margin (%)',
                   'Ebita Margin (%)', 'Ebitda Margin (%)', 'Eps Adjusted (Cents/Share)', 'Eps After Abnormals (Cents/Share)', 'Ev/Ebit', 'Ev/Ebitda', 'Exchange Rate Adj', 'Funds From Ops./Ebitda (%)',
                   'Gross Cf Per Share ($)', 'Gross Debt/Cf', 'Gross Gearing (D/E) (%)', 'Inventory Turnover', 'Inventory/Trading Rev. (%)', 'Invested Capital Turnover', 'Investments Purchased',
                   'Lt Asset Turnover', 'Market Cap./Rep Npat', 'Net Abnormals', 'Net Financing Cashflows', 'Net Gearing (%)', 'Net Interest Cover', 'Net Profit After Tax Before Abnormals', 'Noplat Margin (%)',
                   'Nta Per Share ($)', 'Other Cash Adjustments', 'Other Financing Cashflows', 'Other Investing Cashflows', 'Other Operating Cashflows', 'Ppe Turnover', 'Price/Book Value', 'Price/Gross Cash Flow',
                   'Repayment Of Borrowings', 'Reported Npat After Abnormals', 'Roa (%)', 'Roe (%)', 'Roic (%)', 'Sales Per Share ($)', 'Share Price', 'Shares Outstanding At Period End', 'Total Revenue Excluding Interest',
                   'Weighted Average Number Of Shares', 'Wkg Capital/Revenue (%)', 'Working Cap Turnover (%)']
f_df = forecasting_df.drop(cols_to_exclude, axis=1)
rows_to_drop = f_df.isna().sum(axis=1)
rows_to_drop = rows_to_drop[rows_to_drop > 0].index
f_df = f_df.drop(list(rows_to_drop), axis=0)
f_df.replace("--", 0, inplace=True)
f_df_corr = f_df.corr()

threshold = 0.75

correlated_pairs = {}

for i in range(f_df_corr.shape[0]):
    for j in range(i+1, f_df_corr.shape[1]):
        if abs(f_df_corr.iloc[i, j]) > threshold:
            correlated_pairs[(f_df_corr.columns[i],
                              f_df_corr.columns[j])] = f_df_corr.iloc[i, j]

correl_cols_to_drop = ["Depreciation", "Amortisation", "Ebit", "Total Liabilities", "Total Nca",
                       "Total Curr. Liabilities", "Total Current Assets", "Enterprise Value",
                       "Nca - Goodwill", "Nca - Intangibles(Exgw)", "Proceeds From Sale Of Ppe"
                       ]

f_df = f_df.drop(correl_cols_to_drop, axis=1)

macro_data = pd.read_csv("full_macro_forecast.csv",
                         parse_dates=True, index_col=0)

macro_data = macro_data[macro_data.index.month.isin([6, 12])]
macro_data.iloc[-1, -1] = 0

cols_to_drop = ["Receivables/Op. Rev. (%)", "Se Held Sale", "Total Ncl", "Nca - Future Tax Benefit", 'Ca - Nca Held Sale', 'Payments For Purchase Of Subsidiaries',
                "Total Assets", "Total Equity", "Outside Equity Interests", "Total Ncl", "Company", "Convertible Equity", "Financial Leverage", "Interest Expense",
                "Interest Revenue", 'Market Cap./Trading Rev.', 'Nca - Inventories']

f_df = f_df.drop(cols_to_drop, axis=1)
dup_entries = f_df[f_df.index.get_level_values(1) == "ALL"].iloc[::2].index
f_df.drop(dup_entries, inplace=True)

pct_thresh = 0.5
zero_cols_to_remove = f_df.loc[:, (f_df.apply(
    lambda x: x == 0).sum() / f_df.shape[0] > pct_thresh)].columns

f_df.drop(zero_cols_to_remove, axis=1, inplace=True)
f_df["Market_Cap_Forecast"] = f_df.groupby(level=1)["Market Cap."].shift(-1)

macro_data.index = macro_data.index + pd.DateOffset(months=-6)
macro_data.index = macro_data.index + \
    pd.to_timedelta(np.where(macro_data.index.month == 12, 1, 0), unit='D')

f_df.reset_index(inplace=True)
f_df.set_index("level_0", inplace=True)
df_forecast = pd.merge(f_df, macro_data, left_index=True, right_index=True)
df_forecast.reset_index(inplace=True)
df_forecast.rename(columns={"index": "Date",
                            "level_1": "Company"},
                   inplace=True)
df_forecast.set_index(["Date"], inplace=True)
df_forecast["GFC_Recession"] = 0
df_forecast["GFC_Recession"].loc["2007-06-30":"2008-12-31"] = 1
df_forecast['Market Cap Quartiles'] = pd.qcut(
    df_forecast['Market Cap.'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
dummy_vars = pd.get_dummies(
    df_forecast['Market Cap Quartiles'], drop_first=True, prefix='Quartile')
data_forecast = pd.concat([df_forecast, dummy_vars], axis=1)
data_forecast.drop("Market Cap Quartiles", axis=1, inplace=True)
data_forecast['changeATH'] = data_forecast.groupby(
    'Company')['Market Cap.'].transform('max') - data_forecast['Market Cap.']
data_forecast['Market_Cap_Lag'] = data_forecast.groupby("Company")[
    "Market Cap."].shift(1)
# data_forecast.dropna(inplace=True)


def get_stock_price(stock_symbol, start_date, end_date):
    start_date = start_date
    end_date = end_date
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(start=start_date, end=end_date)["Close"]
    return hist


data_forecast["Ticker"] = data_forecast["Company"] + ".AX"
sd = data_forecast.index[0] + pd.DateOffset(months=-6, days=1)
price_data = pd.DataFrame()
for ticker in np.unique(data_forecast['Ticker']):
    price_data[ticker] = get_stock_price(
        ticker, start_date=sd, end_date=data_forecast.index[-1])
daily_returns = price_data.pct_change()
daily_returns.index = daily_returns.index.tz_localize(None)
data_forecast.index = data_forecast.index.tz_localize(None)
data_forecast["Rolling_Std"] = data_forecast.apply(
    lambda row: daily_returns.loc[daily_returns[row['Ticker']].first_valid_index(): pd.to_datetime(row.name), row['Ticker']].std(), axis=1)
data_forecast.dropna(subset=['Rolling_Std'], inplace=True)
data_forecast["Market_Cap_Change"] = data_forecast.groupby("Company")[
    "Market Cap."].pct_change()

data_extra_lag = data_forecast.copy()
data_extra_lag['Market_Cap_Lag'] = data_extra_lag.groupby("Company")[
    "Market Cap."].shift(1)
data_extra_lag.dropna(inplace=True)

df_forecast_dropped = data_forecast.loc[:"2022-12-31"]
df_forecast_dropped_MCLAG = data_extra_lag.loc[:"2022-12-31"]


def joinPredictions(predicted_series, actual_series):
    forecasting_df["Market_Cap_Forecast"] = forecasting_df.groupby(level=1)[
        "Market Cap."].shift(-1)
    test_df = pd.merge(pd.DataFrame(actual_series).reset_index(), forecasting_df.reset_index(), left_on=["Date", "Market_Cap_Forecast"], right_on=[
                       "level_0", "Market_Cap_Forecast"])[["Date", "Company", "Market Cap.", "Market_Cap_Forecast", "Shares Outstanding At Period End"]]
    test_df["Pred"] = predicted_series
    test_df["Shares Outstanding At Period End"] = pd.to_numeric(
        test_df["Shares Outstanding At Period End"], errors='coerce')
    test_df = test_df.dropna(subset=["Shares Outstanding At Period End"])
    test_df["Pred_Share_Price"] = test_df["Pred"] / \
        test_df["Shares Outstanding At Period End"]
    test_df["Act_Share_Price"] = test_df["Market_Cap_Forecast"] / \
        test_df["Shares Outstanding At Period End"]

    return test_df


# Feature engineering
# Adjust macroeconomic forecasts
# Timeseries factors
# Ensemble of models
# Years since ATH?
# Volatility of share price to date?
# Material announcements during period flag?
# Might be worth doing a cluster analysis to see if the companies with high forecasted growth are 'statistically' in the same group
