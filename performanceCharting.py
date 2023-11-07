import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import yfinance as yf
import cleanFinancialData as cDA
import datetime

fcast_dat = pd.read_excel("stock_forecasts.xlsx")

companys = fcast_dat["Company"]
company_tickers = [company + ".AX" for company in companys]
# company_tickers.append("^AXJO")


def get_stock_price(stock_symbol):
    start_date = "2023-06-30"
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(start=start_date, end=end_date)["Close"]
    return hist


def exitPrice(value, sell_price):
    if value >= 0 and value > sell_price:
        return sell_price
    elif value < 0 and value < sell_price:
        return sell_price
    else:
        return value


start_date = "2023-06-30"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

data = pd.DataFrame()
for ticker in company_tickers:
    data[ticker] = get_stock_price(ticker)

data.to_csv("stock_data.csv")

data = pd.read_csv("stock_data.csv", parse_dates=True, index_col=0)


def portfolioData(data=data, investment_threshold=0.050):

    fcast_dat["Start_Price"] = data.iloc[0].values
    fcast_dat["Forecasted_Change"] = (
        fcast_dat["Market_Cap_Forecast"] / fcast_dat["Market Cap."]) - 1

    conditions = [
        fcast_dat["Forecasted_Change"] >= investment_threshold,
        fcast_dat["Forecasted_Change"] <= -investment_threshold
    ]

    choices = [1000, -1000]

    fcast_dat["Investment"] = np.select(conditions, choices, default=0)

    investment_dollars = fcast_dat["Investment"].sum()
    data = (1 + data.pct_change()).cumprod() - 1
    data = data + 1
    data.fillna(1, inplace=True)
    fcast_dat["Suffix"] = fcast_dat["Company"] + ".AX"
    fcast_dat.set_index("Suffix", inplace=True)
    portfolio_val = data.mul(fcast_dat["Investment"])
    portfolio_val.iloc[0] = 0

    portfolio_series = portfolio_val.sum(axis=1) - investment_dollars
    portfolio_series[0] = 0
    # portfolio_series = portfolio_series[:-1]

    index_series = get_stock_price("^AXJO")
    index_series = (1 + index_series.pct_change()).cumprod() - 1
    index_series = (investment_dollars * index_series)
    index_series.iloc[0] = 0

    return portfolio_val, index_series, fcast_dat, investment_dollars, portfolio_series


def plotPortfolios(exit=False, percentage=False):

    portfolio_val, index_series, fcast_dat, investment_dollars, portfolio_series = portfolioData()

    fcast_dat["Sell_Price"] = fcast_dat["Investment"] * \
        (1 + fcast_dat["Forecasted_Change"])

    if exit:
        for company in portfolio_val.columns:
            sell_price = fcast_dat.loc[company, "Sell_Price"]
            portfolio_val[company] = portfolio_val[company].apply(
                lambda x: exitPrice(x, sell_price))

        portfolio_series = portfolio_val.sum(axis=1) - investment_dollars
        portfolio_series[0] = 0

    # print(f"Starting investment of ${investment_dollars}")

    fig, ax = plt.subplots(figsize=(14, 7))

    dates = portfolio_series.index

    primary_color = '#2F5496'
    secondary_color = '#444444'

    portfolio_series_pct = portfolio_series / investment_dollars
    index_series_pct = index_series / investment_dollars

    if percentage:
        portfolio_plot = portfolio_series_pct
        index_plot = index_series_pct
        Chart_Style = "Percentage"
    else:
        portfolio_plot = portfolio_series
        index_plot = index_series
        Chart_Style = "Dollar"

    ax.plot(dates, portfolio_plot, label='Long/Short Portfolio',
            linewidth=2.5, color=primary_color)
    ax.plot(dates, index_plot, label='ASX200',
            linewidth=2.5, color=secondary_color)

    font = {'family': 'Arial', 'size': 14}
    ax.set_xlabel('Date', fontdict=font, labelpad=10)
    ax.set_ylabel(f'{Chart_Style} Change', fontdict=font, labelpad=10)
    ax.set_title(f'Portfolio and ASX200 Index Performance with ${investment_dollars} Investment', fontdict={
                 'family': 'Arial', 'size': 16, 'weight': 'bold'}, pad=10)

    legend = ax.legend(fontsize=12, loc='upper left', frameon=True)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True)

    # background colour
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # plot border
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.axhline(y=0, color='#B1B1B1', linewidth=2, alpha=0.5, zorder=1)

    left_limit = portfolio_series.index[0] - pd.Timedelta(days=0)
    right_limit = portfolio_series.index[-1] + pd.Timedelta(days=2)
    ax.set_xlim(left_limit, right_limit)

    for i, ((date, value), (date_index, value_index)) in enumerate(zip(portfolio_plot.iteritems(), index_plot.iteritems())):
        if i % 20 == 0 and i != 0 or i == len(portfolio_series) - 1:
            ax.annotate(f'{value:.2f}', (date, value), textcoords="offset points",
                        xytext=(0, 10), ha='center', color=primary_color, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec=primary_color, lw=2)
                        )

    for i, ((date, value), (date_index, value_index)) in enumerate(zip(index_plot.iteritems(), portfolio_plot.iteritems())):
        if i % 20 == 0 and i != 0 or i == len(portfolio_series) - 1:
            ax.annotate(f'{value:.2f}', (date, value), textcoords="offset points",
                        xytext=(0, 10), ha='center', color=secondary_color, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec=secondary_color, lw=2)
                        )

    ax.tick_params(axis='both', which='major',
                   labelsize=12, direction='out', length=0)
    ax.tick_params(axis='x', which='minor', bottom=False)

    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.0, right=0.95)

    plt.savefig('/Users/tomrogers/Desktop/Python/CD_Forecasting/portfolio_performance.png',
                dpi=300, bbox_inches='tight')

    plt.show()


plotPortfolios(exit=False, percentage=False)
