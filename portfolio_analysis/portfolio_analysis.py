import pandas as pd
import numpy as np


# read predicted values (output.csv from the other script)
work_dir = "data"
pred = pd.read_csv("data/output-data.csv", parse_dates=["date"])
# pred.columns = map(str.lower, pred.columns)

# select model (ridge as an example)
model = "en"

# sort stocks into deciles (10 portfolios) each month based on the predicted returns and calculate portfolio returns
# portfolio 1 is the decile with the lowest predicted returns, portfolio 10 is the decile with the highest predicted returns
# portfolio 11 is the long-short portfolio (portfolio 10 - portfolio 1)
# or you can pick the top and bottom n number of stocks as the long and short portfolios
predicted = pred.groupby(["year", "month"])[model]
pred["rank"] = np.floor(
    predicted.transform(lambda s: s.rank())
    * 10  # 10 portfolios
    / predicted.transform(lambda s: len(s) + 1)
)  # rank stocks into deciles
pred = pred.sort_values(
    ["year", "month", "rank", "permno"]
)  # sort stocks based on the rank
monthly_port = pred.groupby(["year", "month", "rank"]).apply(
    lambda df: pd.Series(
        np.average(df["stock_exret"].to_numpy(), axis=0)
    )
) # calculate the realized return for each portfolio using realized stock returns, assume equal-weighted portfolios
monthly_port = monthly_port.unstack().dropna().reset_index()  # reshape the data
monthly_port.columns = ["year", "month"] + [
    "port_" + str(x) for x in range(1, 11)
]  # rename columns
monthly_port["port_11"] = (
    monthly_port["port_10"] - monthly_port["port_1"]
)  # port 11 is the long-short portfolio

# Calculate the Sharpe ratio for long-short Portfolio
# you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
sharpe = (
    monthly_port["port_11"].mean()  # average return
    / monthly_port["port_11"].std()  # standard deviation of return, volatility
    * np.sqrt(12)  # annualized
)  # Sharpe ratio is annualized
print("Sharpe Ratio:", sharpe)

# Calculate the cumulative return of the long-short Portfolio
returns = monthly_port["port_11"].copy()
returns = returns + 1
cumulative_returns = returns.cumprod() - 1

# Max one-month loss of the long-short Portfolio
max_1m_loss = monthly_port["port_11"].min()
print("Max 1-Month Loss:", max_1m_loss)

# Calculate Drawdown of the long-short Portfolio
monthly_port["log_port_11"] = np.log(
    monthly_port["port_11"] + 1
)  # calculate log returns
monthly_port["cumsum_log_port_11"] = monthly_port["log_port_11"].cumsum(
    axis=0
)  # calculate cumulative log returns
rolling_peak = monthly_port["cumsum_log_port_11"].cummax()
drawdowns = rolling_peak - monthly_port["cumsum_log_port_11"]
max_drawdown = drawdowns.max()
print("Maximum Drawdown:", max_drawdown)