import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# 读取预测值
work_dir = "data"
pred = pd.read_csv(f"{work_dir}/output-double.csv", parse_dates=["date"])

# 选择模型（如 XGBoost_pred）
model = "XGBoost_pred"

# 参数：每月选股范围
min_stocks = 50
max_stocks = 100

# 分组逻辑：选取预测收益排名前/后的股票
def select_stocks(df, long_top=True, n_min=min_stocks, n_max=max_stocks):
    df = df.sort_values(model, ascending=not long_top)  # 按预测值排序
    selected = df.head(n_max) if len(df) > n_max else df
    if len(selected) < n_min:  # 如果不足最小数量，补充到 n_min
        selected = df.head(n_min)
    return selected

# 动态权重分配
def calculate_weights(df):
    abs_sum = df[model].abs().sum()
    if abs_sum > 0:
        df["weight"] = df[model] / abs_sum
    else:
        df["weight"] = 0

    # 限制权重范围，避免极端值
    df["weight"] = df["weight"].clip(lower=-0.1, upper=0.1)
    return df

# 投资组合策略计算
results = []
ic_values = []
for (year, month), group in pred.groupby([pred["date"].dt.year, pred["date"].dt.month]):
    # 长多策略
    long_stocks = select_stocks(group, long_top=True)
    long_stocks = calculate_weights(long_stocks)
    long_return = (long_stocks["weight"] * long_stocks["permno"]).sum()

    # 短多策略
    short_stocks = select_stocks(group, long_top=False)
    short_stocks = calculate_weights(short_stocks)
    short_return = (short_stocks["weight"] * short_stocks["permno"]).sum()

    # 多空策略
    long_short_return = long_return - short_return

    # 计算信息系数 (IC)
    if len(group) > 1:
        ic, _ = spearmanr(group[model], group["permno"])
    else:
        ic = np.nan  # 如果样本量不足，IC无法计算

    ic_values.append(ic)

    results.append({
        "year": year,
        "month": month,
        "long_return": long_return,
        "short_return": short_return,
        "long_short_return": long_short_return,
    })

# 转换为 DataFrame
results_df = pd.DataFrame(results)
results_df["IC"] = ic_values

# 使用对数收益计算
results_df["log_long"] = np.log(1 + results_df["long_return"])
results_df["log_short"] = np.log(1 + results_df["short_return"])
results_df["log_long_short"] = np.log(1 + results_df["long_short_return"])

# 确保收益值大于 -1
results_df["log_long_short"] = np.log(1 + results_df["long_short_return"].clip(lower=-0.99))

# 确保对数收益计算时，1 + return 的值大于零
results_df["log_long_short"] = np.where(
    results_df["long_short_return"] > -1,  # 仅对 return > -1 使用对数收益
    np.log(1 + results_df["long_short_return"]),
    0  # 如果 return <= -1，设为 0 或其他合理值
)

# 累计对数收益
results_df["cumulative_log_long"] = results_df["log_long"].cumsum()  # 对数收益的累积
results_df["cumulative_log_short"] = results_df["log_short"].cumsum()
results_df["cumulative_log_long_short"] = results_df["log_long_short"].cumsum()

# Sharpe Ratio 计算（使用对数收益）
log_returns = results_df["log_long_short"]
std_dev = log_returns.std()  # 计算对数收益的标准差
mean_return = log_returns.mean()  # 计算对数收益的均值

# 如果标准差非常小，避免除以零的情况
if std_dev != 0:
    sharpe_ratio = mean_return / std_dev * np.sqrt(12)
else:
    sharpe_ratio = np.nan  # 如果标准差为零，无法计算 Sharpe Ratio

print("Sharpe Ratio:", sharpe_ratio)

# Max 1-Month Loss
max_1m_loss = results_df["log_long_short"].min()
print("Max 1-Month Loss:", max_1m_loss)

# Maximum Drawdown
rolling_peak = results_df["cumulative_log_long_short"].cummax()  # 对数收益的累计最大值
drawdown = rolling_peak - results_df["cumulative_log_long_short"]
max_drawdown = drawdown.max()
print("Maximum Drawdown:", max_drawdown)

# 保存结果
results_df.to_csv(f"{work_dir}/strategy-XGB_IC.csv", index=False)

print("Results saved to strategy-XGB_IC.csv")

# 打印统计描述
print(results_df["log_long_short"].describe())

# 可视化 log 长空策略收益
plt.figure(figsize=(10, 6))
plt.plot(results_df["cumulative_log_long_short"], label="Cumulative Log Return (Long-Short)")
plt.title("Cumulative Log Return of Long-Short Strategy")
plt.xlabel("Months")
plt.ylabel("Cumulative Log Return")
plt.legend()
plt.show()

# 可视化收益分布
plt.hist(results_df["log_long_short"], bins=50)
plt.title("Distribution of Log Returns (Long-Short)")
plt.xlabel("Log Return")
plt.ylabel("Frequency")
plt.show()
