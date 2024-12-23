import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 读取预测值
work_dir = "data"
pred = pd.read_csv(f"{work_dir}/output-double.csv", parse_dates=["date"])

# 选择模型（如 RandomForest_pred）
model = "RandomForest_pred"

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

# 计算性能指标
results_df["cumulative_long"] = (1 + results_df["long_return"].clip(lower=0.01, upper=10)).cumprod() - 1
results_df["cumulative_short"] = (1 + results_df["short_return"].clip(lower=0.01, upper=10)).cumprod() - 1
results_df["cumulative_long_short"] = (1 + results_df["long_short_return"].clip(lower=0.01, upper=10)).cumprod() - 1

# Sharpe Ratio
sharpe_ratio = results_df["long_short_return"].mean() / results_df["long_short_return"].std() * np.sqrt(12)
print("Sharpe Ratio:", sharpe_ratio)

# Max 1-Month Loss
max_1m_loss = results_df["long_short_return"].min()
print("Max 1-Month Loss:", max_1m_loss)

# Maximum Drawdown
rolling_peak = results_df["cumulative_long_short"].cummax()
drawdown = rolling_peak - results_df["cumulative_long_short"]
max_drawdown = drawdown.max()
print("Maximum Drawdown:", max_drawdown)

# 保存结果
results_df.to_csv(f"{work_dir}/strategy-RF_IC.csv", index=False)

print("Results saved to strategy-RF_IC.csv")
