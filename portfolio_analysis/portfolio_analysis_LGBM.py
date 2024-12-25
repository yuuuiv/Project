import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 读取CSV文件
work_dir = "data"
file_path = f"{work_dir}/output-cv=3.csv"  # 替换为实际文件路径
data = pd.read_csv(file_path, parse_dates=["date"])  # 确保 date 列为日期格式

# 参数设置
model = "LightGBM_pred"
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

# 分组数据并进行分析
results = []
for (year, month), group in data.groupby([data["date"].dt.year, data["date"].dt.month]):
    # 长多策略
    long_stocks = select_stocks(group, long_top=True)
    long_stocks = calculate_weights(long_stocks)
    long_return = (long_stocks["weight"] * long_stocks[model]).sum()

    # 短多策略
    short_stocks = select_stocks(group, long_top=False)
    short_stocks = calculate_weights(short_stocks)
    short_return = (short_stocks["weight"] * short_stocks[model]).sum()

    # 多空策略
    long_short_return = long_return - short_return

    results.append({
        "year": year,
        "month": month,
        "long_return": long_return,
        "short_return": short_return,
        "long_short_return": long_short_return,
    })

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 计算累计收益
results_df["cumulative_long"] = (1 + results_df["long_return"]).cumprod() - 1
results_df["cumulative_short"] = (1 + results_df["short_return"]).cumprod() - 1
results_df["cumulative_long_short"] = (1 + results_df["long_short_return"]).cumprod() - 1

# 计算Sharpe Ratio
returns = results_df["long_short_return"]
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12)  # 年化
print("Sharpe Ratio:", sharpe_ratio)

# 计算最大回撤
rolling_peak = results_df["cumulative_long_short"].cummax()
drawdown = rolling_peak - results_df["cumulative_long_short"]
max_drawdown = drawdown.max()
print("Maximum Drawdown:", max_drawdown)

# 保存结果
results_df.to_csv(f"{work_dir}/strategy-LGBM.csv", index=False)
print("strategy-LGBM.csv")

# 绘制累计收益图
plt.figure(figsize=(10, 6))
plt.plot(results_df["cumulative_long_short"], label="Cumulative Long-Short Return")
plt.title("Portfolio Analysis")
plt.xlabel("Months")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()
