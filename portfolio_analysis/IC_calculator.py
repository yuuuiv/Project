import pandas as pd

# 读取预测值文件（这里需要根据实际情况改一下是哪个csv）
work_dir = "data"
pred = pd.read_csv(work_dir, "output.csv")

# 计算每个月的IC
ic_results = pred.groupby(["year", "month"]).apply(
    lambda group: group["predicted"].corr(group["actual"])
)
print("Monthly IC:", ic_results)

# 计算平均IC
mean_ic = ic_results.mean()
print("Mean IC:", mean_ic)

# 计算IC的标准差
std_ic = ic_results.std()
print("Standard Deviation of IC:", std_ic)