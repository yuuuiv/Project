import datetime
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import os

# 优化数据处理部分
# 设置工作目录
work_dir = "data"

# 读取数据
file_path = os.path.join(work_dir, "stock_sample.csv")  # 主数据集路径
raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)  # 优化低内存模式

# 读取预测变量列表
file_path = os.path.join(work_dir, "factor_char_list.csv")
stock_vars = pd.read_csv(file_path)["variable"].values.tolist()  # 转换为列表

# 左侧变量定义
ret_var = "stock_exret"
raw = raw[raw[ret_var].notna()].copy()  # 筛选非空的目标变量数据

# 优化内存使用：调整数据类型
for col in raw.select_dtypes(include=['float']).columns:
    raw[col] = raw[col].astype('float32')
for col in raw.select_dtypes(include=['int']).columns:
    raw[col] = raw[col].astype('int32')

# 分组处理每个月的数据
def process_monthly_data(date, group):
    group = group.copy()
    # 特征处理：填充缺失值，按分位数 [-1, 1] 缩放
    for var in stock_vars:
        var_median = group[var].median(skipna=True)
        group[var] = group[var].fillna(var_median)
        group[var] = group[var].rank(method="dense") - 1
        max_val = group[var].max()
        if max_val > 0:
            group[var] = (group[var] / max_val) * 2 - 1
        else:
            group[var] = 0  # 如果数据全为缺失值，直接置为 0
    return group

# 并行处理数据
processed_data = Parallel(n_jobs=-1, backend="loky")(
    delayed(process_monthly_data)(date, group) for date, group in raw.groupby("date")
)
data = pd.concat(processed_data, ignore_index=True)  # 合并所有处理后的数据

# 输出优化后的数据规模和部分统计信息
print(f"数据规模：{data.shape}")
print(data.describe())

# 数据保存为高效格式
data.to_parquet(os.path.join(work_dir, "processed_data.parquet"), index=False)

# 模型训练和预测部分
if __name__ == "__main__":
    print(datetime.datetime.now())

    # 读取处理后的数据
    data = pd.read_parquet(os.path.join(work_dir, "processed_data.parquet"))

    # 初始化开始时间、计数器和输出数据
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("20240101", format="%Y%m%d"):
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8 + counter),  # 使用 8 年作为训练集
            starting + pd.DateOffset(years=10 + counter),  # 使用接下来的 2 年作为验证集
            starting + pd.DateOffset(years=11 + counter),  # 使用接下来的 1 年作为测试集
        ]

        # 分割数据集
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        X_train = train[stock_vars].values
        Y_train = train[ret_var].values
        X_val = validate[stock_vars].values
        Y_val = validate[ret_var].values
        X_test = test[stock_vars].values
        Y_test = test[ret_var].values

        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean

        reg_pred = test[["year", "month", "date", "permno", ret_var]].copy()

        # 线性回归
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred.loc[:, "ols"] = x_pred

        # Lasso 回归
        lambdas = np.arange(-4, 4.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Lasso(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Lasso(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred.loc[:, "lasso"] = x_pred

        # Ridge 回归
        lambdas = np.arange(-1, 8.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Ridge(alpha=((10**i) * 0.5), fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Ridge(alpha=((10**best_lambda) * 0.5), fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred.loc[:, "ridge"] = x_pred

        # Elastic Net 回归
        lambdas = np.arange(-4, 4.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = ElasticNet(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = ElasticNet(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred.loc[:, "en"] = x_pred

        pred_out = pred_out._append(reg_pred, ignore_index=True)

        counter += 1

    # 输出模型性能
    yreal = pred_out[ret_var].values
    for model_name in ["ols", "lasso", "ridge", "en"]:
        ypred = pred_out[model_name].values
        r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
        print(f"{model_name} {r2}")

    out_path = os.path.join(work_dir, "output.csv")
    print(out_path)
    pred_out.to_csv(out_path, index=False)
    print(datetime.datetime.now())