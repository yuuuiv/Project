import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import os
from sklearn.metrics import mean_squared_error
import warnings

# 禁用 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

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

# 直接使用数据，无需进一步处理
data = raw.copy()
data = data.sort_values(by="date")  # 按时间排序

if __name__ == "__main__":
    print(datetime.datetime.now())

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

        # LightGBM模型定义（不使用 GPU）
        lgb_model = lgb.LGBMRegressor(
            random_state=42,
            verbosity=-1,  # 禁用所有日志输出
            force_col_wise=True  # 强制使用列方向多线程计算，减少内存开销
        )

        # 超参数网格
        param_grid = {
            "n_estimators": [100],
            "max_depth": [5],
            "learning_rate": [0.01],
            "num_leaves": [31],
        }

        # 使用 GridSearchCV 进行超参数搜索
        grid_search = GridSearchCV(lgb_model, param_grid, scoring="neg_mean_squared_error", cv=2, n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_

        print(f"LightGBM最佳模型参数: {grid_search.best_params_}, 验证集MSE: {-grid_search.best_score_}")

        # 使用最佳模型预测测试集
        lgb_test_pred = best_model.predict(X_test)

        # 创建预测结果 DataFrame
        temp_df = test[["date", "permno"]].copy()
        temp_df["LightGBM_pred"] = lgb_test_pred  # 模型的预测值（用于分组逻辑）
        temp_df["stock_exret_pred"] = lgb_test_pred  # 同样是预测值，具体分析时用
        temp_df["actual_stock_exret"] = Y_test       # 实际值

        # 将结果追加到 pred_out
        pred_out = pd.concat([pred_out, temp_df], ignore_index=True)

        counter += 1

    # 保存结果
    output_path = os.path.join(work_dir, "output-LGBM-B-cv=2.csv")
    pred_out.to_csv(output_path, index=False)
    print(f"结果已保存至：{output_path}")

    print(datetime.datetime.now())
