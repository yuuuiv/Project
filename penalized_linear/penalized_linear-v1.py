import datetime
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    print(datetime.datetime.now())
    pd.set_option("mode.chained_assignment", None)

    work_dir = "data"
    file_path = os.path.join(work_dir, "stock_sample.csv")
    raw = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)
    
    file_path = os.path.join(work_dir, "factor_char_list.csv")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    ret_var = "stock_exret"
    new_set = raw[raw[ret_var].notna()].copy()

    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(var_median)
            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            group[var] = (group[var] / group_max) * 2 - 1 if group_max > 0 else 0
        data = data._append(group, ignore_index=True)

    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("20240101", format="%Y%m%d"):
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8 + counter),
            starting + pd.DateOffset(years=10 + counter),
            starting + pd.DateOffset(years=11 + counter),
        ]

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

        reg_pred = test[["year", "month", "date", "permno", ret_var]]

        # Random Forest
        rf_params = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}
        best_mse = float("inf")
        for n_estimators in rf_params["n_estimators"]:
            for max_depth in rf_params["max_depth"]:
                reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                reg.fit(X_train, Y_train_dm)
                mse = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)
                if mse < best_mse:
                    best_mse = mse
                    best_rf = reg
        reg_pred["rf"] = best_rf.predict(X_test) + Y_mean

        # Gradient Boosting
        gb_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 10]}
        best_mse = float("inf")
        for n_estimators in gb_params["n_estimators"]:
            for learning_rate in gb_params["learning_rate"]:
                for max_depth in gb_params["max_depth"]:
                    reg = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                    reg.fit(X_train, Y_train_dm)
                    mse = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)
                    if mse < best_mse:
                        best_mse = mse
                        best_gb = reg
        reg_pred["gb"] = best_gb.predict(X_test) + Y_mean

        pred_out = pred_out._append(reg_pred, ignore_index=True)
        counter += 1

    out_path = os.path.join(work_dir, "output-v1.csv")
    pred_out.to_csv(out_path, index=False)

    yreal = pred_out[ret_var].values
    for model_name in ["ols", "lasso", "ridge", "en", "rf", "gb"]:
        ypred = pred_out[model_name].values
        r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
        print(model_name, r2)

    print(datetime.datetime.now())