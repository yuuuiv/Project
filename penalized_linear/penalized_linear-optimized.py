import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # for timing purpose
    print(datetime.datetime.now())

    # turn off pandas Setting with Copy Warning
    pd.set_option("mode.chained_assignment", None)

    # set working directory
    work_dir = "data"

    # read sample data
    file_path = os.path.join(work_dir, "stock_sample.csv")  # name of the main dataset
    raw = pd.read_csv(
        file_path, parse_dates=["date"], low_memory=False
    )  # the date is the first day of the return month

    # read list of predictors for stocks
    file_path = os.path.join(
        work_dir, "factor_char_list.csv"
    )  # name of the dataset that contains the list of predictors
    stock_vars = list(
        pd.read_csv(file_path)["variable"].values
    )  # variable is the column name

    # define the left hand side variable
    ret_var = "stock_exret"
    new_set = raw[
        raw[ret_var].notna()
    ].copy()  # create a copy of the data and make sure the left hand side is not missing

    # transform each variable in each month to the same scale
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        # rank transform each variable to [-1, 1]
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(
                var_median
            )  # fill missing values with the cross-sectional median of each month

            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            if group_max > 0:
                group[var] = (group[var] / group_max) * 2 - 1
            else:
                group[var] = 0  # in case of all missing values
                print("Warning:", date, var, "set to zero.")

        # add the adjusted values
        data = data._append(
            group, ignore_index=True
        )  # append may not work with certain versions of pandas, use concat instead if needed

    # initialize the starting date, counter, and output data
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    # estimation with expanding window
    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime(
        "20240101", format="%Y%m%d"
    ):
        cutoff = [
            starting,
            starting
            + pd.DateOffset(
                years=8 + counter
            ),  # use 8 years and expanding as the training set
            starting
            + pd.DateOffset(
                years=10 + counter
            ),  # use the next 2 years as the validation set
            starting + pd.DateOffset(years=11 + counter),
        ]  # use the next year as the out-of-sample testing set

        # cut the sample into training, validation, and testing sets
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        # get Xs and Ys
        X_train = train[stock_vars].values
        Y_train = train[ret_var].values
        X_val = validate[stock_vars].values
        Y_val = validate[ret_var].values
        X_test = test[stock_vars].values
        Y_test = test[ret_var].values

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=20)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

        # de-mean Y (because the regressions are fitted without an intercept)
        # if you want to include an intercept (or bias in neural networks, etc), you can skip this step
        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean

        # prepare output data
        reg_pred = test[
            ["year", "month", "date", "permno", ret_var]
        ]  # minimum identifications for each stock

        # Lasso with RandomizedSearchCV
        param_distributions = {'alpha': np.logspace(-2, 2, 20)}
        lasso = Lasso(max_iter=50000, fit_intercept=False)
        random_search = RandomizedSearchCV(lasso, param_distributions, n_iter=10, cv=5, n_jobs=-1)
        random_search.fit(X_train, Y_train_dm)

        best_lasso = random_search.best_estimator_
        x_pred = best_lasso.predict(X_test) + Y_mean
        reg_pred["lasso"] = x_pred

        # Ridge with GridSearchCV
        param_grid = {'alpha': [0.01, 0.1, 1, 10]}
        ridge = Ridge(fit_intercept=False)
        grid_search = GridSearchCV(ridge, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, Y_train_dm)

        best_ridge = grid_search.best_estimator_
        x_pred = best_ridge.predict(X_test) + Y_mean
        reg_pred["ridge"] = x_pred

        # ElasticNet with RandomizedSearchCV
        param_distributions = {'alpha': np.logspace(-2, 2, 20)}
        en = ElasticNet(max_iter=50000, fit_intercept=False)
        random_search = RandomizedSearchCV(en, param_distributions, n_iter=10, cv=5, n_jobs=-1)
        random_search.fit(X_train, Y_train_dm)

        best_en = random_search.best_estimator_
        x_pred = best_en.predict(X_test) + Y_mean
        reg_pred["en"] = x_pred

        # add to the output data
        pred_out = pred_out._append(reg_pred, ignore_index=True)

        # go to the next year
        counter += 1

    # output the predicted value to csv
    out_path = os.path.join(work_dir, "output.csv")
    print(out_path)
    pred_out.to_csv(out_path, index=False)

    # print the OOS R2
    yreal = pred_out[ret_var].values  # the real value of the left hand side variable
    for model_name in ["lasso", "ridge", "en"]:
        ypred = pred_out[
            model_name
        ].values  # the predicted value of the left hand side variable
        r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
        print(model_name, r2)

    # for timing purpose
    print(datetime.datetime.now())
