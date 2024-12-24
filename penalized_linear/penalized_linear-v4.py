import datetime
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from skopt import BayesSearchCV
from scipy.signal import savgol_filter

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

    # Smooth data to reduce noise
    for var in stock_vars:
        if var in data.columns:
            data[var] = savgol_filter(data[var], window_length=5, polyorder=2)
    print("Data smoothing completed.")

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

        # Feature selection based on importance from Random Forest
        print("Starting feature selection using Random Forest...")
        rf_feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_feature_selector.fit(X_train, Y_train)
        feature_importances = rf_feature_selector.feature_importances_
        important_features = [stock_vars[i] for i in np.argsort(feature_importances)[-20:]]  # Select top 20 features
        print(f"Selected top 20 features based on Random Forest importance: {important_features}")

        X_train = train[important_features].values
        X_val = validate[important_features].values
        X_test = test[important_features].values
        print("Feature selection completed.")

        # PCA for dimensionality reduction
        print("Performing PCA for dimensionality reduction...")
        pca = PCA(n_components=10)  # Reduce to 10 principal components
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        print(f"PCA completed. Reduced to {pca.n_components_} principal components.")

        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean

        reg_pred = test[["year", "month", "date", "permno", ret_var]]

        # Random Forest with Cross-Validation
        print("Training Random Forest model with GridSearchCV...")
        rf_params = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
        rf_grid.fit(X_train, Y_train_dm)
        best_rf = rf_grid.best_estimator_
        reg_pred["rf"] = best_rf.predict(X_test) + Y_mean
        rf_r2 = 1 - np.sum(np.square((Y_test - reg_pred["rf"]))) / np.sum(np.square(Y_test))
        print(f"Random Forest R²: {rf_r2:.4f}")

        # Gradient Boosting with Bayesian Optimization
        print("Training Gradient Boosting model with Bayesian Optimization...")
        gb_params = {
            'n_estimators': (50, 200),
            'learning_rate': (0.01, 0.2, 'log-uniform'),
            'max_depth': (3, 10)
        }
        gb_bayes = BayesSearchCV(GradientBoostingRegressor(random_state=42), gb_params, scoring='neg_mean_squared_error', n_iter=20, cv=kf, n_jobs=-1, random_state=42)
        gb_bayes.fit(X_train, Y_train_dm)
        best_gb = gb_bayes.best_estimator_
        reg_pred["gb"] = best_gb.predict(X_test) + Y_mean
        gb_r2 = 1 - np.sum(np.square((Y_test - reg_pred["gb"]))) / np.sum(np.square(Y_test))
        print(f"Gradient Boosting R²: {gb_r2:.4f}")

        pred_out = pred_out._append(reg_pred, ignore_index=True)
        counter += 1

    out_path = os.path.join(work_dir, "output-v4.csv")
    pred_out.to_csv(out_path, index=False)

    print("Modeling and prediction completed.")
    print(datetime.datetime.now())
