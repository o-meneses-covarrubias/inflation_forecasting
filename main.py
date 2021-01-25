# This is a Python script to build and test a very simple model to forecast inflation.

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


if __name__ == '__main__':

    # Preprocessing
    ########################

    # Load data (data is monthly provided by the Federal Reserve Bank of St. Louis). Please see
    # https: // research.stlouisfed.org / econ / mccracken / fred - databases /
    data = pd.read_csv("2020-12.csv")
    data = data.drop(index=0).drop(index=max(data.index))
    data["sasdate"] = pd.to_datetime(data["sasdate"])
    data = data.set_index("sasdate")

    # Test which columns have missing data and print proportion
    cols_with_miss = data.columns[data.isnull().any()]
    cols_del = list()
    for col in cols_with_miss:
        prop = round(data[col].isnull().sum() / data.shape[0], 4) * 100
        print(f'Column {col} has {prop}% missing values')
        if prop > 5:
            cols_del.append(col)

    # Delete columns with more than 5% missing values, for all other take mean or majority vote
    data = data.drop(columns = cols_del)
    cols_replace = [x for x in cols_with_miss if x not in cols_del]
    for col in cols_replace:
        if is_numeric_dtype(data[col]):
            mean = np.mean(data[col])
            data[col] = data[col].fillna(mean)
        else:
            raise TypeError("Only for numeric columns implemented")

    # Transform cateogorical values using one-hot encoding (only numeric values currently)
    # data = pd.get_dummies(data)

    # Define prediction target and train/test split
    ####################################################

    # Set prediction target to consumer price index (including all items) average increase for the next 3 months
    # Introduce 3 month time lag for target
    y = (data["CPIAUCSL"].shift(periods=-3) / data["CPIAUCSL"] -1) / 3
    # y.plot()
    # Keep y also as feature in the feature table as it is assumed that the inflation of the current month is known

    # Create baseline predictions
    # The baseline predictions are the rolling 3-month moving averages
    data["inflation_month_to_month"] = data["CPIAUCSL"] / data["CPIAUCSL"].shift(periods=1) - 1
    data["baseline"] = data["inflation_month_to_month"].rolling(window=3).mean()

    # Define training and test set
    # Train data will be from 1990 until 2010
    features_train = data.loc["1990-01-01":"2019-12-01"]
    y_train = y.loc["1990-01-01":"2019-12-01"]
    # Test data is 2020 (3 months next inflation is only available until August)
    features_test = data.loc["2020-01-01":"2020-08-01"]
    y_test = y.loc["2020-01-01":"2020-08-01"]

    # Train random forest
    ########################
    from sklearn.ensemble import RandomForestRegressor

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf.fit(features_train, y_train)
    pred_train = rf.predict(features_train)
    pred_test = rf.predict(features_test)

    # Evaluate model
    #############################
    print(f'The RF on training set has MAE of {round(mean_absolute_error(pred_train, y_train), 4)}')
    print(f'The RF on test set has MAE of {round(mean_absolute_error(pred_test, y_test), 4)}')

    # See benchmark result
    print(f'The baseline on train has an MAE of {round(mean_absolute_error(features_train["baseline"], y_train), 4)}')
    print(f'The baseline on test has an MAE of {round(mean_absolute_error(features_test["baseline"], y_test),4)}')





