import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def main():
    # read data
    operational_settings = ['Op_Setting_' + str(i) for i in range(1, 4)]
    sensor_columns = ['Sensor_' + str(i) for i in range(1, 27)]
    labels = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns
    train = pd.read_csv('CMAPSSData/train_FD001.txt', sep=' ', header=None, names=labels, index_col=False)
    # test = pd.read_csv('CMAPSSData/test_FD001.txt', sep=' ', header=None, names=labels, index_col=False)
    # rul = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=' ', header=None, names=target_var, index_col=False)

    # for each engine find RUL = time.max() - current time
    mapper = {}
    for engine_no in train['engine_no'].unique():
        mapper[engine_no] = train['time_in_cycles'].loc[train['engine_no'] == engine_no].max()
    train['RUL'] = train['engine_no'].apply(lambda nr: mapper[nr]) - train['time_in_cycles']
    train['failure_near'] = train['RUL'] < 10.0

    # get rid of nan and constant values
    cols_nan = train.columns[train.isna().any()].tolist()
    cols_const = [col for col in train.columns if len(train[col].unique()) <= 2 and col != 'failure_near']
    train = train.drop(columns=cols_nan + cols_const)

    # TODO: normalize the dataset, preprocessing.minmaxscaler()

    # split the training data set
    features_train, features_test, labels_train, labels_test = train_test_split(train, train['RUL'])

    rf = RandomForestRegressor(n_estimators=200, max_depth=15).fit(features_train, labels_train)
    predictions = rf.predict(features_test)
    print(f"RandomForestRegressor: Accuracy: {sum(predictions == labels_test) / len(features_test)}")

    dt = DecisionTreeRegressor().fit(features_train, labels_train)
    predictions = dt.predict(features_test)
    print(f"DecisionTreeRegressor: Accuracy: {sum(predictions == labels_test) / len(features_test)}")

    lr = LinearRegression().fit(features_train, labels_train)
    predictions = lr.predict(features_test)
    print(f"LinearRegression: Accuracy: {sum(predictions == labels_test) / len(features_test)}")

if __name__ == "__main__":
    main()
