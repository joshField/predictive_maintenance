import pandas as pd


def get_training_data():
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
    return train

#TODO plot all sensor data, plot performance of model, compare model performance

if __name__ == "__main__":
    data = get_training_data()
    print(data.head(5))