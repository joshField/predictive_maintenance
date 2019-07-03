import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_training_data():
    # read data
    operational_settings = ['Op_Setting_' + str(i) for i in range(1, 4)]
    sensor_columns = ['Sensor_' + str(i) for i in range(1, 23)]
    labels = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns
    train = pd.read_csv('CMAPSSData/train_FD001.txt', sep=' ', header=None, names=labels, index_col=False)
    # test = pd.read_csv('CMAPSSData/test_FD001.txt', sep=' ', header=None, names=labels, index_col=False)
    # rul = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=' ', header=None, names=target_var, index_col=False)

    # for each engine find RUL = time.max() - current time
    mapper = {}
    for engine_no in train['engine_no'].unique():
        mapper[engine_no] = train['time_in_cycles'].loc[train['engine_no'] == engine_no].max()
    train['RUL'] = train['engine_no'].apply(lambda nr: mapper[nr]) - train['time_in_cycles']

    # add this for classifier
    # train['failure_near'] = train['RUL'] < 10.0

    # explore data features
    plot_features(train, sensor_columns, operational_settings)
    plot_std(train, sensor_columns, operational_settings)

    # get rid of nan and constant values
    cols_nan = train.columns[train.isna().any()].tolist()
    cols_const = [col for col in train.columns if len(train[col].unique()) <= 2 and col != 'failure_near']
    train = train.drop(columns=cols_nan + cols_const)

    return train


def plot_features(data, sensor_cols, setting_cols):
    """ Plots raw feature data for the first 15 engines """
    sns.set()
    explore = sns.PairGrid(data=data.query('engine_no < 15'), x_vars=['RUL'], y_vars=sensor_cols + setting_cols, hue="engine_no")
    explore = explore.map(plt.scatter)
    explore = explore.set(xlim=(400,0))
    explore = explore.add_legend()
    plt.show()


def plot_std(data, sensor_cols, setting_cols):
    """ Visualize the standard deviation of each feature """
    data[sensor_cols+setting_cols].std().plot(kind='bar', title="Feature STD")
    plt.show()


if __name__ == "__main__":
    data = get_training_data()
    print(data.head(5))
