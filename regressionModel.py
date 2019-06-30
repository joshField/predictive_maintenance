import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import data_utils

#TODO abstract training/grid search for multiple models, compare model performance, create classifier

def main():
    train = data_utils.get_training_data()
    features_train, features_test, labels_train, labels_test = train_test_split(train.drop(['RUL'],axis=1), train['RUL'])

    # setup 5 fold cross validation with standardization
    rf = RandomForestRegressor(n_estimators=100, verbose=1)
    cv = model_selection.KFold(5)
    pipeline = Pipeline(steps=[('standardize', preprocessing.StandardScaler()), ('model', rf)])

    # tune the model
    my_min_samples_leaf = [2]
    my_max_depth = [7]

    # run the model using gridsearch, select the model with best search
    optimized_rf = GridSearchCV(estimator=pipeline
                                , cv=cv
                                , param_grid=dict(model__min_samples_leaf=my_min_samples_leaf,
                                                  model__max_depth=my_max_depth)
                                , scoring='neg_mean_squared_error'
                                , verbose=10
                                , n_jobs=-1)
    optimized_rf.fit(features_train, labels_train)

    # show the best model estimators
    print(optimized_rf.best_estimator_)

    # evaluate metrics on holdout
    y_pred = optimized_rf.predict(features_test)
    print("Random Forest Mean Squared Error: ", mean_squared_error(labels_test, y_pred))
    print("Random Forest Mean Absolute Error: ", mean_absolute_error(labels_test, y_pred))
    print("Random Forest r-squared: ", r2_score(labels_test, y_pred))

    #view feature importances
    importances = optimized_rf.best_estimator_.named_steps['model'].feature_importances_
    feature_names = features_test.columns
    indices = np.argsort(importances)[::-1]
    f, ax = plt.subplots(figsize=(11,9))
    plt.bar(range(features_test.shape[1]), importances[indices], align="center")
    plt.xticks(range(features_test.shape[1]), feature_names[indices], rotation="vertical")
    plt.xlim([-1,features_test.shape[1]])
    plt.title("RandomForestRegressor Importances")
    plt.show()
    important_features = pd.Series(data = importances,index=feature_names)
    important_features.sort_values(ascending=False, inplace=True)

    # dt = DecisionTreeRegressor().fit(features_train, labels_train)
    # predictions = dt.predict(features_test)
    #
    # #view feature importances
    # importances = dt.feature_importances_
    # feature_names = features_test.columns
    # indices = np.argsort(importances)[::-1]
    #
    # f, ax = plt.subplots(figsize=(11,9))
    # plt.bar(range(features_test.shape[1]), importances[indices], align="center")
    # plt.xticks(range(features_test.shape[1]), feature_names[indices], rotation="vertical")
    # plt.xlim([-1,features_test.shape[1]])
    # plt.title("DecisionTreeRegressor Importances")
    # plt.show()
    #
    # important_features = pd.Series(data = importances,index=feature_names)
    # important_features.sort_values(ascending=False,inplace=True)

    # lr = LinearRegression().fit(features_train, labels_train)
    # predictions = lr.predict(features_test)
    # print(f"LinearRegression: Accuracy: {sum(predictions == labels_test) / len(features_test)}")

if __name__ == "__main__":
    main()
