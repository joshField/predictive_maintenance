import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import data_utils

# TODO reframe as a classifier


def rfr():
    """ Defines Random Forest Regression model and params """
    model = RandomForestRegressor(
        max_depth=10,
        n_estimators=100,
        verbose=0
    )
    params = {
        'model__n_estimators': [10, 50],
        'model__max_depth': [6],
        'model__min_samples_leaf': [2]
    }
    return model, params


def dtr():
    """ Defines Decision Tree Regression model and params """
    model = DecisionTreeRegressor(
        max_depth=10
    )
    params = {
        'model__max_depth': [6],
        'model__min_samples_leaf': [2]
    }
    return model, params


def lr():
    """ Defines Linear Regression model and params """
    model = LinearRegression()
    params = {}
    return model, params


def abr():
    """ Defines Decision Tree Regression model and params """
    model = AdaBoostRegressor()
    params = {
        'model__n_estimators': [10, 50]
    }
    return model, params


def en():
    """ Defines ElasticNet model and params """
    model = ElasticNet()
    params = {
        'model__alpha': np.linspace(.01, 1, num=5),
        'model__l1_ratio': np.linspace(.01, 1, num=3)
    }
    return model, params


def svm():
    """ Defines a State Vector Regression model and params """
    model = SVR()
    params = {
        'model__C': [1],
        'model__epsilon': [.05, .1, .15]
    }
    return model, params


def main():

    # get training data and split into training and test sets
    train = data_utils.get_training_data()
    features_train, features_test, labels_train, labels_test = train_test_split(train.drop(['RUL'], axis=1), train['RUL'])
    feature_names = features_test.columns
    features_train = np.array(features_train)
    features_test = np.array(features_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    # create models and parameters
    regression_models = [
        rfr(),
        dtr(),
        lr(),
        abr(),
        en(),
        svm()
    ]

    # these models don't have feature_importances_ member in scikit
    no_imp = ['SVM', 'LinearRegression', 'ElasticNet']
    results = {}

    for model, params in regression_models:

        # setup pipeline and 5 fold cross validation with standardization
        pipeline = Pipeline(steps=[('standardize', preprocessing.StandardScaler()), ('model', model)])
        cv = model_selection.KFold(5)
        model_name = type(model).__name__
        print("Testing %s -----------------------------\n\tparams:%s" % (model_name, params))

        # run the model using gridsearch, select the best model
        optimized_model = GridSearchCV(estimator=pipeline,
                                       cv=cv,
                                       param_grid=params,
                                       scoring='neg_mean_squared_error',
                                       verbose=1,
                                       n_jobs=-1)
        optimized_model.fit(features_train, labels_train)

        # show the best estimator parameters and performance
        print("Best estimator: \n%s" % optimized_model.best_estimator_)
        y_pred = optimized_model.predict(features_test)
        score =  mean_absolute_error(labels_test, y_pred)
        print("Mean Squared Error: ", mean_squared_error(labels_test, y_pred))
        print("Mean Absolute Error: ", score)
        print("R-Squared: ", r2_score(labels_test, y_pred))
        results[model_name] = score
        print("-----------------------------")

        # view feature importances
        if model_name not in no_imp:
            print(model_name)
            importances = optimized_model.best_estimator_.named_steps['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            f, ax = plt.subplots(figsize=(11,9))
            plt.bar(range(features_test.shape[1]), importances[indices], align="center")
            plt.xticks(range(features_test.shape[1]), feature_names[indices], rotation="vertical")
            plt.xlim([-1,features_test.shape[1]])
            plt.title("%s Importances" % model_name)
            plt.show()

        # view model performance by plotting actual vs predicted Remaining Useful Life for the best model
        fig, ax = plt.subplots()
        ax.scatter(labels_test, y_pred, edgecolors=(0, 0, 0))
        ax.plot([labels_test.min(), labels_test.max()], [labels_test.min(), labels_test.max()], 'k--', lw=4)
        ax.set_xlabel('Actual RUL')
        ax.set_ylabel('Predicted RUL')
        ax.set_title('(%s) Remaining Useful Life Actual vs. Predicted' % model_name)
        plt.show()

    # comopare all models by viewing scores side by side
    print(results)
    fig, ax = plt.subplots()
    names = results.keys()
    scores = results.values()
    plt.bar(range(len(results)), scores, align="center")
    plt.xticks(range(len(results)), names, rotation="vertical")
    plt.xlim([-1, len(results)])
    plt.title("Model Mean Absolute Error Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
