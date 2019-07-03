import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import data_utils


def main():
    """ Problem formulated as a binary classifier """
    train = data_utils.get_training_data(is_classifier=True)
    x_train, x_test, y_train, y_test = train_test_split(train.drop(['failure_near'], axis=1), train['failure_near'])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    rfc = RandomForestClassifier(n_estimators=100)
    cv = KFold(5)
    pipeline = Pipeline(steps=[('standardize', preprocessing.StandardScaler()), ('model', rfc)])

    min_samples_leaf = [2]
    max_depth = [8]

    optimized_model = GridSearchCV(estimator=pipeline, cv=cv, param_grid=dict(model__min_samples_leaf=min_samples_leaf,
                                                                              model__max_depth=max_depth),
                                   scoring='roc_auc', verbose=1, n_jobs=-1)
    optimized_model.fit(x_train, y_train)
    y_prob = optimized_model.predict_proba(x_test)[:, 1]
    y_pred = optimized_model.predict(x_test)
    print(optimized_model.best_estimator_)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Random Forest Accuracy: " + "{:.1%}".format(accuracy_score(y_test, y_pred)))
    print("Random Forest Precision: " + "{:.1%}".format(precision_score(y_test, y_pred)))
    print("Random Forest Recall: " + "{:.1%}".format(recall_score(y_test, y_pred)))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    fpr, tpr, threshold = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
