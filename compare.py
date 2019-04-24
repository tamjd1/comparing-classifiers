import csv
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from timer import Timer

X_train = None
X_test = None
y_train = None
y_test = None


def prepare():
    with open("./data/data_banknote_authentication.txt") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        global X_train, X_test, y_train, y_test
        X = np.array([[np.float(item) for item in row[:-1]] for row in rows])
        y = [row[-1] for row in rows]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=750)


def _determine_optimal_gamma_value(k=3):
    """
    SVM 3-fold Cross Validation to determine optimal gamma value
    Note: gamma value in this function is the inverse of the
    standard deviation of the Radial Basis Function kernel
    used as a similarity measure between two points
    :return: gamma value
    """
    # possible values of gamma to test with
    test_gamma_values = [0.01, 1/X_train.shape[1], 1.0]

    # dictionary of gamma value index and scores for determining
    # optimal gamma value based on cross validation tests
    gamma_value_scores = {0: [], 1: [], 2: []}

    X = X_train
    y = y_train

    # 3 folds, repeated 3 times; once for each value of gamma
    rkf = RepeatedKFold(n_splits=k, n_repeats=len(test_gamma_values))
    for i, (train_indices, test_indices) in enumerate(rkf.split(X, y)):
        _X_train = np.array([X[_i] for _i in train_indices])
        _y_train = [y[_i] for _i in train_indices]
        _X_test = np.array([X[_i] for _i in test_indices])
        _y_test = [y[_i] for _i in test_indices]

        gamma_value_index = i // len(test_gamma_values)
        gamma = test_gamma_values[gamma_value_index]
        svc = SVC(kernel="rbf", gamma=gamma)
        svc.fit(X_train, y_train)
        score = svc.score(X_train, y_train)
        gamma_value_scores[gamma_value_index].append(score)

    max_score = 0
    best_gamma_value = 0

    # calculate average score for each gamma value
    # and set the highest average as the best gamma value
    for gamma_value_index, scores in gamma_value_scores.items():
        score = sum(scores)/len(scores)
        print("Gamma value: {} | Score: {:.4f}".format(
            test_gamma_values[gamma_value_index], score
        ))
        if score > max_score:
            max_score = score
            best_gamma_value = test_gamma_values[gamma_value_index]

    print("Best gamma value as determined by {}-fold cross validation: {}".format(k, best_gamma_value))

    return best_gamma_value


def classify():
    print("\n--- Logistic Regression Classifier ---")
    with Timer() as t:
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, y_train)
    print("Total training time for Logistic Regression classifier: {}".format(t.milliseconds))

    training_score = logistic_regression.score(X_train, y_train)
    print("Training accuracy score for Logistic Regression classifier: {}".format(training_score))

    y_pred = logistic_regression.predict(X_test)
    testing_score = accuracy_score(y_test, y_pred)
    print("Prediction accuracy score for Logistic Regression classifier: {}".format(testing_score))

    _confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix for Logistic Regression classifier: \n{}".format(_confusion_matrix))

    # ------------------------------------------------------------------------------------------------------------------

    print("\n--- Support Vector Machine Classifier ---")
    gamma = _determine_optimal_gamma_value()
    with Timer() as t:
        # use Radial Basis Function kernel and gamma value
        # determined by 3-fold cross validation checks
        svc = SVC(kernel="rbf", gamma=gamma)
        svc.fit(X_train, y_train)
    print("Total training time for Support Vector Machine classifier: {}".format(t.milliseconds))

    training_score = svc.score(X_train, y_train)
    print("Training accuracy score for Support Vector Machine classifier: {}".format(training_score))

    y_pred = svc.predict(X_test)
    testing_score = accuracy_score(y_test, y_pred)
    print("Prediction accuracy score for Support Vector Machine classifier: {}".format(testing_score))

    _confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix for Support Vector Machine classifier: \n{}".format(_confusion_matrix))

    # ------------------------------------------------------------------------------------------------------------------

    print("\n--- Adaboost Classifier with Naive Bayes base estimator ---")
    with Timer() as t:
        nb = GaussianNB()
        adaboost = AdaBoostClassifier(base_estimator=nb)
        adaboost.fit(X_train, y_train)
    print("Total training time for Adaboost classifier: {}".format(t.milliseconds))

    training_score = adaboost.score(X_train, y_train)
    print("Training accuracy score for Adaboost classifier: {}".format(training_score))

    y_pred = adaboost.predict(X_test)
    testing_score = accuracy_score(y_test, y_pred)
    print("Prediction accuracy score for Adaboost classifier: {}".format(testing_score))

    _confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix for Adaboost classifier: \n{}".format(_confusion_matrix))

    # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    prepare()
    classify()
