import csv
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from timer import Timer


def _prepare_data(data_file="./data/data_banknote_authentication.txt", training_size=750):
    """
    Read in data from input file and parse it into features and class labels arrays
    Use sklearn's train_test_split function to split the data into training and testing sets

    :param data_file: path to data file (str, default: "./data/data_banknote_authentication.txt"
    :param training_size: count or percentage of training data points (int, float, default: 750)
    :return:
        X_train : training features (numpy.ndarray)
        X_test : testing features (numpy.ndarray)
        y_train : training class labels (list)
        y_test : testing class labels (list)
    """
    with open(data_file) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        X = np.array([[np.float(item) for item in row[:-1]] for row in rows])
        y = [row[-1] for row in rows]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size)

    return X_train, X_test, y_train, y_test


def _determine_optimal_gamma_value(X, y, k=3):
    """
    SVM 3-fold Cross Validation to determine optimal gamma value

    Note: gamma value in sklearn.svm is the inverse of the
        standard deviation of the Radial Basis Function kernel
        used as a similarity measure between two points
    :param X: features (numpy.ndarray)
    :param y: labels (list)
    :param k: number of folds (int, default: 3)
    :return:
    """
    # possible values of gamma to test with
    test_gamma_values = [0.01, 1/X.shape[1], 1.0]

    # dictionary of gamma value index and scores for determining
    # optimal gamma value based on cross validation tests
    gamma_value_scores = {0: [], 1: [], 2: []}

    # 3 folds, repeated 3 times; once for each value of gamma
    rkf = RepeatedKFold(n_splits=k, n_repeats=len(test_gamma_values))
    for i, (train_indices, test_indices) in enumerate(rkf.split(X, y)):
        X_train = np.array([X[_i] for _i in train_indices])
        y_train = [y[_i] for _i in train_indices]
        X_test = np.array([X[_i] for _i in test_indices])
        y_test = [y[_i] for _i in test_indices]

        gamma_value_index = i // len(test_gamma_values)
        gamma = test_gamma_values[gamma_value_index]
        svc = SVC(kernel="rbf", gamma=gamma)
        svc.fit(X_train, y_train)  # train
        y_pred = svc.predict(X_test)  # test
        score = accuracy_score(y_test, y_pred)  # score
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


def main():
    """
    Entry point for comparing classifiers

    Compares training time, prediction accuracy, confusion matrices
        of the following classifiers:
    1. Logistic Regression
    2. Support Vector Machine
    3. Adaboost with Naive Bayes base estimator
    """
    X_train, X_test, y_train, y_test = _prepare_data()

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
    gamma = _determine_optimal_gamma_value(X_train, y_train)
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
    main()
