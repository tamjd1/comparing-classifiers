import csv
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
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
    :return: training and testing data (dict)
        X_train : training features (numpy.ndarray)
        X_test : testing features (numpy.ndarray)
        y_train : training class labels (list)
        y_test : testing class labels (list)
        y_test_matrix: testing class labels as a matrix of row vectors (numpy.ndarray)
    """
    with open(data_file) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        X = np.array([[np.float(item) for item in row[:-1]] for row in rows])
        y = [row[-1] for row in rows]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size)

    y_test_matrix = np.array([[int(_y == '0'), int(_y == '1')] for _y in y_test])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_matrix": y_test_matrix
    }


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


def _compute_and_draw_roc_curve(y_test, y_scores):
    """
    Based on class labels and prediction scores,
        compute and generate ROC curve
    :param y_test: matrix of class labels (numpy.ndarray)
    :param y_scores:  matrix of prediction probability scores (numpy.ndarray)
    """
    false_positive_rate = {}
    true_positive_rate = {}

    for i in range(y_test.shape[1]):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test[:, i], y_scores[:, i])

    # todo : draw curve


def _classify(data, classifier):
    """
    This function does training and testing of classifier specified on the given data
        and outputs training and prediction times, training and prediction accuracy,
        confusion matrix, and ROC curve
    :param data: training and testing data (dict)
    :param classifier: model to train and test
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    with Timer() as t:
        classifier.fit(X_train, y_train)
    print("Total training time: {}".format(t.milliseconds))

    training_score = classifier.score(X_train, y_train)
    print("Training accuracy score: {}".format(training_score))

    with Timer() as t:
        y_pred = classifier.predict(X_test)
    print("Total prediction time: {}".format(t.milliseconds))

    testing_score = accuracy_score(y_test, y_pred)
    print("Prediction accuracy score: {}".format(testing_score))

    _confusion_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n{}".format(_confusion_matrix))

    # ROC curve
    y_test_matrix = data["y_test_matrix"]
    y_scores_matrix = classifier.predict_proba(X_test)
    _compute_and_draw_roc_curve(y_test_matrix, y_scores_matrix)


def main():
    """
    Entry point for comparing classifiers:
    1. Logistic Regression
    2. Support Vector Machine
    3. Adaboost with Naive Bayes base estimator
    """
    data = _prepare_data()

    # ------------------------------------------------------------------------------------------------------------------

    print("\n--- Logistic Regression Classifier ---")
    logistic_regression = LogisticRegression()
    _classify(data, logistic_regression)

    # ------------------------------------------------------------------------------------------------------------------

    print("\n--- Support Vector Machine Classifier ---")
    # Note: gamma value in sklearn.svm is the inverse of the
    #     standard deviation of the Radial Basis Function kernel
    #     used as a similarity measure between two points
    gamma = _determine_optimal_gamma_value(data["X_train"], data["y_train"])
    svc = SVC(
        kernel="rbf",  # use Radial Basis Function kernel and gamma value
        gamma=gamma,  # determined by 3-fold cross validation checks
        probability=True  # setting this to True may have performance implications
    )
    _classify(data, svc)

    # ------------------------------------------------------------------------------------------------------------------

    print("\n--- Adaboost (w/ Naive Bayes base estimator) Classifier ---")
    # default base estimator is DecisionTreeClassifier, however,
    # since our data is entirely numeric, Naive Bayes,
    # or another classifier suitable for numeric data is a better fit
    nb = GaussianNB()
    adaboost = AdaBoostClassifier(base_estimator=nb)
    _classify(data, adaboost)

    # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
