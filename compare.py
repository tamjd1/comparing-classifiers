import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None


def prepare():
    with open("./data/data_banknote_authentication.txt") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        global X, y, X_train, X_test, y_train, y_test
        X = np.array([[np.float(item) for item in row[:-1]] for row in rows])
        y = [row[-1] for row in rows]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=750)


def train():
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    score = logistic_regression.score(X_train, y_train)
    print(score)

    svc = SVC()
    svc.fit(X_train, y_train)
    score = svc.score(X_train, y_train)
    print(score)

    nb = GaussianNB()
    adaboost = AdaBoostClassifier(base_estimator=nb)
    # adaboost = AdaBoostClassifier(base_estimator=logistic_regression)
    # adaboost = AdaBoostClassifier()
    adaboost.fit(X_train, y_train)
    score = adaboost.score(X_train, y_train)
    print(score)


def test():
    return


if __name__ == '__main__':
    prepare()
    train()
    test()
