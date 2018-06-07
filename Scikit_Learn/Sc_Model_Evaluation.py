from sklearn import metrics, svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB
import settings


def scikit_learn_naivebayes_test(vector, label):
    x_train, x_test, y_train, y_test = train_test_split(vector, label, train_size=0.7,
                                                        test_size=0.3)
    nb_model = MultinomialNB(1.0).fit(x_train, y_train)
    pred = nb_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    settings.sc_naive_bayes_acc = 100 * accuracy
    print("Scikit Learn Naive Bayes Model Accuracy = %.2f" % (100 * accuracy))


def scikit_learn_svm_test(vector, label):
    x_train, x_test, y_train, y_test = train_test_split(vector, label, train_size=0.7,
                                                        test_size=0.3)
    svm_model = svm.SVC().fit(x_train, y_train)
    pred = svm_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    settings.sc_svm_acc = 100* accuracy
    print("Scikit Learn Support Vector Machine  Model Accuracy = %.2f" % (100 * accuracy))


def scikit_learn_logistic_tet(vector, label):
    x_train, x_test, y_train, y_test = train_test_split(vector, label, train_size=0.7,
                                                        test_size=0.3)
    log_model = LogisticRegression().fit(x_train, y_train)
    pred = log_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    settings.sc_log_acc = 100*accuracy
    print("Scikit Learn logistic regression  Model Accuracy = %.2f" % (100 * accuracy))


def scikit_learn_tree_test(vector, label):
    x_train, x_test, y_train, y_test = train_test_split(vector, label, train_size=0.7,
                                                        test_size=0.3)
    tree_model = tree.DecisionTreeClassifier().fit(x_train, y_train)
    pred = tree_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    settings.sc_tree_acc = 100*accuracy
    print("Scikit Learn Decision Tree  Model Accuracy = %.2f" % (100 * accuracy))


