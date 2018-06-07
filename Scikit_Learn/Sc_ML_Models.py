import numpy as np
import pandas as pd
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def scikit_labeling_data(datapath_pos, datapath_neg, label_pos, label_neg, sc_tf):
    data_pos = pd.read_csv(datapath_pos, header=None, encoding='utf8', engine='python', index_col=False)
    data_neg = pd.read_csv(datapath_neg, header=None, encoding='utf8', engine='python', index_col=False)

    label_p = np.full(data_pos.shape[0], label_pos)
    label_n = np.full(data_neg.shape[0], label_neg)

    total_data = pd.concat((data_pos, data_neg))
    total_label = np.concatenate((label_p, label_n))

    vectorized = sc_tf.fit_transform(total_data[0], None)
    return vectorized, total_label


def scikit_naive_bayes(vector, label):
    nb_model = MultinomialNB(1.0).fit(vector, label)
    #print("Scikit Learn Naive Bayes Model Trained")
    return nb_model


def scikit_svm_model(vector, label):
    svm_model = svm.SVC().fit(vector, label)
    #print("Scikit Learn SVM Model Trained")
    return svm_model


def scikit_logistic_regression(vector, label):
    log_model = LogisticRegression().fit(vector, label)
    #print("Scikit Learn Logistic Regression  Model Trained")
    return log_model


def scikit_decision_tree(vector, label):
    tree_model = tree.DecisionTreeClassifier().fit(vector, label)
    #print("Scikit Learn Decision Tree Model Trained")
    return tree_model

# unionized_vectors = vstack((vector_pos, vector_neg))
# unionized_labels = np.concatenate((label_pos, label_neg))
