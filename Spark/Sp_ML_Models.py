from typing import List

from pyspark.mllib.classification import SVMWithSGD, NaiveBayes, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree


def spark_labeling_data(datapath, label, sparkcontext, tf):
    """
    reads the data and then turns it into vector space with "tf.tranfsorm()" function, after
    it takes the vector space and labels it with given label and stores Labeledpoints in Array.
    Given array then transformed to RDD with "sparkcontext.parallelize(data)".

    :param datapath: Path of data to be processed.
    :param label: A float number for supervised learning
    :param sparkcontext: Sparks own initiator for analysis.
    :param tf: Term Frequency is used for converting strings to mathematical expressions.
    :return: returns the labeled RDD to use on creating ML model.

    """
    datardd = sparkcontext.textFile(datapath).map(lambda line: line.split("0a0d")).toLocalIterator()
    data: List[LabeledPoint] = []
    for i in datardd:
        data.append(LabeledPoint(label, tf.transform(list(i))))
    # print(data)
    labeled_rdd = sparkcontext.parallelize(data)
    return labeled_rdd


def spark_naive_bayes_model(labeled_raw, lambda_value):
    """
    Creates Naive Bayes ML model with given parameters, to analyize data.
    :param labeled_raw: at least two categorical labeled data required to train the model.
    :param lambda_value: Lambda Value to prevent zero case regulator
    :return: Return ans Naive Bayes ML Model
    """
    model = NaiveBayes.train(labeled_raw, lambda_value)
    #print("Spark Naive Bayes Model Trained")
    return model


def spark_svm_model(labeled_data, iteration):
    """
    :param labeled_data: at least two categorical labeled data required to train the model.
    :param iteration: How many times algorithm runs
    :return: SVMModel
    """
    model = SVMWithSGD.train(labeled_data, iteration)
    # model.setThreshold(0.5)
    #print("Spark Support Vector Machine Model Trained")
    return model


def spark_logistic_regression_model(labeled_data, iteration):
    """
    :param iteration: How many times algorithm runs
    :param labeled_data: at least two categorical labeled data required to train the model.
    :return: Logistic Regression Model
    """
    model = LogisticRegressionWithLBFGS.train(labeled_data, iterations=iteration)
    # model.setThreshold(0.5)
    #print("Spark Logistic Regression Model Trained")
    return model


def spark_desicion_tree_model(labeled_data, numclass, categoricalfeaturesinfo, purity, maxdep, maxbin):
    """
    :param labeled_data: at least two categorical labeled data required to train the model.
    :param numclass: Class number for categorization
    :param categoricalfeaturesinfo: Leave it blank
    :param purity: tree purity mostly 'gini'
    :param maxdep: maximum deph of tree
    :param maxbin:  ....
    :return:  returns desicion tree model.
    """
    model = DecisionTree.trainClassifier(labeled_data, numClasses=numclass,
                                         categoricalFeaturesInfo=categoricalfeaturesinfo,
                                         impurity=purity, maxDepth=maxdep, maxBins=maxbin)
    #print("Spark Decision Tree Model Trained")
    return model


def spark_kmeans_model():
    pass
