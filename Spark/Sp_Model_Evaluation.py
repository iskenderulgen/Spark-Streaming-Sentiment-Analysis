from pyspark.mllib.classification import NaiveBayes, SVMWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.tree import DecisionTree
import settings


def spark_naive_bayes_precision(labeled_data, lambda_value):
    training, test = labeled_data.randomSplit([0.7, 0.3])
    nb_model = NaiveBayes.train(training, lambda_value)
    prediction_and_label = test.map(lambda p: (nb_model.predict(p.features), p.label))
    accuracy = 100 * prediction_and_label.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
    settings.sp_naive_bayes_acc = accuracy
    print("Apache Spark Naive Bayes Model Accuracy =  %.2f" % accuracy)


def spark_svm_precision(labeled_data, iteration, identifier):
    training, test = labeled_data.randomSplit([0.7, 0.3])
    model = SVMWithSGD.train(training, iteration)
    prediction_and_label = test.map(lambda p: (p.label, model.predict(p.features)))
    accuracy = 100 * prediction_and_label.filter(lambda lp: lp[0] == lp[1]).count() / float(test.count())
    settings.sp_svm_acc = accuracy
    print("Spark Support Vector Machine " + identifier + " Model Accuracy = %.2f" % accuracy)


def spark_logistic_regression_precision(labeled_data, iteration, identifier):
    training, test = labeled_data.randomSplit([0.7, 0.3], seed=11)
    model = LogisticRegressionWithLBFGS.train(training, iterations=iteration)
    prediction_and_label = test.map(lambda p: (p.label, model.predict(p.features)))
    accuracy = 100 * prediction_and_label.filter(lambda lp: lp[0] == lp[1]).count() / float(test.count())
    settings.sp_log_acc = accuracy
    print("Logistic Regression " + identifier + " Model Accuracy = %.2f" % accuracy)


def spark_desicion_tree_precision(labeled_data, numclass,
                                  categoricalfeaturesinfo, purity, max_dep, max_bin, identifier):
    training, test = labeled_data.randomSplit([0.7, 0.3])
    decision_model = DecisionTree.trainClassifier(training, numClasses=numclass,
                                                  categoricalFeaturesInfo=categoricalfeaturesinfo,
                                                  impurity=purity, maxDepth=max_dep, maxBins=max_bin)

    predictions = decision_model.predict(test.map(lambda x: x.features))
    prediction_and_label = test.map(lambda lp: lp.label).zip(predictions)
    accuracy = 100 * prediction_and_label.filter(
        lambda lp: lp[0] == lp[1]).count() / float(test.count())
    # print('Learned classification tree model:')
    # print(decision_model.toDebugString())
    settings.sp_tree_acc = accuracy
    print("Decision Tree " + identifier + " Model Accuracy = %.2f" % accuracy)
