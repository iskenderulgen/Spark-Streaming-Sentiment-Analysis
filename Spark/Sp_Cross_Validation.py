from pyspark.mllib.classification import NaiveBayes, SVMWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.tree import DecisionTree
from Spark import Sp_ML_Models
import settings

total_fold_sum = 0


def cross_validation(positive_path_data, negative_path_data, lambda_value, tf, iteration, numclass,
                     categoricalfeaturesinfo, purity, max_dep, max_bin, sc):
    global total_fold_sum
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    nb_pos_label = Sp_ML_Models.spark_labeling_data(positive_path_data, 1.0, sc, tf)
    nb_neg_label = Sp_ML_Models.spark_labeling_data(negative_path_data, 0.0, sc, tf)
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = splitting(nb_pos_label, sc=sc)
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = splitting(nb_neg_label, sc=sc)
    list_pos = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
    list_neg = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]



    # Naive Bayes Cross Validation Section Starts
    fn = numbers.copy()
    for i in range(10):
        spark_naive_bayes_precision(list_pos, list_neg, lambda_value, fn[0], fn[1], fn[2], fn[3], fn[4], fn[5], fn[6],
                                    fn[7], fn[8], fn[9])
        if i == 9:
            break
        fn[0], fn[i + 1] = fn[i + 1], fn[0]
    settings.sp_naive_cross_acc = total_fold_sum / 10
    print("Naive Bayes Cross Validation Score = " + str(total_fold_sum / 10))
    total_fold_sum = 0
    # Naive Bayes Cross Validation Section Ends

    # SVM Cross Validation Section Starts
    fn = numbers.copy()
    for i in range(10):
        spark_svm_precision(list_pos, list_neg, iteration, fn[0], fn[1], fn[2], fn[3], fn[4], fn[5], fn[6],
                            fn[7], fn[8], fn[9])
        if i == 9:
            break
        fn[i], fn[i + 1] = fn[i + 1], fn[i]
    settings.sp_svm_cross_acc = total_fold_sum / 10
    print("Support Vector Machine Cross Validation Score = " + str(total_fold_sum / 10))
    total_fold_sum = 0
    # SVM Cross Validation Sections Ends

    # Logistic Regression Cross Validation Section Starts
    fn = numbers.copy()
    for i in range(10):
        spark_logistic_regression_precision(list_pos, list_neg, iteration, fn[0], fn[1], fn[2], fn[3], fn[4], fn[5],
                                            fn[6], fn[7], fn[8], fn[9])
        if i == 9:
            break
        fn[i], fn[i + 1] = fn[i + 1], fn[i]
    settings.sp_log_cross_acc = total_fold_sum / 10
    print("Logistic Regression Cross Validation Score = " + str(total_fold_sum / 10))
    total_fold_sum = 0
    # Logistic Regression Cross Validation Section Starts

    # Decision Tree Cross Validation Section Starts
    fn = numbers.copy()
    for i in range(10):
        spark_desicion_tree_precision(list_pos, list_neg, numclass, categoricalfeaturesinfo, purity, max_dep,
                                      max_bin, fn[0], fn[1], fn[2], fn[3], fn[4], fn[5],
                                      fn[6], fn[7], fn[8], fn[9])
        if i == 9:
            break
        fn[i], fn[i + 1] = fn[i + 1], fn[i]
    settings.sp_tree_cross_acc = total_fold_sum / 10
    print("Desicion Tree Cross Validation Score = " + str(total_fold_sum / 10))
    total_fold_sum = 0
    # Decision Tree Cross Validation Section Starts


def splitting(data, sc):
    count = data.count()
    i = int(count / 10)

    data1 = sc.parallelize(data.take(i))
    data2 = sc.parallelize(data.take(2 * i)).subtract(sc.parallelize(data.take(i)))
    data3 = sc.parallelize(data.take(3 * i)).subtract(sc.parallelize(data.take(2 * i)))
    data4 = sc.parallelize(data.take(4 * i)).subtract(sc.parallelize(data.take(3 * i)))
    data5 = sc.parallelize(data.take(5 * i)).subtract(sc.parallelize(data.take(4 * i)))
    data6 = sc.parallelize(data.take(6 * i)).subtract(sc.parallelize(data.take(5 * i)))
    data7 = sc.parallelize(data.take(7 * i)).subtract(sc.parallelize(data.take(6 * i)))
    data8 = sc.parallelize(data.take(8 * i)).subtract(sc.parallelize(data.take(7 * i)))
    data9 = sc.parallelize(data.take(9 * i)).subtract(sc.parallelize(data.take(8 * i)))
    data10 = data.subtract(sc.parallelize(data.take(i)))
    return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10


def spark_naive_bayes_precision(list_pos, list_neg, lambda_value, a, b, c, d, e, f, g, h, j, k):
    global total_fold_sum
    test = list_pos[a].union(list_neg[a])

    train = list_pos[b].union(list_pos[c]).union(list_pos[d]).union(list_pos[e]).union(list_pos[f]).union(list_pos[g]) \
        .union(list_pos[h]).union(list_pos[j]).union(list_pos[k]) \
        .union(list_neg[b]).union(list_neg[c]).union(list_neg[d]).union(list_neg[e]).union(list_neg[f]) \
        .union(list_neg[g]).union(list_neg[h]).union(list_neg[j]).union(list_neg[k])

    nb_model = NaiveBayes.train(train, lambda_value)
    prediction_and_label = test.map(lambda p: (nb_model.predict(p.features), p.label))
    accuracy = 100 * prediction_and_label.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
    total_fold_sum += accuracy


def spark_svm_precision(list_pos, list_neg, iteration, a, b, c, d, e, f, g, h, j, k):
    global total_fold_sum
    test = list_pos[a].union(list_neg[a])

    train = list_pos[b].union(list_pos[c]).union(list_pos[d]).union(list_pos[e]).union(list_pos[f]).union(list_pos[g]) \
        .union(list_pos[h]).union(list_pos[j]).union(list_pos[k]) \
        .union(list_neg[b]).union(list_neg[c]).union(list_neg[d]).union(list_neg[e]).union(list_neg[f]) \
        .union(list_neg[g]).union(list_neg[h]).union(list_neg[j]).union(list_neg[k])

    model = SVMWithSGD.train(train, iteration)

    prediction_and_label = test.map(lambda p: (p.label, model.predict(p.features)))
    accuracy = 100 * prediction_and_label.filter(lambda lp: lp[0] == lp[1]).count() / float(test.count())
    total_fold_sum += accuracy


def spark_logistic_regression_precision(list_pos, list_neg, iteration, a, b, c, d, e, f, g, h, j, k):
    global total_fold_sum
    test = list_pos[a].union(list_neg[a])

    train = list_pos[b].union(list_pos[c]).union(list_pos[d]).union(list_pos[e]).union(list_pos[f]).union(list_pos[g]) \
        .union(list_pos[h]).union(list_pos[j]).union(list_pos[k]) \
        .union(list_neg[b]).union(list_neg[c]).union(list_neg[d]).union(list_neg[e]).union(list_neg[f]) \
        .union(list_neg[g]).union(list_neg[h]).union(list_neg[j]).union(list_neg[k])

    model = LogisticRegressionWithLBFGS.train(train, iterations=iteration)

    prediction_and_label = test.map(lambda p: (p.label, model.predict(p.features)))
    accuracy = 100 * prediction_and_label.filter(lambda lp: lp[0] == lp[1]).count() / float(test.count())
    total_fold_sum += accuracy


def spark_desicion_tree_precision(list_pos, list_neg, numclass, categoricalfeaturesinfo, purity, max_dep, max_bin,
                                  a, b, c, d, e, f, g, h, j, k):
    global total_fold_sum
    test = list_pos[a].union(list_neg[a])

    train = list_pos[b].union(list_pos[c]).union(list_pos[d]).union(list_pos[e]).union(list_pos[f]).union(list_pos[g]) \
        .union(list_pos[h]).union(list_pos[j]).union(list_pos[k]) \
        .union(list_neg[b]).union(list_neg[c]).union(list_neg[d]).union(list_neg[e]).union(list_neg[f]) \
        .union(list_neg[g]).union(list_neg[h]).union(list_neg[j]).union(list_neg[k])

    desicion_model = DecisionTree.trainClassifier(train, numClasses=numclass,
                                                  categoricalFeaturesInfo=categoricalfeaturesinfo,
                                                  impurity=purity, maxDepth=max_dep, maxBins=max_bin)

    predictions = desicion_model.predict(test.map(lambda x: x.features))
    labels_and_predictions = test.map(lambda lp: lp.label).zip(predictions)
    accuracy = 100 * labels_and_predictions.filter(
        lambda lp: lp[0] == lp[1]).count() / float(test.count())
    total_fold_sum += accuracy
