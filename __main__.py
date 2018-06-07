"""
   GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""


import os
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from Spark import Sp_ML_Models, Sp_Model_Evaluation, Sp_Cross_Validation
from Scikit_Learn import Sc_ML_Models, Sc_Model_Evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
import analysis
import settings
from visualization import graph_vis

positive_path = "/home/ozlem/Documents/pycharm_workspace/Spark_Twitter/Keywords/positive.csv"
negative_path = "/home/ozlem/Documents/pycharm_workspace/Spark_Twitter/Keywords/negative.csv"
os.environ['PYSPARK_PYTHON'] = "/home/ozlem/anaconda3/envs/pytwitter/bin/python3.6"


if __name__ == '__main__':
    #   ********************************* Spark Section Starts Here ***************************************
    conf = SparkConf().setMaster("local[4]").setAppName("Twitter")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 5)
    sc.setLogLevel('ERROR')

    sp_pos_label_1 = Sp_ML_Models.spark_labeling_data(positive_path, 1.0, sc, settings.sp_tf)
    sp_pos_label_0 = Sp_ML_Models.spark_labeling_data(positive_path, 0.0, sc, settings.sp_tf)
    sp_neg_label_1 = Sp_ML_Models.spark_labeling_data(negative_path, 1.0, sc, settings.sp_tf)
    sp_neg_label_0 = Sp_ML_Models.spark_labeling_data(negative_path, 0.0, sc, settings.sp_tf)
    sp_nb_pos_label = Sp_ML_Models.spark_labeling_data(positive_path, 1.0, sc, settings.sp_tf)
    sp_nb_neg_label = Sp_ML_Models.spark_labeling_data(negative_path, 2.0, sc, settings.sp_tf)

    # Naive Bayes Model Training and Evaluation Starts
    nb_model = Sp_ML_Models.spark_naive_bayes_model(sp_nb_pos_label.union(sp_nb_neg_label), settings.lambda_val)
    Sp_Model_Evaluation.spark_naive_bayes_precision(sp_nb_pos_label.union(sp_nb_neg_label), settings.lambda_val)
    # Naive Bayes Model Training and Evaluation Completed

    # SVM Model Training and Evaluation Starts
    svm_pos_model = Sp_ML_Models.spark_svm_model(sp_pos_label_1.union(sp_neg_label_0), settings.iteration)
    svm_neg_model = Sp_ML_Models.spark_svm_model(sp_neg_label_1.union(sp_pos_label_0), settings.iteration)
    Sp_Model_Evaluation.spark_svm_precision(sp_pos_label_1.union(sp_neg_label_0), settings.iteration, "positive")
    Sp_Model_Evaluation.spark_svm_precision(sp_neg_label_1.union(sp_pos_label_0), settings.iteration, "negative")
    # SVM Model Training and Evaluation Completed

    # Logistic Regression Model Training Stars
    log_pos_model = Sp_ML_Models.spark_logistic_regression_model(sp_pos_label_1.union(sp_neg_label_0),
                                                                 settings.iteration)
    log_neg_model = Sp_ML_Models.spark_logistic_regression_model(sp_neg_label_1.union(sp_pos_label_0),
                                                                 settings.iteration)
    Sp_Model_Evaluation.spark_logistic_regression_precision(sp_pos_label_1.union(sp_neg_label_0), settings.iteration,
                                                            "positive")
    Sp_Model_Evaluation.spark_logistic_regression_precision(sp_neg_label_1.union(sp_pos_label_0), settings.iteration,
                                                            "negative")
    # Logistic Regression Model Training Completed

    # Desicion Tree Model Training Starts
    tree_model = Sp_ML_Models.spark_desicion_tree_model(sp_pos_label_1.union(sp_neg_label_0), settings.numclass,
                                                        settings.categoricalfeaturesinfo, settings.purity,
                                                        settings.max_bin, settings.max_dep)
    Sp_Model_Evaluation.spark_desicion_tree_precision(sp_pos_label_1.union(sp_neg_label_0), settings.numclass,
                                                      settings.categoricalfeaturesinfo, settings.purity,
                                                      settings.max_dep, settings.max_bin, "positive")
    Sp_Model_Evaluation.spark_desicion_tree_precision(sp_neg_label_1.union(sp_pos_label_0), settings.numclass,
                                                      settings.categoricalfeaturesinfo, settings.purity,
                                                      settings.max_dep,
                                                      settings.max_bin, "negative")
    # Desicion Tree Model Training Completed

    # Sp_Cross_Validation.cross_validation(positive_path, negative_path, lambda_val, tf, iteration, numclass,
    #                                    categoricalfeaturesinfo, purity, max_dep, max_bin, sc)

    #   ********************************* Spark Section Ends Here ***************************************

    # ********************************* Scikit Section Starts Here ***************************************

    sc_tf = TfidfVectorizer(analyzer='word', max_features=20, stop_words=None, tokenizer=None,
                            lowercase=None, norm="l2", smooth_idf=True)

    vectors, labels = Sc_ML_Models.scikit_labeling_data(positive_path, negative_path, 1.0, 2.0, sc_tf)

    # Scikit Naive Bayes Starts Here
    sc_nb_model = Sc_ML_Models.scikit_naive_bayes(vectors, labels)
    Sc_Model_Evaluation.scikit_learn_naivebayes_test(vectors, labels)
    # Scikit Naive Bayes Ends Here

    # Scikit SVM Stars Here
    sc_svm_model = Sc_ML_Models.scikit_svm_model(vectors, labels)
    Sc_Model_Evaluation.scikit_learn_svm_test(vectors, labels)

    # Scikit SVM Ends Here

    # Scikit Logistic Regression Starts Here
    sc_log_model = Sc_ML_Models.scikit_logistic_regression(vectors, labels)
    Sc_Model_Evaluation.scikit_learn_logistic_tet(vectors, labels)
    # Scikit Logistic Regression Starts Here

    # Scikit Decision Tree Starts Here
    sc_tree_model = Sc_ML_Models.scikit_decision_tree(vectors, labels)
    Sc_Model_Evaluation.scikit_learn_tree_test(vectors, labels)
    # Scikit Decision Tree Ends Here
    graph_vis.model_evaluations()

Dstream = ssc.socketTextStream("localhost", 9995)
Dstream.foreachRDD(lambda rdd: analysis.analyize_section(rdd, settings.sp_tf, nb_model, svm_pos_model, svm_neg_model,
                                                         log_pos_model, log_neg_model, tree_model, sc_tf,
                                                         sc_nb_model, sc_svm_model, sc_log_model, sc_tree_model))
ssc.start()
ssc.awaitTermination()
