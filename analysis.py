import json
import re
import settings
from visualization import graph_vis
from visualization import heatmap_vis

def analyize_section(rdd_batch, sp_tf, nb_model, svm_pos_model, svm_neg_model, log_pos_model, log_neg_model, tree_model,
                     sc_tf, sc_nb_model, sc_svm_model, sc_log_model, sc_tree_model):
    # global total_tweets, tree_neg_count, tree_pos_count, lg_neg_count, lg_pos_count, svm_neg_count, \
    #     svm_pos_count, nb_neg_count, nb_pos_count, bos_str, sc_nb_pos_count, sc_nb_neg_count, \
    #     sc_svm_pos_count, sc_svm_neg_count, sc_log_pos_count, sc_log_neg_count, sc_tree_pos_count, sc_tree_neg_count

    print("*****************************************************************************************")
    settings.total_tweets += rdd_batch.count()
    for i in rdd_batch.toLocalIterator():
        msg = json.loads(i)
        tweet_text = re.sub('\s+', ' ', msg['text']).lower()
        location = msg['user']['location']
        if location is not None: settings.location.append((location.split(",")[0]).lower())
        settings.text.append(tweet_text)
       # print("Tweeti kendisi =" + str(tweet_text))
        vector = sp_tf.transform(tweet_text.split(" "))
        nb_returned_label = nb_model.predict(vector)
        if nb_returned_label == 1.0:
            settings.nb_pos_count += 1
        if nb_returned_label == 2.0:
            settings.nb_neg_count += 1
        svm_pos_returned = svm_pos_model.predict(vector)
        if svm_pos_returned == 1.0:
            settings.svm_pos_count += 1
        svm_neg_returned = svm_neg_model.predict(vector)
        if svm_neg_returned == 1.0:
            settings.svm_neg_count += 1
        log_pos_returned = log_pos_model.predict(vector)
        if log_pos_returned == 1.0:
            settings.lg_pos_count += 1
        log_neg_returned = log_neg_model.predict(vector)
        if log_neg_returned == 1.0:
            settings.lg_neg_count += 1
        tree_returned = tree_model.predict(vector)
        if tree_returned == 1.0:
            settings.tree_pos_count += 1
        if tree_returned == 0.0:
            settings.tree_neg_count += 1

        sc_vector = sc_tf.transform([tweet_text])
        sc_nb_returned = sc_nb_model.predict(sc_vector)
        if sc_nb_returned == 1.0:
            settings.sc_nb_pos_count += 1
        if sc_nb_returned == 2.0:
            settings.sc_nb_neg_count += 1
        sc_svm_returned = sc_svm_model.predict(sc_vector)
        if sc_svm_returned == 1.0:
            settings.sc_svm_pos_count += 1
        if sc_svm_returned == 2.0:
            settings.sc_svm_neg_count += 1
        sc_log_returned = sc_log_model.predict(sc_vector)
        if sc_log_returned == 1.0:
            settings.sc_log_pos_count += 1
        if sc_log_returned == 2.0:
            settings.sc_log_neg_count += 1
        sc_tree_returned = sc_tree_model.predict(sc_vector)
        if sc_tree_returned == 1.0:
            settings.sc_tree_pos_count += 1
        if sc_tree_returned == 2.0:
            settings.sc_tree_neg_count += 1

    graph_vis.p_result()
    graph_vis.pie_chart()
    heatmap_vis.heat_g()

    print("-------------------------------------------------------")
    print("Analyzed Tweets=    " + str(settings.total_tweets))
    print("Spark Naive Bayes Positive Count =     " + str(settings.nb_pos_count))
    print("Spark Naive Bayes Positive Result = %.2f" % (100 * (settings.nb_pos_count / settings.total_tweets)))
    print("Spark Naive Bayes Negative Count =     " + str(settings.nb_neg_count))
    print("Spark Naive Bayes Negative Result =   %.2f" % (100 * (settings.nb_neg_count / settings.total_tweets)))
    print(" ")
    print("Spark SVM Positive Count =     " + str(settings.svm_pos_count))
    print("Spark SVM Positive Result = %.2f" % (100 * (settings.svm_pos_count / settings.total_tweets)))
    print("Spark SVM Negative Count =     " + str(settings.svm_neg_count))
    print("Spark SVM Negative Result = %.2f" % (100 * (settings.svm_neg_count / settings.total_tweets)))
    print(" ")
    print("Spark Logistic Regression Positive Count =     " + str(settings.lg_pos_count))
    print("Spark Logistic Regression Positive Result = %.2f" % (100 * (settings.lg_pos_count / settings.total_tweets)))
    print("Spark Logistic Regression Negative Count =     " + str(settings.lg_neg_count))
    print("Spark Logistic Regression Negative Result = %.2f" % (100 * (settings.lg_neg_count / settings.total_tweets)))
    print(" ")
    print("Spark Decision Tree Positive Count = " + str(settings.tree_pos_count))
    print("Spark Decision Tree Positive Result = %.2f" % (100 * (settings.tree_pos_count / settings.total_tweets)))
    print("Spark Decision Tree Negative Count = " + str(settings.tree_neg_count))
    print("Spark Decision Tree Negative Result = %.2f" % (100 * (settings.tree_neg_count / settings.total_tweets)))
    print(" ")
    print("Scikit Learn  Naive Bayes Positive Count =" + str(settings.sc_nb_pos_count))
    print("Scikit Naive Bayes Positive Result =%.2f " % (100 * (settings.sc_nb_pos_count / settings.total_tweets)))
    print("Scikit Learn  Naive Bayes Negative Count =" + str(settings.sc_nb_neg_count))
    print("Scikit Naive Bayes Positive Result = %.2f" % (100 * (settings.sc_nb_neg_count / settings.total_tweets)))
    print(" ")
    print("Scikit Learn SVM Positive Count =" + str(settings.sc_svm_pos_count))
    print("Scikit Learn SVM Positive Result =%.2f " % (100 * (settings.sc_svm_pos_count / settings.total_tweets)))
    print("Scikit Learn SVM Negative Count =" + str(settings.sc_svm_neg_count))
    print("Scikit Learn SVM Negative Result =%.2f " % (100 * (settings.sc_svm_neg_count / settings.total_tweets)))
    print(" ")
    print("Scikit Learn Logistic Regression Count =" + str(settings.sc_log_pos_count))
    print("Scikit Logistic Reg Positive Result =%.2f " % (100 * (settings.sc_log_pos_count / settings.total_tweets)))
    print("Scikit Learn Logistic Regression Negative Count =" + str(settings.sc_log_neg_count))
    print("Scikit Logistic Reg Negative Result =%.2f " % (100 * (settings.sc_log_neg_count / settings.total_tweets)))
    print(" ")
    print("Scikit Learn Decision Tree Count =" + str(settings.sc_tree_pos_count))
    print("Scikit Decision Tree Positive Result =%.2f " % (100 * (settings.sc_tree_pos_count / settings.total_tweets)))
    print("Scikit Learn Decision Tree Negative Count =" + str(settings.sc_tree_neg_count))
    print("Scikit Decision Tree Negative Result =%.2f " % (100 * (settings.sc_tree_neg_count / settings.total_tweets)))
    print("*****************************************************************************************")
