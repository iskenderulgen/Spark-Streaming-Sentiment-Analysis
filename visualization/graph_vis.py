import settings
import matplotlib.pyplot as plt
import numpy as np


def model_evaluations():
    n_groups = 4
    spark_md = [settings.sp_naive_bayes_acc, settings.sp_svm_acc, settings.sp_log_acc, settings.sp_tree_acc]
    scikit_md = [settings.sc_naive_bayes_acc, settings.sc_svm_acc, settings.sc_log_acc, settings.sc_tree_acc]
    index = np.arange(n_groups)
    bar_width = 0.35
    plt.bar(index, spark_md, bar_width, color='b', label='Spark')
    plt.bar(index + bar_width, scikit_md, bar_width, color='g', label='Scikit')
    plt.xlabel('Methods')
    plt.ylabel('Percent')
    plt.title('Model Evaluation')
    plt.xticks(index + bar_width / 2, ('NB', 'SVM', 'LOG', 'TREE'))
    plt.legend(loc='best')
    plt.show()


def p_result():
    n_groups = 4
    spark_ps = [100 * (settings.nb_pos_count / settings.total_tweets),
                100 * (settings.svm_pos_count / settings.total_tweets),
                100 * (settings.lg_pos_count / settings.total_tweets),
                100 * (settings.tree_pos_count / settings.total_tweets)]
    scikit_ps = [100 * (settings.sc_nb_pos_count / settings.total_tweets),
                 100 * (settings.sc_svm_pos_count / settings.total_tweets),
                 100 * (settings.sc_log_pos_count / settings.total_tweets),
                 100 * (settings.sc_tree_pos_count / settings.total_tweets)]
    spark_ps1 = [100 * (settings.nb_neg_count / settings.total_tweets),
                 100 * (settings.svm_neg_count / settings.total_tweets),
                 100 * (settings.lg_neg_count / settings.total_tweets),
                 100 * (settings.tree_neg_count / settings.total_tweets)]
    scikit_ps1 = [100 * (settings.sc_nb_neg_count / settings.total_tweets),
                  100 * (settings.sc_svm_neg_count / settings.total_tweets),
                  100 * (settings.sc_log_neg_count / settings.total_tweets),
                  100 * (settings.sc_tree_neg_count / settings.total_tweets)]
    index = np.arange(n_groups)
    bar_width = 0.35
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.bar(index, spark_ps, bar_width, color='orange', label='Spark')
    plt.bar(index + bar_width, scikit_ps, bar_width, color='blue', label='Scikit')
    plt.xlabel('Methods')
    plt.ylabel('Percent')
    plt.title('Pozitif Analysis Result')
    plt.xticks(index + bar_width / 2, ('NB', 'SVM', 'LOG', 'TREE'))
    plt.legend(loc='best')
    fig.add_subplot(1, 2, 2)
    plt.bar(index, spark_ps1, bar_width, color='orange', label='Spark')
    plt.bar(index + bar_width, scikit_ps1, bar_width, color='blue', label='Scikit')
    plt.xlabel('Methods')
    plt.ylabel('Percent')
    plt.title('Negatif Analysis Result')
    plt.xticks(index + bar_width / 2, ('NB', 'SVM', 'LOG', 'TREE'))
    plt.legend(loc='best')

    plt.show()


def pie_chart():
    labels = ['Pozitif', 'Negatif']
    a = (settings.nb_pos_count + settings.svm_pos_count + settings.lg_pos_count + settings.tree_pos_count) / 4
    b = (settings.nb_neg_count + settings.svm_neg_count + settings.lg_neg_count + settings.tree_neg_count) / 4
    c = (
                settings.sc_nb_pos_count + settings.sc_svm_pos_count + settings.sc_log_pos_count + settings.sc_tree_pos_count) / 4
    d = (
                settings.sc_nb_neg_count + settings.sc_svm_neg_count + settings.sc_log_neg_count + settings.sc_tree_neg_count) / 4
    colors = ['#6666ff', '#ff0033']
    values = [a, b]
    value = [c, d]
    fig = plt.figure()
    # First plot
    fig.add_subplot(1, 2, 1)
    plt.axis('equal')
    plt.pie(values, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title('Spark Pie Chart')
    fig.add_subplot(1, 2, 2)
    plt.axis('equal')
    plt.pie(value, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title('Scikit-Learn Pie Chart')

    plt.show()
