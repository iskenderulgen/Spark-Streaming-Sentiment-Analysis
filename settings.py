from pyspark.mllib.feature import HashingTF

socket_error_no = None
# Spark Setting

sp_tf = HashingTF(100)
iteration = 5
numclass = 2
categoricalfeaturesinfo = {}
purity = 'gini'
max_dep = 5
max_bin = 30
lambda_val = 1.0

# Spark model Accuracy Results
sp_naive_bayes_acc = 0
sp_svm_acc = 0
sp_log_acc = 0
sp_tree_acc = 0

sp_naive_cross_acc = 0
sp_svm_cross_acc = 0
sp_log_cross_acc = 0
sp_tree_cross_acc = 0

# Spark Model Accuracy Results End

# Scikit Model Accuracy Results

sc_naive_bayes_acc = 0
sc_svm_acc = 0
sc_log_acc = 0
sc_tree_acc = 0

# Scikit Model Accuracy Ends

# Spark Analysis Results

nb_pos_count, nb_neg_count, svm_pos_count, svm_neg_count, lg_pos_count, lg_neg_count, tree_pos_count, tree_neg_count, \
total_tweets = 0, 0, 0, 0, 0, 0, 0, 0, 1
sc_nb_pos_count, sc_nb_neg_count, sc_svm_pos_count, sc_svm_neg_count, sc_log_pos_count, sc_log_neg_count, \
sc_tree_pos_count, sc_tree_neg_count = 0, 0, 0, 0, 0, 0, 0, 0

# Spark Analysis Results Ends
location = []
text = []


# for i in settings.location:
#     print(i.split(","))

