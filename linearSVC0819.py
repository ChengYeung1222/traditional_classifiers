from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os
import pandas as pd

all_file = './DYGZ/DYGZ_all.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, :6]
y = (train_df.values[:, 7] == 1).astype(np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)
# start svm
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc",
     LinearSVC(C=1.0, tol=0.0001, max_iter=1000, penalty='l2', loss='squared_hinge', dual=True,
               fit_intercept=True, intercept_scaling=1)),
])
svm_clf.fit(train_x, train_y)

predict_prob_y = svm_clf.decision_function(test_x)
print(predict_prob_y)
print(test_y)
# end svm ,start metrics
predict_prob_train=svm_clf.decision_function(train_x)
train_auc = metrics.roc_auc_score(train_y, predict_prob_train)
print(train_auc)
test_auc = metrics.roc_auc_score(test_y, predict_prob_y)
print(test_auc)  # linear svc: 0.6739436887335847

# ---------------------------------------#
# output_file = open("./DYGZ/label_prob_DYGZ_LinearSVM.csv", 'w')
# predictions = []
# # output_file.write("Prediction , " + "Actual , " + "Accuracy" + '\n')
# known_preds = svm_clf.decision_function(X)
# for i, unknown_pred in enumerate(known_preds):
#     pred_prob = known_preds
#     # pred_label = unknown_pred.argmax(axis=0)
#     predictions.append(pred_prob)
#     output_file.write(str(i) + ', ' + str(y[i]) + ', ' + str(pred_prob[i]) + '\n')
# output_file.close()
