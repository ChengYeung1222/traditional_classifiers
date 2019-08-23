from sklearn import cross_validation,metrics
from sklearn import svm
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os
import pandas as pd

all_file='./DYGZ/DYGZ_all.csv'
train_df=pd.read_csv(all_file)
X = train_df.values[:,:6]
y = (train_df.values[:,7] == 1).astype(np.float64)
train_x,test_x,train_y,test_y = cross_validation.train_test_split(X,y,test_size=0.2,random_state=27)
#start svm
svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])
svm_clf.fit(train_x,train_y)

predict_prob_y = svm_clf.decision_function(test_x)
print(predict_prob_y)
print(test_y)
#end svm ,start metrics
test_auc = metrics.roc_auc_score(test_y,predict_prob_y)
print(test_auc)#linear svc: 0.6739436887335847