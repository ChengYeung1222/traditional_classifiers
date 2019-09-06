from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

all_file = './DYGZ/DYGZ_all.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, :6]
y = (train_df.values[:, 7] == 1).astype(np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=.1, C=.5,probability=True))
])

rbf_kernel_svm_clf.fit(train_x, train_y)

predict_prob_y = rbf_kernel_svm_clf.predict_proba(test_x)
print(predict_prob_y)
print(test_y)
# end svm ,start metrics
predict_prob_train=rbf_kernel_svm_clf.predict_proba(train_x)
train_auc = metrics.roc_auc_score(train_y, predict_prob_train[:,1])
print(train_auc)
test_auc = metrics.roc_auc_score(test_y, predict_prob_y[:,1])
print(test_auc)

# gamma=1., C=1e+1
# 0.8959166278361342
# 0.8504614996165858
# gamma = .5, C = 1e+1
# 0.8653428859061038
# 0.8287439240819363
# gamma=.5, C=1.
# 0.8406477906814893
# 0.8089351063896111
# gamma=.1, C=.5
# 0.7783610614230418
# 0.7679773401246459

