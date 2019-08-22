from sklearn import cross_validation, metrics
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
    ("svm_clf", SVC(kernel="rbf", gamma=5., C=5e+2))
])

rbf_kernel_svm_clf.fit(train_x, train_y)

pred = rbf_kernel_svm_clf.predict(test_x)
predict_prob_y = rbf_kernel_svm_clf.decision_function(test_x)
print(predict_prob_y)
print(test_y)
# end svm ,start metrics
test_auc = metrics.roc_auc_score(test_y, predict_prob_y)
print(test_auc)
# gamma=0.5,C=1e-2:0.78924081671061
# gamma=1., C=1e-1:0.7982063719269364
# gamma=.5, C=1.:0.799752808053463
# gamma=1., C=1.:0.8054106651355691
# gamma=5., C=1.:0.814152346449259
# gamma=5., C=10.:0.8345287493126735
# gamma=5., C=100.:0.8640346058384555
# gamma=5., C=1e+3:0.8711179140743007
# gamma=10., C=1e+3:0.8397034420483274
# gamma=5., C=5e+2:0.8714426087875935
