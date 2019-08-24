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

polynomial_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", coef0=1, degree=2, C=10))
])

polynomial_svm_clf.fit(train_x, train_y)

pred = polynomial_svm_clf.predict(test_x)
predict_prob_y = polynomial_svm_clf.decision_function(test_x)
print(predict_prob_y)
print(test_y)
# end svm ,start metrics
test_auc = metrics.roc_auc_score(test_y, predict_prob_y)
print(test_auc)
# param_grid = [{'svm_clf__degree': [2, 3, 10]}, {'svm_clf__C': [0.1, 1, 5, 25]}]
#
# grid_search = GridSearchCV(polynomial_svm_clf, param_grid, cv=3, scoring='roc_auc')
# grid_search.fit(train_x, train_y)
# print(grid_search.best_estimator_)

# coef0=1, degree=3, C=5:0.7463369085082165
# coef0=1, degree=3, C=1:0.7408726790855501
# coef0=1, degree=2, C=10:0.746953983572
