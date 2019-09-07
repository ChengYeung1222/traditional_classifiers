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

param_grid = [{'svm_clf__gamma': [1e-3, .1, 5],  # overfitting: reduce gamma
               'svm_clf__C': [.1, 1, 10, 100]}]
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel='rbf', probability=True))
])

grid_search = GridSearchCV(rbf_kernel_svm_clf, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(train_x, train_y)
print(grid_search.best_estimator_)
print("Best score on train set:{:.2f}".format(grid_search.best_score_))
