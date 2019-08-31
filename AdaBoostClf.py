from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

all_file = './DYGZ/DYGZ_all.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, :6]
y = (train_df.values[:, 7] == 1).astype(np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

if __name__ == '__main__':
    param_grid = [{'base_estimator__max_depth': [2, 5, 10, 20, 50]}, {'learning_rate': [0.1, .01, .001]}]  #

    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2), n_estimators=5000,
        algorithm="SAMME.R", learning_rate=0.1, random_state=42)
    # ada_clf.fit(train_x, train_y)
    grid_search = GridSearchCV(ada_clf, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(train_x, train_y)
    print(grid_search.best_estimator_)
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))
    y_pred_ada = grid_search.decision_function(test_x)  # .predict
    # print(y_pred_ada)
    # print(test_y)
    test_auc = metrics.roc_auc_score(test_y, y_pred_ada)
    print(test_auc)
    print("Test set score:{:.2f}".format(grid_search.score(test_x, test_y)))

# max_depth=1, n_estimators=1000, algorithm="SAMME.R", learning_rate=0.5:0.5381385738094936
# max_depth=50, n_estimators=1000, algorithm="SAMME.R", learning_rate=0.5:0.718371283751382
# max_depth=2, n_estimators=500, algorithm="SAMME.R", learning_rate=0.5:0.5487494471012595
# max_depth=2, n_estimators=1000, algorithm="SAMME.R", learning_rate=0.5:0.5851214593476274
# max_depth=2, n_estimators=1000, algorithm="SAMME.R", learning_rate=0.5:0.619765428750838
# max_depth=2, n_estimators=5000, algorithm="SAMME.R", learning_rate=0.5:0.6586659752254832
'''AdaBoostClassifier(algorithm='SAMME.R',
                   base_estimator=DecisionTreeClassifier(class_weight=None,
                                                         criterion='gini',
                                                         max_depth=10,
                                                         max_features=None,
                                                         max_leaf_nodes=None,
                                                         min_impurity_decrease=0.0,
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1,
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0,
                                                         presort=False,
                                                         random_state=None,
                                                         splitter='best'),
                   learning_rate=0.1, n_estimators=5000, random_state=42)
Best score on train set:0.91
0.6955022802306676
Test set score:0.91'''
