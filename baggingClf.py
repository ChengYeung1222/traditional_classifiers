from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    all_file = './DYGZ/DYGZ_all.csv'
    train_df = pd.read_csv(all_file)
    X = train_df.values[:, :6]
    y = (train_df.values[:, 7] == 1).astype(np.float64)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(max_depth=100, random_state=42), n_estimators=200,
        max_samples=15000, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42)

    bag_clf1 = Pipeline([('scaler', StandardScaler()), ('bagging', BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=True, random_state=42))])
    bag_clf.fit(train_x, train_y)

    pred = bag_clf.predict_proba(test_x)
    # predict_prob_y = bag_clf.predict(test_x)
    print(pred)
    print(test_y)
    # end svm ,start metrics
    test_auc = metrics.roc_auc_score(test_y, pred[:, 1])
    print(test_auc)

# n_estimators=1000, max_samples=5000, bootstrap=True:0.6493996379240323
# n_estimators=500, max_samples=5000, bootstrap=True:0.6493996379240323
# n_estimators=200, max_samples=5000, bootstrap=True:0.6475621140210984
# n_estimators=200, max_samples=10000, bootstrap=True:0.6713036053263376
# !!!n_estimators=200, max_samples=15000, bootstrap=True:0.9066750873586472
# n_estimators=200, max_samples=20000, bootstrap=True:0.700267743881017

# n_estimators=200, max_samples=20000, bootstrap=False:0.7235152132140408
# n_estimators=200, max_samples=20000, bootstrap=True:0.712464735336324
