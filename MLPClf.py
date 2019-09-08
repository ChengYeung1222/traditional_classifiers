from sklearn.neural_network import MLPClassifier
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

mlp_clf = Pipeline([("scaler", StandardScaler()),
                    ('mlp_clf', MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[50]))])

mlp_clf.fit(train_x,train_y)
predict_prob_y = mlp_clf.predict_proba(test_x)
print(predict_prob_y)
print(test_y)
# end svm ,start metrics
predict_prob_train=mlp_clf.predict_proba(train_x)
train_auc = metrics.roc_auc_score(train_y, predict_prob_train[:,1])
print(train_auc)
test_auc = metrics.roc_auc_score(test_y, predict_prob_y[:,1])
print(test_auc)

# hidden_layer_sizes=[50, 50]
# 0.8784009654898947
# 0.8506412574318946
# hidden_layer_sizes=[50]
# 0.8607950398302288
# 0.8348838743731853