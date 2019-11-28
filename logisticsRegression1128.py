from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics

all_file = './DYGZ/DYGZ_all.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, :6]
y = (train_df.values[:, 7] == 1).astype(np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

log_clf = Pipeline([('scaler', StandardScaler()), ('log_clf', LogisticRegression(solver="lbfgs", random_state=42))])
log_clf.fit(train_x, train_y)
y_pred = log_clf.predict_proba(test_x)
print(log_clf.__class__.__name__, metrics.roc_auc_score(test_y, y_pred[:, 1]))

output_file = open("./DYGZ/label_prob_DYGZ_LogisticRegression.csv", 'w')
predictions = []
# output_file.write("Prediction , " + "Actual , " + "Accuracy" + '\n')
known_preds = log_clf.predict_proba(X)
for i, unknown_pred in enumerate(known_preds):
    pred_prob = known_preds[:,1]
    # pred_label = unknown_pred.argmax(axis=0)
    predictions.append(pred_prob)
    output_file.write(str(i) + ', ' + str(y[i]) + ', ' + str(pred_prob[i]) + '\n')
output_file.close()