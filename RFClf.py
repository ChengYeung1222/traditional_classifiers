from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

all_file = './DYGZ/DYGZ_all.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, :6]
y = (train_df.values[:, 7] == 1).astype(np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

if __name__ == '__main__':
    rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=32, random_state=42)

    rnd_clf1 = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(
        n_estimators=100, max_leaf_nodes=32, random_state=42))])
    rnd_clf.fit(train_x, train_y)

    y_pred_rf = rnd_clf.predict_proba(test_x)
    print(y_pred_rf)
    print(test_y)
    test_auc = metrics.roc_auc_score(test_y, y_pred_rf[:, 1])
    print(test_auc)

    # ---------------------------------------#
    output_file = open("./DYGZ/label_prob_DYGZ_RandomForests.csv", 'w')
    predictions = []
    # output_file.write("Prediction , " + "Actual , " + "Accuracy" + '\n')
    known_preds = rnd_clf.predict_proba(X)
    for i, unknown_pred in enumerate(known_preds):
        pred_prob = known_preds[:, 1]
        # pred_label = unknown_pred.argmax(axis=0)
        predictions.append(pred_prob)
        output_file.write(str(i) + ', ' + str(y[i]) + ', ' + str(pred_prob[i]) + '\n')
    output_file.close()
# n_estimators = 500, max_leaf_nodes = 16, random_state = 42:0.5107115648449726
# n_estimators = 100, max_leaf_nodes = 16, random_state = 42:0.5113018836171094
# n_estimators = 100, max_leaf_nodes = 32, random_state = 42:0.8267178508932594
