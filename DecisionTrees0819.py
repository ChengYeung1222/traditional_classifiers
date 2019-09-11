from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# from graphviz import Source
# from sklearn.tree import export_graphviz
import os

all_file = './DYGZ/DYGZ_all.csv'
train_df = pd.read_csv(all_file)
X = train_df.values[:, :6]
y = (train_df.values[:, 7] == 1).astype(np.float64)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27)

tree_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("dt_clf", DecisionTreeClassifier(max_depth=100))
])
tree_clf.fit(train_x, train_y)

# pred = tree_clf.predict(test_x)
predict_prob_y = tree_clf.predict_proba(test_x)
print(predict_prob_y)
print(test_y)
# end dt ,start metrics
test_auc = metrics.roc_auc_score(test_y, predict_prob_y[:, 1])
print(test_auc)

# ---------------------------------------#
output_file = open("./DYGZ/label_prob_DYGZ_DecisionTrees.csv", 'w')
predictions = []
# output_file.write("Prediction , " + "Actual , " + "Accuracy" + '\n')
known_preds = tree_clf.predict_proba(X)
for i, unknown_pred in enumerate(known_preds):
    pred_prob = known_preds[:, 1]
    # pred_label = unknown_pred.argmax(axis=0)
    predictions.append(pred_prob)
    output_file.write(str(i) + ', ' + str(y[i]) + ', ' + str(pred_prob[i]) + '\n')
output_file.close()

# export_graphviz(
#         tree_clf,
#         out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
#         feature_names=iris.feature_names[2:],
#         class_names=iris.target_names,
#         rounded=True,
#         filled=True
#     )

# Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))

# max_depth=100:0.7203612021873462
# max_depth=None:0.7163384518421643
