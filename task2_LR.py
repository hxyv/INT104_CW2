import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import manifold, datasets
from sklearn.decomposition import NMF
from sklearn.metrics import f1_score


# Load dataset
df = pd.read_csv("data/Data.csv", encoding = "gbk", header = 0, index_col = 0).reset_index(drop = True)

# delete outlier
df = df[~(df['Label'].isin([2]))]
y = df["Label"]

# input data
X = df.iloc[:, 0:15]
X = np.array(X)
X = pd.DataFrame(X)

# Input data
np.random.seed(1)
def get_train_x_y():
    y = df["Label"]
    x = PCA(n_components=10).fit_transform(X)
    # x = NMF(n_components=10, init='random', random_state=0).fit_transform(X)
    # x = manifold.TSNE(n_components=2, init="pca", random_state=1).fit_transform(X)
    # x = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    # x = manifold.MDS(n_components=2, max_iter=100, n_init=1).fit_transform(X)
    return x, y

x_train, y_train = get_train_x_y()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)

# Create the logistic regression model
clf = LogisticRegression(random_state=0)

# Train model
clf.fit(x_train, y_train)

# Predict on the test set
y_pred = clf.predict(x_test)
y_train_pred = clf.predict(x_train)

print(f1_score(y_train, y_train_pred, average="weighted"))
print(f1_score(y_test, y_pred, average='weighted'))

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# 5-fold cv
kfold = KFold(n_splits = 5)
cv_cross = cross_validate(clf, x_train, y_train, cv = kfold, scoring = ("accuracy", "f1"))
print(cv_cross)


# fpr, tpr and auc (plot)
y_pre = clf.predict_proba(x_test)
y_0 = list(y_pre[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, y_0)
auc = roc_auc_score(y_test, y_0)

print(classification_report(y_test, y_pred, labels=[0, 1]))

plt.figure(figsize = (6, 6))
plt.title("Validation ROC")
plt.plot(fpr, tpr, "b", label = "Val AUC = %0.3f" % auc)
plt.legend(loc = "lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()