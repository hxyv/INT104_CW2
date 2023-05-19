import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import manifold, datasets
from functools import partial
from sklearn.decomposition import NMF
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/Data.csv", encoding="gbk", header=0, index_col=0).reset_index(drop=True)

# delete outlier
df = df[~(df['Label'].isin([2]))]
y = df["Label"]

# input data
X = df.iloc[:, 0:15]
X = np.array(X)
X = pd.DataFrame(X)

methods = [PCA(n_components=10),
           NMF(n_components=10, init='random', random_state=0),
           manifold.Isomap(n_neighbors=10, n_components=2),
           manifold.MDS(n_components=2, max_iter=100, n_init=1)]

names = ["PCA",
         "NMF",
         "Isomap",
         "MDS"]

colors = ["yellow",
          "steelblue",
          "orange",
          "green"]

plt.figure(figsize=(8, 10), dpi=200)

for (name, method, colorname) in zip(names, methods, colors):
    def get_train_x_y():
        y = df["Label"]
        x = method.fit_transform(X)
        return x, y


    x_train, y_train = get_train_x_y()
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    # Create the logistic regression model
    lr = LogisticRegression(random_state=0)

    # Train model
    lr.fit(x_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(x_test)

    # 5-fold cv
    kfold = KFold(n_splits=5)
    cv_cross = cross_validate(lr, x_train, y_train, cv=kfold, scoring=("accuracy", "f1"))
    print(cv_cross)

    # fpr, tpr and auc (plot)
    y_test_predprob = lr.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)

    plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
    plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc='lower right', fontsize=20)
plt.savefig("fig/lr_roc.png")
plt.show()
