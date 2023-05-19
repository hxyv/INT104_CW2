import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn import metrics
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import manifold, datasets
from sklearn.decomposition import NMF

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
np.random.seed(10)
def get_train_x_y():
    y = df["Label"]
    # x = PCA(n_components=10).fit_transform(X)
    # x = NMF(n_components=10, init='random', random_state=0).fit_transform(X)
    # x = manifold.TSNE(n_components=2, init="pca", random_state=1).fit_transform(X)
    # x = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    x = manifold.MDS(n_components=2, max_iter=100, n_init=1).fit_transform(X)
    return x, y

x_train, y_train = get_train_x_y()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)

# Create an SVM model project
gnb_model = GaussianNB()

# Fit the model to the training data
gnb_model.fit(x_train, y_train)

# Use the trained model to make predictions on the train and test data
y_train_pred = gnb_model.predict(x_train)
y_pred = gnb_model.predict(x_test)

# Evaluate the accuracy of the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)
print("Train set accuracy: ", train_accuracy)
print("Test set accuracy: ", test_accuracy)

# 5-fold cv
kfold = KFold(n_splits = 5)
cv_cross = cross_validate(gnb_model, x_train, y_train, cv = kfold, scoring = ("accuracy", "f1"))
print(cv_cross)

# fpr, tpr and auc (plot)
y_pre = gnb_model.predict_proba(x_test)
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