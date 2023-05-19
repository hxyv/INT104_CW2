import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import manifold, datasets
from sklearn.decomposition import NMF

# Load dataset
df = pd.read_csv("data/Data.csv", encoding = "gbk", header = 0, index_col = 0).reset_index(drop = True)

# delete outlier
df = df[~(df['Label'].isin([2]))]
y = df["Label"]

# input data
X = df.iloc[:, 0:15]
X = pd.DataFrame(X)

# Input data
np.random.seed(10)
def get_train_x_y():
    y = df["Label"]
    x = PCA(n_components=10).fit_transform(X)
    # x = NMF(n_components=10, init='random', random_state=0).fit_transform(X)
    # x = manifold.TSNE(n_components=2, init="pca", random_state=1).fit_transform(X)
    # x = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    return x, y

x_train, y_train = get_train_x_y()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)

# use random forest modeling
RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(x_train, y_train)

# Use the trained model to make predictions on the train and test data
y_train_pred = RF_model.predict(x_train)
y_pred = RF_model.predict(x_test)

# Evaluate the accuracy of the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)
print("Train set accuracy: ", train_accuracy)
print("Test set accuracy: ", test_accuracy)

# 5-fold cv 交叉验证
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
from sklearn.model_selection import cross_validate
cv_cross = cross_validate(RF_model, x_train, y_train, cv=kfold, scoring=('accuracy', 'f1'))

# fpr, tpr and auc (plot)#假阳性率，真阳性率
y_pre = RF_model.predict_proba(x_test)
y_0=list(y_pre[:,1])
fpr,tpr,thresholds=roc_curve(y_test,y_0)
auc=roc_auc_score(y_test,y_0)

plt.figure(figsize=(6,6))
plt.title('Validation ROC - Random forest')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
