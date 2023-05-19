import pandas as pd
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import metrics

# Load dataset
df = pd.read_csv("data/Data.csv", encoding="gbk", header=0,
                 index_col=0).reset_index(drop=True)

# delete outlier
df = df[~(df['Label'].isin([2]))]
y = df["Label"]

# input data
X = df.iloc[:, 0:15]
X = np.array(X)
X = pd.DataFrame(X)

# Input data
np.random.seed(101)

def get_train_x_y():
    y = df["Label"]
    x = PCA(n_components=10).fit_transform(X)
    return x, y

x_train, y_train = get_train_x_y()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    test_size=0.2)

# plt.figure(figsize=(6,4))
# model = KMeans(random_state=1)
# plt.figure(dpi=600)
# visualizer = KElbowVisualizer(model, k=(1,20))
# visualizer.fit(x_train)
# plt.xticks(np.arange(0, 21, 1))
# plt.grid(False)
# visualizer.show()

# Create a Kmeans model project
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300, random_state=100)

# Fit the model to the training data
km_model = kmeans.fit(x_train)

plt.figure(dpi=600)
visualizer = SilhouetteVisualizer(km_model, colors='yellowbrick')
visualizer.fit(x_train)        # Fit the data to the visualizer
plt.savefig("fig/km_silhouette_plot.png")
visualizer.show()        # Finalize and render the figure


# Use the trained model to make predictions on the test data
y_pred = kmeans.predict(x_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

x0 = x_train[kmeans.labels_ == 0]
x1 = x_train[kmeans.labels_ == 1]
plt.figure(dpi=600)
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0', s=9)
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='o', label='label1', s=9)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', label="centroid", s=100, alpha=1)
plt.legend(loc="upper left")
plt.grid(False)
plt.savefig("fig/kmeans.png")
plt.show()


# 5-fold cv
kfold = KFold(n_splits = 5)
cv_cross = cross_validate(km_model, x_train, y_train, cv = kfold, scoring = ("accuracy", "f1"))
print(cv_cross)

# fpr, tpr and auc (plot)
y_pre = km_model.predict(x_test)
y_0 = list(y_pre)
fpr, tpr, thresholds = roc_curve(y_test, y_0)
auc = roc_auc_score(y_test, y_0)

print(classification_report(y_test, y_pred, labels=[0, 1]))

plt.figure(dpi=600)
plt.figure(figsize = (6, 6))
plt.title("Validation ROC")
plt.plot(fpr, tpr, "b", label = "Val AUC = %0.3f" % auc)
plt.legend(loc = "lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.grid(False)
plt.savefig("fig/km_roc_curve.png")
plt.show()


scores = []
for i in range(2, 21):
    km = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km.fit(x_train)
    scores.append(metrics.silhouette_score(x_train, km.labels_, metric="euclidean"))
plt.figure(dpi=600)
plt.style.use("grayscale")
plt.plot(range(2, 21), scores, marker="o")
plt.xticks(np.arange(2, 21, 1))
plt.xlabel("Number of clusters")
plt.ylabel("silhouette_score")
plt.grid(False)
plt.savefig("fig/km_silhouette_score.png")
plt.show()






