import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizerx`
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

# import data
df = pd.read_csv(r"D:\Desktop\INT104\CW2\int104cw2\data\Data.csv", encoding='gbk', header=0, index_col=0).reset_index(drop=True)
df

# delete outlier
df = df[~(df['Label'].isin([2]))]
y = df["Label"]

# input data
x = df.iloc[:, 0:15]
x = np.array(x)

for i in range(2, 14):
    X = x
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=i).fit_transform(X)

    np.random.seed(3)
    kmeans = KMeans(n_clusters=2)
    km_model = kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    df["Label"] = kmeans.labels_

    x0 = X[kmeans.labels_ == 0]
    x1 = X[kmeans.labels_ == 1]
    # x2 = X[kmeans.labels_ == 2]
    # x3 = X[kmeans.labels_ == 3]
    # x4 = X[kmeans.labels_ == 3]
    # x5 = X[kmeans.labels_ == 3]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0', s=2)
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='o', label='label1', s=2)
    # plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='o', label='label2', s=2)
    # plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='o', label='label3', s=2)
    # plt.scatter(x4[:, 0], x4[:, 1], c="pink", marker='o', label='label4', s=2)
    # plt.scatter(x5[:, 0], x5[:, 1], c="pink", marker='o', label='label4', s=2)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    print("ConfusionMatrix", confusion_matrix(y, kmeans.labels_))
    print(classification_report(y, kmeans.labels_))