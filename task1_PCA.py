import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import seaborn

# Import data
df = pd.read_csv("data/Data.csv", encoding = "gbk", header = 0, index_col = 0).reset_index(drop = True)

# Delete outlier
df = df[~(df["Label"].isin([2]))]

# Select data
X = df.iloc[:, 0:15]
X = np.array(X)

y = df["Label"]

# PCA
pca = PCA()
t0 = time()
X1 = pca.fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ("PCA", t1 - t0))

# plt.plot(range(15), pca.explained_variance_ratio_)
# plt.plot(range(15), np.cumsum(pca.explained_variance_ratio_))
# plt.title("Component-wise and Cumulative Explained Variance")

# Cumulative sum plot
explained_variance = pca.explained_variance_ratio_
cumsum = np.cumsum(explained_variance)
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(cumsum, linewidth = 3)
plt.axhline(y=0.85, color='r', linestyle='--')
plt.yticks(np.arange(0.15, 1.05, 0.05), fontsize=15)
plt.xticks(np.arange(0, 15, 1), fontsize=15)
plt.xlabel("Dimensions", fontsize=20)
plt.ylabel("Explained Variance",fontsize=20)
plt.grid(False)
plt.savefig("fig/pca_cumsum.png")
plt.show()

# Individual explained variance plot
plt.figure(dpi=600)
plt.figure(figsize=(6, 4))
plt.bar(range(15), pca.explained_variance_ratio_, alpha=0.5, align='center',
        label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("fig/pca_individual_explained.png")
plt.show()


pca = PCA(n_components=10)
t0 = time()
X = pca.fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ("PCA", t1 - t0))

pc_df = pd.DataFrame(data = X,
                     columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
pc_df["Cluster"] = y

# PCA plot of first two PCs
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_style("white")
sns.lmplot( x="0", y="1",
  data=pc_df,
  fit_reg=False,
  hue='Cluster', # color by cluster
  legend=True,
  scatter_kws={"s": 10}).fig.suptitle("%s (%.2g sec)" % ("Time", t1 - t0)) # specify the point size
plt.savefig("fig/pca.png")
plt.show()

# Density plot
class1 = pc_df.loc[pc_df["Cluster"] == 0]
class2 = pc_df.loc[pc_df["Cluster"] == 1]
plt.figure(figsize=(16, 10), dpi=80)
seaborn.kdeplot(x="0", y="1", data=class1, fill=True, color="red", label="Red: Cluster=0", alpha=.7)
seaborn.kdeplot(x='0', y='1', data=class2, fill=True, color="blue", label="Green: Cluster=1", alpha=.7)
# plt.title("Density Plot", fontsize=32)
plt.legend(fontsize=25)
plt.xlabel("0", fontsize=25)
plt.ylabel("1", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig("fig/pca_density.png")
plt.show()
