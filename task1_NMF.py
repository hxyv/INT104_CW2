import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
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

# NMF
nmf = NMF(n_components=10, init='random', random_state=0)
t0 = time()
X_nmf = nmf.fit_transform(X)
t1 = time()
H = nmf.components_

X_df = pd.DataFrame(data=X_nmf,
                     columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],)
X_df["Cluster"] = y

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_style("white")
sns.lmplot(x="3", y="4",
  data=X_df,
  fit_reg=False,
  hue="Cluster", # color by cluster
  legend=True,
  scatter_kws={"s": 10}).fig.suptitle("%s (%.2g sec)" % ("Time", t1 - t0)) # specify the point size
plt.savefig("fig/nmf.png")
plt.show()

err = []
for i in range(2, 14):
  nmf = NMF(n_components=i, init="random", random_state=0)
  W = nmf.fit_transform(X)
  H = nmf.components_
  err.append(np.linalg.norm(X - W @ H)**2/np.linalg.norm(X)**2)
err_df = {"err": err, "y": range(2, 14)}
err_df = pd.DataFrame(err_df)
plt.figure(figsize=(6, 7), dpi=200)
plt.scatter(x="y", y="err", data=err_df)
plt.xlabel("n_component", fontsize=18)
plt.ylabel("precision score",fontsize=18)
plt.savefig("fig/nmf_err.png")
plt.show()


# Density plot
class1 = X_df.loc[X_df["Cluster"] == 0]
class2 = X_df.loc[X_df["Cluster"] == 1]
plt.figure(figsize=(16, 10), dpi=80)
seaborn.kdeplot(x="0", y="1", data=class1, fill=True, color="red", label="Red: Cluster=0", alpha=.7)
seaborn.kdeplot(x="0", y="1", data=class2, fill=True, color="blue", label="Blue: Cluster=1", alpha=.7)
plt.title("Density Plot")
plt.legend()
plt.savefig("fig/nmf_density.png")
plt.show()

# for i in range(1, 9):
#     sns.lmplot(x=str(i), y=str(i+1),
#                data=X_df,
#                fit_reg=False,
#                hue="Cluster",  # color by cluster
#                legend=True,
#                scatter_kws={"s": 10}).fig.suptitle(
#         "%s (%.2g sec)" % ("NMF", t1 - t0))  # specify the point size
#     plt.show()