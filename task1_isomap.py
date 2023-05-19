import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold, datasets
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

# Isomap
tsne = manifold.Isomap(n_neighbors=10, n_components=2)
t0 = time()
X_isomap = tsne.fit_transform(X)
t1 = time()
print("%s: %.2g sec" % ("Isomap", t1 - t0))

X_df = pd.DataFrame(data=X_isomap,
                     columns=["0", "1"],)
X_df["Cluster"] = y

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_style("white")
sns.lmplot(x="0", y="1",
  data=X_df,
  fit_reg=False,
  hue="Cluster", # color by cluster
  legend=True,
  scatter_kws={"s": 10}).fig.suptitle("%s (%.2g sec)" % ("Time", t1 - t0)) # specify the point size
plt.savefig("fig/isomap.png")
plt.show()

# Density plot
class1 = X_df.loc[X_df["Cluster"] == 0]
class2 = X_df.loc[X_df["Cluster"] == 1]
plt.figure(figsize=(16, 10), dpi=150)
seaborn.kdeplot(x="0", y="1", data=class1, fill=True, color="red", label="Red: Cluster=0", alpha=.7)
seaborn.kdeplot(x="0", y="1", data=class2, fill=True, color="blue", label="Blue: Cluster=1", alpha=.7)
plt.title("Density Plot")
plt.legend()
plt.savefig("fig/isomap_density.png")
plt.show()