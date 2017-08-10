import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("seaborn-dark")

import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report

#%matplotlib inline
plt.figure(figsize=(7,4))

df = pd.read_csv("2015_June_cluster.csv")

#target_df = df['Cyclical_Time'].copy()
target_df = df.loc[:,'Cyclical_Time']
target_df = target_df.reset_index()

del target_df['index']

y = target_df
del df['Cyclical_Time']

dataset = df.as_matrix()
target_set = target_df['Cyclical_Time'].as_matrix()
X = scale(dataset)




iris = datasets.load_iris()

variable_names = list(df.columns)

clustering = KMeans(n_clusters=2, random_state=5)
clustering.fit(X)

pdf = pd.DataFrame(dataset)
pdf.columns = df.columns
y.columns = ["Targets"]



color_theme = np.array(['r', 'c', 'b'])
plt.subplot(1, 4, 1)

plt.scatter(x=pdf.Avg_S3,y=pdf.Avg_S1, c=color_theme[target_set], s=50)
plt.title("Time Classification")

relabel = np.choose(clustering.labels_, [0, 1, 2]).astype(np.int64)

plt.subplot(1, 4, 2)
plt.scatter(x=pdf.Avg_S3,y=pdf.Avg_S1, c=color_theme[clustering.labels_], s=50)
plt.title("5 v 10")

plt.subplot(1, 4, 3)
plt.scatter(x=pdf.Avg_S3,y=pdf.Avg_S4, c=color_theme[clustering.labels_], s=50)
plt.title("5 v 16")

plt.subplot(1, 4, 4)
plt.scatter(x=pdf.Avg_S3,y=pdf.Avg_S5, c=color_theme[clustering.labels_], s=50)
plt.title("E5 v A5")

plt.show()

print(classification_report(y, relabel))






