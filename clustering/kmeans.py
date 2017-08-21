import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("seaborn-dark")

import seaborn as sb
sb.set_style('whitegrid')

import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report

#clustering is important for kmeans

plt.figure(figsize=(7,4))

iris = datasets.load_iris()

X = scale(iris.data) #has no header
y = pd.DataFrame(iris.target)

variable_names = iris.feature_names

clustering = KMeans(n_clusters=3, random_state=5)

clustering.fit(X)

iris_df = pd.DataFrame(iris.data)
iris_df.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y.columns = ['Target']
ax = plt.gca()
ax.axis([0, 7, 0,3])

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length, y =iris_df.Petal_Width, c=color_theme[iris.target], s=50, linewidths=.55, edgecolors='gray', facecolor=None, hatch=None)
plt.xlim([0,8])
plt.ylim([0,3])
plt.title('Ground Truth Classification')

relabel = np.choose(clustering.labels_, [2,0,1]).astype(np.int64)

plt.subplot(1,2,2)
plt.xlim([0,8])
plt.ylim([0,3])
plt.scatter(x=iris_df.Petal_Length, y =iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50, edgecolors='gray', linewidths=1, facecolor=None, hatch=None)
plt.title('KMeans classification')

plt.show()

print(classification_report(y, relabel))