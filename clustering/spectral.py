import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("seaborn-dark")

import seaborn as sb
sb.set_style('whitegrid')

import sklearn
from sklearn import datasets
from sklearn.cluster import SpectralClustering

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report

#scalinging is important for kmeans

plt.figure(figsize=(17,4))

cita_df = pd.read_csv('cita.csv')
feature_df = cita_df.copy(deep=True)
#feature_df = feature_df.drop('Sensor',1)

X = scale(feature_df.values) #has no header
# y = score_df['Sensor']

variable_names = feature_df.columns

clustering = SpectralClustering(n_clusters=11, affinity='rbf', n_init=10)
clustering.fit(X)

ax = plt.gca()
ax.axis([0, 7, 0,3])

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue', "#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"])

plt.subplot(1,2,1)
# plt.scatter(x=cita_df.Temp, y=cita_df.Sensor, c=color_theme[cita_df['Sensor']], s=50, linewidths=.55, edgecolors='gray', facecolor=None, hatch=None)
plt.xlim([0,35])
plt.ylim([0,35])
plt.title('Ground Truth Classification')

# relabel = np.choose(clustering.labels_, [3,6,3,]).astype(np.int64)

plt.subplot(1,2,2)
plt.xlim([0,35])
plt.ylim([0,35])
plt.scatter(x=cita_df.Temp, y =cita_df.Sensor, c=color_theme[clustering.labels_], s=50, edgecolors='gray', linewidths=1, facecolor=None, hatch=None)
plt.title('KMeans classification')

plt.show()
# l = zip(np.array(y), clustering.labels_)

from collections import Counter

# print(classification_report(np.array(y), clustering.labels_))