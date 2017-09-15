import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb
from matplotlib import style
style.use("seaborn-dark")

import sklearn
from sklearn import datasets

from sklearn.cluster import DBSCAN
from collections import Counter

rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

df = pd.read_csv(
    filepath_or_buffer='iris.data.csv',
    header=None,
    sep=',')

df.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width', 'Species']
data = df.ix[:,0:4].values
target = df.ix[:, 4].values

#min_samples: number of samples to be considered as core point
model = DBSCAN(eps=0.8, min_samples=19).fit(data)
outliers_df = pd.DataFrame(data)

print(Counter(model.labels_))
print(outliers_df[model.labels_==-1])

fig = plt.figure()
#ax = fig.add_axes([.1, .1, 1, 1])

colors = model.labels_
color_theme = np.array(['darkgray', 'lightsalmon', 'blue'])

ax = plt.gca()
ax.axis([0, 7, 0,3])

plt.subplot(1,2,1)
plt.xlim([0,7])
plt.ylim([0,3])
plt.scatter(data[:,2], data[:,3], c = 'b', edgecolors=None, linewidths=.5, alpha=.9, s=50)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Dataset: Petal Length v. Petal Length")

plt.subplot(1,2,2)
plt.xlim([0,7])
plt.ylim([0,3])
plt.scatter(data[:,2], data[:,3], c = color_theme[colors], edgecolors=None, linewidths=.5, alpha=.9, s=50)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("DBScan for outlier detection")

plt.show()