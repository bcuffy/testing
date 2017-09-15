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

df = pd.read_csv('CITA_ML_Ext_Cond_Imputed.csv')
feature_df = df.drop(['o_time','d_time','Yr','Mo','Day'], 1)


data = feature_df.ix[:,0:7].values
target = feature_df.ix[:, 6].values

# Fit data to dbscan model
# Specify min_samples: number of samples to be considered as core point
# Specify eps: distance between p and q to be considered core sample
model = DBSCAN(eps=11, min_samples=50).fit(data)
outliers_df = pd.DataFrame(data)

print(Counter(model.labels_))
# print(outliers_df[model.labels_==-1])

fig = plt.figure()

colors = model.labels_
color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

# feature_df = df.drop(['SP2_10','SP2_11','SP2_2','SP2_3','SP2_4','SP2_5','SP2_6','SP2_7','SP2_8','SP2_9'], 1)
feature_df['k'] = colors
# sb.pairplot(feature_df, plot_kws=dict(alpha=.6, linewidth=0), hue='k', palette=sb.color_palette(color_theme))


# ax = plt.gca()
# ax.axis([-2,2, 0, 40])

# ax.scatter(feature_df['c_time'], feature_df['SP2_7'], c = color_theme[colors], edgecolors='black', linewidths=.5, alpha=.9, s=50)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# plt.title("DBScan for outlier detection")
# plt.show()

index_time = np.arange(0,len(feature_df))
feature_df['index_time'] = index_time
sb.pairplot(feature_df, vars=['index_time', 'T_in', 'Solar Azimuth', 'Solar Alt','SP2_1', 'SP2_4', 'SP2_7', 'SP2_10', 'SP2_9'], plot_kws=dict(alpha=.9, linewidth=0, s=5), hue='k', palette=color_theme)
plt.show()

plt.subplot(3,1,1)
plt.xlim([0,len(index_time)])
plt.ylim([0,40])
plt.scatter(index_time, feature_df['SP2_1'], c = color_theme[colors], edgecolors='black', linewidths=.1, alpha=.8, s=50)
plt.xlabel('Index Time')
plt.ylabel('Sensor 1 Temp')


plt.subplot(3,1,2)
plt.xlim([0,len(index_time)])
plt.ylim([0,40])
plt.scatter(index_time, feature_df['SP2_4'], c = color_theme[colors], edgecolors='black', linewidths=.1, alpha=.8, s=50)
plt.xlabel('Index Time')
plt.ylabel('Sensor 4 Temp')


plt.subplot(3,1,3)
plt.xlim([0,len(index_time)])
plt.ylim([0,40])
plt.scatter(index_time, feature_df['SP2_7'], c = color_theme[colors], edgecolors='black', linewidths=.1, alpha=.8, s=50)
plt.xlabel('Index Time')
plt.ylabel('Sensor 7 Temp')


# plt.show()