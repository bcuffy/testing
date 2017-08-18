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

df = pd.read_csv('auto-mpg2.csv')
#df = df.rename(columns={df.columns[0]: 'name'})

df.drop(['name'], 1, inplace=True)
df.drop(['class'], 1, inplace=True)
df.drop(['year'], 1, inplace=True)

# create feature, X, and target, y, sets. Format as np array
feature_data =df.drop(['am'], axis=1)

feature_data =feature_data.ix[:,0:5]
target_data = df.ix[:,6].values

# Create DBSCAN model
model = DBSCAN(eps=.2, min_samples=19).fit(feature_data)
outliers_df = pd.DataFrame(feature_data)

print(Counter(model.labels_))
print(outliers_df[model.labels_==-1])

idf = pd.read_csv(
    filepath_or_buffer='iris.data.csv',
    header=None,
    sep=',')

idf.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width', 'Species']
data = df.ix[:,0:4].values
target = df.ix[:,4].values

#min_samples: number of samples to be considered as core point
model = DBSCAN(eps=0.8, min_samples=19).fit(data)
outliers_df = pd.DataFrame(data)

print(Counter(model.labels_))
print(outliers_df[model.labels_==-1])

fig = plt.figure()
ax = fig.add_axes([.1, .1, 1, 1])

colors = model.labels_

ax = plt.gca()
ax.axis([0, 10, 0,450])

print(data)

# ax.scatter(feature_data[:,2], feature_data[:,1], c = colors, s=120)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# plt.title("DBScan for outlier detection")
# plt.show()




# for s in range(0,len(idf['Species'])):
   
#     if idf['Species'][s] == "setosa":
#          idf['Species'][s]=0
#     elif idf['Species'][s] == "versicolor":
#         idf['Species'][s]=1
#     else:
#         idf['Species'][s]=2


# print(list(idf['Species']))


#sb.pairplot(idf, x_vars=idf.columns, y_vars=idf.columns, hue='Species', palette='coolwarm')
#plt.show()

#S

#target_df = df['Cyclical_Time'].copy()
#target_df = df.loc[:,'Time']
#target_df = target_df.reset_index()

# del target_df['index']

# y = target_df
# del df['Time']

# dataset = df.as_matrix()
# target_set = target_df['Time'].as_matrix()


   
#print(df.head(5)) 