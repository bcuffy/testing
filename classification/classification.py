import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from pylab import rcParams

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics

plt.style.use('seaborn-whitegrid')


df = pd.read_csv("CITA_ML_Ext_Cond_Imputed_stacked.csv")
feature_df = df.drop(['Solar Azimuth', 'Solar Alt', 'c_time','o_time','d_time','Yr','Mo','Day'], 1)

X_prime = feature_df.ix[:,0:9].values
y = feature_df.ix[:,9].values

X = preprocessing.scale(X_prime)

X_train, X_test, y_train, y_expect = train_test_split(X,y, test_size=.33, random_state=17)

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)

y_predict = clf.predict(X_test)

print(metrics.classification_report(y_expect, y_predict))

plt.subplot(2,1,1)

