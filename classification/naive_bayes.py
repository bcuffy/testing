import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from pylab import rcParams

import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

plt.style.use('seaborn-whitegrid')


df = pd.read_csv("CITA_ML_Ext_Cond_Imputed_stacked.csv")
feature_df = df.drop(['Solar Azimuth', 'Solar Alt', 'c_time','o_time','d_time','Yr','Mo','Day'], 1)

X = feature_df.ix[:,0:9]
y = feature_df.ix[:,9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=17)

# BernNB = BernoulliNB(binarize=True)
# BernNB.fit(X_train, y_train)
# print(BernNB)

y_expect = y_test
# y_pred = y_testy_pred = BernNB.predict(X_test)
# print(accuracy_score(y_expect, y_pred))

GausNB = GaussianNB()
GausNB.fit(X_train,y_train)
print(GausNB)

y_pred = GausNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))