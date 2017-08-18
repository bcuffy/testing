import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
#style.use("seaborn-dark")

import seaborn as sb

import sklearn
from sklearn.cluster import KMeans

from sklearn import preprocessing
import sklearn.metrics as sm

df = pd.read_csv("2015_June_cluster2.csv")
df.drop(['DayM','Mo','DayW'], 1, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
    
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x =0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))

        return df

df = handle_non_numerical_data(df)

X = np.array(df.drop(['Time'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Time'])
"""
clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct+=1

color_theme = np.array( ['#FCEFA1','#FBAE5E','#EE6C5E','#CB5053','#0A3C55','#E0E0E2','#81D2C7','#606EC7','#4A5759','#EDAFB8','#389A9E','#F7E1D7','#D33E43','#F3A712','#5B8AA2',
     '#606C38','#645771','r','g','b','c','m','y','k'])
"""

#print(len(clf.labels_))
# plt.scatter(x=df.OT,y=df.Avg_S1, c=color_theme[clf.labels_], s=50)
# plt.title("Time Classification")
# plt.show()

cols = df.drop(['Time'],1).columns

sb.pairplot(df, x_vars=cols, y_vars=cols, hue=[4], palette='coolwarm')
plt.show()
#print(correct/len(X))
