import numpy as np 
import pandas as pd
import webbrowser
import os

from pandas import Series, DataFrame

missing = np.nan

df = pd.read_csv("2017_Jan-July_Poles (A_C_E)_15min.csv")

"""series_obj = Series(['row 1','row 2', missing, 'row 4','row 5', 'row 6', missing, 'row 8'])

print(series_obj.isnull())

np.random.seed(25)
DF_obj = DataFrame(np.random.randn(36).reshape(6,6))

#set missing values for test
DF_obj.ix[3:5,0] = missing
DF_obj.ix[1:4, 5] = missing

print(DF_obj.isnull)"""

#find and fill missing value with 0
#filled_DF = DF_obj.fillna()

#fill with dictionary
#filled_DF = DF_obj.fillna({0: 0.1, 5: 1.25})

#fill with dictionary
#filled_DF = DF_obj.fillna(method='ffill')

#Count missing values
#DF_obj.isnull().sum()

html = filled_DF[0:6].to_html()
# Save the html to a temporary file
with open("data.html", "w") as f:
    f.write(html)

# Open the web page in our web browser
full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))




