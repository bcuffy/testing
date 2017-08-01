import numpy as np 
import pandas as pd
import webbrowser
import os

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

address = 'C:/Users/629/Documents/test_scripts/python/ML/Orlieb/2017_Jan-July_Poles (A_C_E)_15min.csv'
df = pd.read_csv(address, index_col='Unix', date_parser=None )

"""from pandas import Series, DataFrame

missing = np.nan

df = pd.read_csv("2017_Jan-July_Poles (A_C_E)_15min.csv")
df = df.transpose()"""

html = df[0:6].to_html()
# Save the html to a temporary file
with open("data.html", "w") as f:
    f.write(html)

# Open the web page in our web browser
full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))


