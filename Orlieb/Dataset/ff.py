import numpy as np 
import webbrowser
import os
import pandas as pd

#create data frame from imported dataset
df = pd.read_csv("2017_Jan-July_Poles (A_C_E)_15min.csv")

#find cells with missing data and fill forward with most recent previous values
fill_df = df.fillna(method='ffill')

#Write filled data fram to new csv file
fill_df.to_csv('filled_data.csv',',')

#preview data frame with pandas. This previews first 1000 rows
html = fill_df[0:1000].to_html()
with open('data.html', 'w') as f:
    f.write(html)

full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))
