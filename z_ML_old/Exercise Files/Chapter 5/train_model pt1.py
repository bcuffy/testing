import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# Load the data set
df = pd.read_csv("ml_house_data_set.csv")

# Remove the fields from the data set that we don't want to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
features_df = pd.get_dumies(df, columns=['grage_type', 'city'])

# Remove the sale price from the feature data
del features_df['sales_price']

#Features Data Frame (.as_matrix ensures data type is numpy Matrix)
X = features_df.as_matrix()
y = df['sales_price'].as_matrix()

# Create the X and y arrays
X =
y =