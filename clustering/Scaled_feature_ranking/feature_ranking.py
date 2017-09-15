import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import math
import matplotlib.pyplot as plt

import numpy as np


def featureRank():
    # These are the feature labels from our data set
    # Feature labels in file 'c_time','s_x','s_y','s_z','Solar Alt','Solar Azimuth','T_in','SP2_1','SP2_10','SP2_11','SP2_2','SP2_3','SP2_4','SP2_5','SP2_6','SP2_7','SP2_8','SP2_9'
    var = "sensor"
    feature_labels = np.array([var,'c_time', 's_x', 's_y', 's_z', 'Solar Alt', 'Solar Azimuth', 'T_in', 'SP2_1', 'SP2_10', 'SP2_11', 'SP2_2', 'SP2_3', 'SP2_4', 'SP2_5', 'SP2_6', 'SP2_7', 'SP2_9'])

    print(feature_labels)
    # Load the trained model created with train_model.py
    model = joblib.load('CITA_trained_model.pkl')

    # Create a numpy array based on the model's feature importances
    importance = model.feature_importances_

    # Sort the feature labels based on the feature importance rankings from the model
    feauture_indexes_by_importance = importance.argsort()

    # Print each feature label, from most important to least important (reverse order)
    for index in reversed(feauture_indexes_by_importance):
        if index < 13:
            print("{} - {:.2f}%".format(feature_labels[index], (importance[index] * 100.0)))

def createModel():
    # Load the data set into DataFrame object
    df = pd.read_csv("../CITA_ML_Ext_Cond_Imputed.csv")

    # Remove the fields from the data set that we don't want to include in our model
    # Replace categorical data with one-hot encoded data
    #features_df = pd.get_dummies(df, columns=['Day_of_Week', "Month"])
    features_df = df.copy(deep=True)
    del features_df['Yr']
    del features_df['Mo']
    del features_df['Day']
    del features_df['d_time']
    del features_df['o_time']

    # Remove prediction targets
    del features_df['SP2_8']

    # Create the X and y arrays
    X = features_df.as_matrix()     #Standard name for input features
    Y = df['SP2_8']                   #Standard name for output

    # Split the data set in a training set (70%) and a test set (30%)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size = 0.3)

    #Construct Regression Model
    #todo investigate hyper-parameter derevation
    model = ensemble.GradientBoostingRegressor(
        n_estimators=500,      #How many decision trees to build
        learning_rate = 0.1,    #How much each additional tree influences the overall prediction
        max_depth = 2,
        min_samples_leaf = 9,   #how many times a value must appear for a decision tree to make a decision base on input
        max_features = 0.9,     #Max feature to consider when making new tree
        loss = "huber",         #how model calculates error rate as it learns
        random_state=4
    )

    # Fit regression model
    model.fit(X_TRAIN, Y_TRAIN)

    # Save the trained model to a file so we can use it in other programs
    joblib.dump(model, 'CITA_trained_model.pkl')

    # Track error with mean sqaured error

    # Find the error rate on the training set
    mse = mean_squared_error(Y_TRAIN, model.predict(X_TRAIN))
    print("Training Set Mean Error: %.4f" % mse)

    # Find the error rate on the test set
    mse = mean_squared_error(Y_TEST, model.predict(X_TEST))
    print("Test Set Mean Error: %.4f" % mse)

    featureRank()


if __name__ == "__main__":
    createModel()