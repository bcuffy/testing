import pandas as pd
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt

training_data_df = pd.read_csv("2015_training_scaled.csv")

X = training_data_df.drop('Avg', axis=1).values
Y = training_data_df[['Avg']].values



# Define the model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(100, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error", optimizer='adam')


# Train the model
#training features and expected output
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load the separate test data set
test_data_df = pd.read_csv("2015_test_scaled.csv")

X_test = test_data_df.drop('Avg', axis=1).values
Y_test = test_data_df[['Avg']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
#print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
