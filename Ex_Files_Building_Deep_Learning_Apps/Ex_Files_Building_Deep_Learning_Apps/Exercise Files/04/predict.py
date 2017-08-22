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
model.add(Dense(50, activation="relu"))
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

# Load the data we make to use to make a prediction
X_predictions = pd.read_csv("2015_test_predict.csv")
X = X_predictions.values

# Make a prediction with the neural network
prediction = model.predict(X)
p = np.array(prediction)

# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 1.517543
prediction = prediction / 0.0765550239

p = p + 1.517543
p = p / 0.0765550239

x=[]
for i in range(0,19):
    x.append(i)

#np.savetxt("predict.csv", p, delimiter=",", header='Avg', comments='')

ndf = pd.read_csv('predict.csv')
#ndf.plot(y='Avg')
#plt.legend(loc='best')


#Load original test data
orig_test_df = pd.read_csv("2015_test.csv")
pframe = orig_test_df.head(n=19)

print(pframe['Avg'])
print(len(x))
print(len(ndf))
print(len(pframe))

plt.plot(x, ndf)
plt.plot(x, pframe['Avg'])

#pframe.plot(y='Avg')
plt.legend(loc='best')
plt.show()


#print("Earnings Prediction for Proposed Product - ${}".format(prediction))
