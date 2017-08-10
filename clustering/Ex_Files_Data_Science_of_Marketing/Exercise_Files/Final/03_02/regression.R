# Connect to our data
myRegressionData <- read.csv("~/Desktop/Exercise_Files/03_02/regression-r.csv")

# Plot our data (broadcast & sales)
plot(myRegressionData$BROADCAST,myRegressionData$NET.SALES)

# Fit a line
myLm <- lm(myRegressionData$NET.SALES ~ myRegressionData$BROADCAST)

# Visualize the line
lines(myRegressionData$BROADCAST,myLm$fitted)

# Show our coefficients
myLm$coeff 
