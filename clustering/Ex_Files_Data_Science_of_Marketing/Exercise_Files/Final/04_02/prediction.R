# Connect to our data
myPredictionData <- read.csv('~/Desktop/Exercise_Files/04_02/prediction-r.csv')

# Sum our classifications so we can see them 
table(myPredictionData$sales.classification)

# Output our column names for easy reference 
names(myPredictionData)

# Install our algorithm (the tree package)
install.packages('tree')

# Bring our newly installed algorithm into our library of packages
library(tree)

# Configure our algorithm to create our tree
myDecisionTree <- tree(sales.classification ~ capita + drive.by.traffic + complimentary.establishments + competition + weather + unemployment.rate + var1 + var2 + var3, data=myPredictionData)

# Plot our tree so we can see the algorithms output
plot(myDecisionTree)

# Label our tree
text(myDecisionTree)

# Prune our tree
myPrunedTree <- prune.tree(myDecisionTree,best=3)

# Plot our pruned tree
plot(myPrunedTree)

# Label our pruned tree 
text(myPrunedTree)
