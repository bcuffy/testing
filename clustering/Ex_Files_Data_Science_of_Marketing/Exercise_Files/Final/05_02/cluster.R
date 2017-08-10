# Cluster Analysis 

# Connect to our case study data
myClusterData <- read.csv("~/Desktop/Exercise_Files/05_02/cluster-r.csv")

# Review our data
head(myClusterData)

# Standardize the data
myClusterDataStandardized <- scale(myClusterData[-1])

# Run kmeans on our standardized data
ourGroups <- kmeans(myClusterDataStandardized, 3)

# Load in our cluster library 
library(cluster)

# Visualize our clusters
clusplot(myClusterDataStandardized, ourGroups$cluster)

# Summarize our data
ourGroups$size
