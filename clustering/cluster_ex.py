from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot(dataset, history_centroids, belongs_to, color):
    colors = color

    fig, ax = plt.subplots()

    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    history_points = []
    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

                plt.pause(0.8)

def euclidian(a, b):
    return np.linalg.norm(a-b)

    #k = number of clusters
    #epsilon = minimum error to use in stop condition
def kmeans(k, epsilon=0, distance='euclidian'):
    #list to store past centroids for visualization
    history_centroids = []
    #set distance calculation type
    if distance == 'euclidian':
        dist_method = euclidian
    
    #load Data
    df = pd.read_csv("2015_June-Oct_Poles (A_C_E)_15min_imputed.csv")
    dataset = df.as_matrix()

    # dataset = dataset[:, 0:dataset.shape[1] - 1]
    num_instances, num_features = dataset.shape
    #define k centroids (random points to initate clustering)
    #number of centroids equals number of clusters
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    #set centroids to histroy centroid list to keep over time
    history_centroids.append(prototypes)
    #intialize list to hold previous set of centroid
    prototypes_old = np.zeros(prototypes.shape)
    #store clusters overtime
    belongs_to = np.zeros((num_instances, 1))
    #calculate distance between current centroids, prototypes
    #and previous centroids, prototypes_old, and store in norm 
    norm = dist_method(prototypes, prototypes_old)
    #track number of iterations
    iteration = 0

    while norm > epsilon:
        iteration += 1
        #compute norm
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        #for each instance in dataset
        for index_instance, instance in enumerate(dataset):
            #define distance vector of size k
            dist_vec = np.zeros((k, 1))
            # for each centroid
            for index_prototype, prototype in enumerate(prototypes):
                # compute distance between x and centroid
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)
            #find the smallest distance, assignm that distance to a cluster
            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        # for each cluster, k of them
        for index in range(len(prototypes)):
            # get all points assigned to a cluster
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            # find the mean of those points, this is our new centroid
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            # add our new centorid to our new temporary list
            tmp_prototypes[index, :] = prototype
        # set the new list to the current list
        prototypes = tmp_prototypes
        # add our calcualted centroids to our history for plotting
        history_centroids.append(tmp_prototypes)

    # plot(dataset, history_centroids, belongs_to)
    #return calculated centroids, history of them all, and assignments for clusters
    return prototypes, history_centroids, belongs_to

def execute():
    #load dataset
    df = pd.read_csv("2015_June-Oct_Poles (A_C_E)_15min_imputed.csv")
    dataset = df.as_matrix()
    #train the model on the data

    colors = ['r', 'm']
    centroids, history_centroids, belongs_to = kmeans(len(colors))
    #plot results
    plot(dataset, history_centroids, belongs_to, colors)

execute()
