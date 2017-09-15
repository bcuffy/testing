import numpy as np
import pandas as pd

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource, LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar
from bokeh.charts import Scatter

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.cluster import AgglomerativeClustering #same as hierarchical clustering
import sklearn.metrics as sm 
from sklearn.metrics import confusion_matrix, classification_report

#set number precision
np.set_printoptions(precision=4, suppress=2)

#set plot parameters
plt.figure(figsize=(10, 3))
plt.style.use('seaborn-whitegrid')


def map_text_to_index(output_list):
    index = 0
    l = []
    target_set = {}
    for item in y:
        if(item in target_set):
            continue
        else:
            target_set[item] = index
            index = index + 1

    for item in y:
        # print(target_set[item])
        l.append(target_set[item])

    return np.array(l), len(target_set)


cita_df = pd.read_csv('cita.csv')
feature_df = cita_df.copy(deep=True)

#get column feature vectors for model, [mpg, disp, hp, wt]
X = cita_df.ix[:,0:2].values

#use ward linkage method to cluster data
color_theme = ['darkgray', 'lightsalmon', 'powderblue', 'powderblue', '#A5C969', '#584371']
Z = linkage(X,'ward')

c, coph_dists = cophenet(Z, pdist(X))

print(c)

#draw dendrogram
#use dendrogram to deterimne how many clusters to make
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=0., leaf_font_size=15, show_contracted=True)


# Dendrogram titling
# plt.title("truncated hierarchical dendogram")
# plt.xlabel('Cluster Size')
# plt.ylabel('Distance')

# # draw line @ y value that approximates the number of desired clusters
# plt.axhline(y=10)
# plt.show()


# create list of target values and set of target values for for cluster assignments
y = cita_df.ix[:,2].values
y, k = map_text_to_index(y)
print(k)

# affinity functions (distance metrics): 'manhattan', 'euclidean', 'cosine'
# linkage (linkage parameters): 'ward', 'complete', 'average'
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average') 
Hclustering.fit(X)
 
print(Hclustering.labels_)
print(y)

#relabel groups to match indeices to target set.
#This process need to occur if the algorithm is clustering the data correctly
#but mislabeling the cluster index

# relabel = np.choose(Hclustering.labels_, [1,0,2]).astype(np.int64)
# Hclustering.labels_ = relabel

#score accuracy of predictions against actual values in y
print(sm.accuracy_score(y, Hclustering.labels_))

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue', "#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"])

# Build figures to plot
def plot(df, dfx, dfy, df_label, color_theme):
    ##### bokeh plotting
    output_file("lines.html")

    # map colors for plot
    mapper = LinearColorMapper(palette=color_theme, low=df_label.min(), high=df_label.max())

    source = ColumnDataSource(df)

    # define tools
    TOOLS = ['hover,pan,box_zoom,reset,wheel_zoom']

    # Define Plot Canvas
    p = figure(width=600, height=600, title="Hierarchical Clustering", x_axis_label=str(dfx), 
                x_range=[0,7], y_range=[0,3],
                y_axis_label=str(dfy),  tools=TOOLS)

    # define what to draw on canvas
    p.circle(source=source, x=df[dfx], y=df[dfy], radius=.05, fill_color=color_theme[df_label], fill_alpha=.9, line_color='black', line_alpha=.25)

    # Define HoverTool
    p.select_one(HoverTool).tooltips = [
        ('W', '@Sepal_Width'),
        ('L', '@Sepal_Length')
    ]

    return p


p = row(plot(cita_df, 'Temp', 'Sensor', y, color_theme), plot(cita_df, 'Temp', 'Sensor', Hclustering.labels_, color_theme))
show(p)

# score precision of each cluster
print(classification_report(y, Hclustering.labels_))
