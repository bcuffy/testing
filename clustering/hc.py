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

cars = pd.read_csv('Iris/Exercise Files/Ch06/06_02/mtcars.csv')
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

#get column feature vectors for model, [mpg, disp, hp, wt]
X = cars.ix[:, (1,2,3,4,6)].values

#target value 'am' transmission type
y = cars.ix[:,10].values
temp_gear = list(map(lambda x:x-y.min(), y))
y = np.array(temp_gear)
kset = set(y)

#use ward linkage method to cluster data
Z = linkage(X,'ward')

#draw dendrogram
#use dendrogram to deterimne how many clusters to make
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)

"""
# Dendrogram titling
plt.title("truncated hierarchical dendogram")
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

# draw line @ y value that approximates the number of desired clusters
plt.axhline(y=250)
plt.show()
"""

#create clustering object
k = len(kset)

# affinity functions (distance metrics): 'manhattan', 'euclidean', 'cosine'
# linkage (linkage parameters): 'ward', 'complete', 'average'
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average') 
Hclustering.fit(X)

#score accuracy of predictions against actual values in y
print(sm.accuracy_score(y, Hclustering.labels_))

color_theme = np.array(['darkgray','#2E5266', 'lightsalmon', 'powderblue', '#A5C969', '#584371'])

# Build figures to plot
def plot(df, dfx, dfy, df_label, color_theme):
    ##### bokeh plotting
    output_file("lines.html")

    # map colors for plot
    mapper = LinearColorMapper(palette=color_theme, low=df_label.min(), high=df_label.max())

    source = ColumnDataSource(cars)

    # define tools
    TOOLS = ['hover,pan,box_zoom,reset,wheel_zoom']

    mpg = list(map(str,dfx))
    hp = list(map(str,dfy))

    # Define Plot Canvas
    p = figure(width=600, height=600, title="Hierarchical Clustering", x_axis_label='Miles per Gallon', 
                x_range=[0,40], y_range=[0,400],
                y_axis_label='Horse Power',  tools=TOOLS)

    # define what to draw on canvas
    p.circle(source=source, x=dfx, y=dfy, radius=.3, fill_color=color_theme[df_label], fill_alpha=1.0, line_color='black', line_alpha=.25)

    # Define HoverTool
    p.select_one(HoverTool).tooltips = [
        ('MPG', '@mpg'),
        ('Name', '@car_names'),
        ('Gears', '@gear')
    ]

    return p


# print(Hclustering.labels_)
p = row(plot(cars, cars.mpg, cars.hp, y, color_theme), plot(cars, cars.mpg, cars.hp, Hclustering.labels_, color_theme))
show(p)

# score precision of each cluster
print(classification_report(y, Hclustering.labels_))
