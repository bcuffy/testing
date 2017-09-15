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
#from bokeh.charts import Scatter

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
# plt.figure(figsize=(10, 3))
plt.style.use('seaborn-whitegrid')


df = pd.read_csv("CITA_ML_Ext_Cond_ImputedS.csv")
feature_df = df.drop(['o_time','d_time','Yr','Mo','Day'], 1)

# print(feature_df.ix[:,0:8])
X = feature_df.ix[:,0:8]

# ward - calculates distance between clusters and creates cluster tree
Z = linkage(X, 'average')

c, coph_dists = cophenet(Z, pdist(X))
print(c)

# plot dendrogram
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

#use dendrogram to deterimne how many clusters to make

fancy_dendrogram(
           Z, 
           truncate_mode='lastp',       # show last p merged clusters
           p=12,                        # show last p merged clusters
           leaf_rotation=0.,            # rotate text labels
           leaf_font_size=15,           # label size
           show_contracted=True,         # show height of merged clusters as dots
           annotate_above=200
        #    max_d = max_d
       )
# plt.show()


# use 'max_d' or 'k' to find clusters
# max_d if you want to cluster by max distance - criterion='distance'
# k if you want to cluster manually - criterion='maxclust'
k=3
max_d = 30.4
clusters = fcluster(Z, k, criterion='maxclust')

df['k'] = clusters

# matplotlib chart
# plt.plot(feature_df.ix[:,7:15])

# plt.scatter(feature_df['Solar Alt'],feature_df.ix[:,7:15], c = 'r', cmap='prism')
# plt.show()

# bokeh chart
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
    p = figure(width=1200, height=900, title="Hierarchical Clustering", x_axis_label=str(dfx), 
                x_range=[-80,80], y_range=[0,40],
                y_axis_label=str(dfy),  tools=TOOLS)

    # define what to draw on canvas
    p.circle(source=source, x=df[dfx], y=df[dfy], radius=.1, fill_color=color_theme[df_label], fill_alpha=.5, line_color='black', line_alpha=.25)

    # Define HoverTool
    p.select_one(HoverTool).tooltips = [
        ('Sensor', '@S')
    ]

    return p

color_theme = np.array(["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", 'darkgray', 'lightsalmon', 'powderblue', "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"])
color_theme_a = np.array([  "#933b41", 'lightsalmon', 'powderblue',"#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", 'darkgray', "#dfccce", "#ddb7b1", "#cc7878","#550b1d"])
# p = row(plot(df, 'STemp', 'T_in', clusters, color_theme))#, plot(df, 'Solar Alt', 'SP2_2', 'r', color_theme))
# show(p)


feature_df['k'] = clusters
print(list(clusters))
# sb.pairplot(feature_df, vars=['c_time','STemp', 'T_in', 'Solar Azimuth', 'Solar Alt'], plot_kws=dict(alpha=.3, linewidth=0, s=2), hue='S', palette=sb.color_palette(sb.color_palette(),11))
# plt.show()


aph = .6

# plt.subplot(3,3,1)
# plt.xlim([-2,2])
# plt.ylim([0,40])
# plt.scatter(feature_df.loc[feature_df['k'] == 1,'c_time'],feature_df.loc[feature_df['k'] == 1,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 1,'k']],alpha=aph)
# plt.title("Cluster 1")

# plt.subplot(3,3,2)
# plt.xlim([-2,2])
# plt.ylim([0,40])
# plt.scatter(feature_df.loc[feature_df['k'] == 2,'c_time'],feature_df.loc[feature_df['k'] == 2,'STemp'],color=color_theme_a[feature_df.loc[feature_df['S'] == 1,'k']], alpha=aph)

# plt.title("Cluster 2")

# plt.subplot(3,3,3)
# plt.xlim([-2,2])
# plt.ylim([0,40])
# plt.scatter(feature_df.loc[feature_df['k'] == 3,'c_time'],feature_df.loc[feature_df['k'] == 3,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 1,'k']], alpha=aph)
# plt.title("Cluster 3")


plt.subplot(2,3,1)
plt.xlim([-2,2])
plt.ylim([0,40])
plt.scatter(feature_df.loc[feature_df['S'] == 1,'c_time'],feature_df.loc[feature_df['S'] == 1,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 1,'k']], alpha=aph)
plt.title("Cylical Temp Sensor 1")
plt.xlabel('Cyclical Time')
plt.ylabel('Sensor 1 Temp')

plt.subplot(2,3,2)
plt.xlim([-2,2])
plt.ylim([0,40])
plt.scatter(feature_df.loc[feature_df['S'] == 4,'c_time'],feature_df.loc[feature_df['S'] == 4,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 4,'k']], alpha=aph)
plt.title("Cylical Temp Sensor 4")
plt.xlabel('Cyclical Time')
plt.ylabel('Sensor 4 Temp')


plt.subplot(2,3,3)
plt.xlim([-2,2])
plt.ylim([0,40])
plt.scatter(feature_df.loc[feature_df['S'] == 7,'c_time'],feature_df.loc[feature_df['S'] == 7,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 7,'k']], alpha=aph)
plt.title("Cylical Temp Sensor 7")
plt.xlabel('Cyclical Time')
plt.ylabel('Sensor 7 Temp')

index_time = np.arange(0,len(feature_df.loc[feature_df['S'] == 1,'STemp']))

# Observed Values
plt.subplot(2,3,4)
plt.xlim([0,index_time[-1:]])
plt.ylim([0,40])
plt.scatter(index_time,feature_df.loc[feature_df['S'] == 1,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 1,'k']], alpha=aph)
plt.title("Sensor 1")
plt.xlabel('Index Time')
plt.ylabel('Sensor 1 Temp')

plt.subplot(2,3,5)
plt.xlim([0,index_time[-1:]])
plt.ylim([0,40])
plt.scatter(index_time,feature_df.loc[feature_df['S'] == 4,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 4,'k']], alpha=aph)
plt.title("Sensor 4")
plt.xlabel('Index Time')
plt.ylabel('Sensor 4 Temp')

plt.subplot(2,3,6)
plt.xlim([0,index_time[-1:]])
plt.ylim([0,40])
plt.scatter(index_time,feature_df.loc[feature_df['S'] == 7,'STemp'], linewidth=.1, color=color_theme_a[feature_df.loc[feature_df['S'] == 7,'k']], alpha=aph)
plt.title("Sensor 7")
plt.xlabel('Index Time')
plt.ylabel('Sensor 7 Temp')
plt.show()

# feature_df.to_csv('labeled.csv',sep=',')