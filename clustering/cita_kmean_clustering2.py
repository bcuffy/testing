import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("seaborn-dark")

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource, LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar

import seaborn as sb
sb.set_style('whitegrid')

import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report

#scaling is important for kmeans

plt.figure(figsize=(17,4))

df = pd.read_csv("CITA_ML_Ext_Cond_ImputedS.csv")

feature_df = df.copy(deep=True)
feature_df = df.drop(['o_time','d_time','Yr','Mo','Day'], 1)

X = scale(feature_df.values) #has no header
# y = df['Sensor']

variable_names = feature_df.columns

# Fit data to kmeans clustering algorithm
clustering = KMeans(n_clusters=3, random_state=4)

clustering.fit(X)

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue', "#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"])
color_theme_a = np.array([  "#933b41", 'lightsalmon', 'powderblue',"#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", 'darkgray', "#dfccce", "#ddb7b1", "#cc7878","#550b1d"])
color_theme_a = np.array([  "#933b41", 'lightsalmon', "#a5bab7",'darkgray', 'powderblue', "#c9d9d3", "#e2e2e2", "#75968f", "#dfccce", "#ddb7b1", "#cc7878","#550b1d"])

tindex = np.arange(0, len(df.STemp))
feature_df['k'] = clustering.labels_
aph = .6
print(clustering.labels_)

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
# l = zip(np.array(y), clustering.labels_)

from collections import Counter

# print(classification_report(np.array(y), clustering.labels_))

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
                x_range=[df[dfx].min(), df[dfx].max()], y_range=[-80,80],
                y_axis_label=str(dfy),  tools=TOOLS)

    # define what to draw on canvas
    p.circle(source=source, x=df[dfx], y=df[dfy], radius=.5, fill_color=color_theme[df_label], fill_alpha=.5, line_color='black', line_alpha=.25)

    # Define HoverTool
    p.select_one(HoverTool).tooltips = [
        ('Sensor', '@S')
    ]

    return p

df['index'] = tindex 

color_theme = np.array(["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", 'darkgray', 'lightsalmon', 'powderblue', "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"])
# p = row(plot(df, 'index', 'Solar Azimuth', clustering.labels_, color_theme))#, plot(df, 'Solar Alt', 'SP2_2', 'r', color_theme))
# show(p)