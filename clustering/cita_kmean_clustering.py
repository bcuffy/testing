import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
import seaborn as sb
style.use("seaborn-darkgrid")

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource, LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar

import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report

#scaling is important for kmeans
plt.figure(figsize=(17,4))

df = pd.read_csv("CITA_ML_Ext_Cond_Imputed.csv")

feature_df = df.copy(deep=True)
feature_df = df.drop(['o_time','d_time','Yr','Mo','Day'], 1)

X = scale(feature_df.values) #has no header
# y = df['Sensor']

clustering = KMeans(n_clusters=10, random_state=4)
clustering.fit(X)

# print(classification_report(np.array(y), clustering.labels_))

color_theme = np.array(["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", 'darkgray', 'lightsalmon', 'powderblue', "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"])

tindex = np.arange(0, len(df.c_time))

def matplotlib_plot():

    f = 'SP2_1'

    plt.subplot(2,2,1)
    plt.xlim([-80,80])
    plt.ylim([0, 40])
    plt.scatter(x=df['Solar Azimuth'], y =df[f], c=color_theme[clustering.labels_], s=50, edgecolors='gray', linewidths=.2, facecolor=None, hatch=None)
    plt.xlabel('Azimuth')
    plt.ylabel('Temperature')
    plt.title('KMeans classification - Sensor 1')

    plt.subplot(2,2,2)
    plt.xlim([0, tindex[-1:]])
    plt.ylim([-80,80])
    plt.scatter(x=tindex, y =df['Solar Azimuth'], c=color_theme[clustering.labels_], s=50, edgecolors='gray', linewidths=.2, facecolor=None, hatch=None)
    plt.xlabel('Index Time')
    plt.ylabel('Azimuth')
    plt.title('KMeans-Time classification - Sensor 1')

    plt.subplot(2,2,3)
    plt.xlim([0, tindex[-1:]])
    plt.ylim([0,40])
    plt.scatter(x=tindex, y=df[f], c=color_theme[clustering.labels_], s=50, edgecolors='gray', linewidths=.2, facecolor=None, hatch=None)
    plt.xlabel('Index Time')
    plt.ylabel('Azimuth')
    plt.title('KMeans-Time classification - Sensor 1')
    
    plt.subplot(2,2,4)
    plt.xlim([0, tindex[-1:]])
    plt.ylim([0,40])
    plt.scatter(x=tindex, y=df['SP2_9'], c=color_theme[clustering.labels_], s=50, edgecolors='gray', linewidths=.2, facecolor=None, hatch=None)
    plt.xlabel('Index Time')
    plt.ylabel('Azimuth')
    plt.title('KMeans-Time classification - Sensor 9')


    plt.show()






def bokeh_plot(df, dfx, dfy, df_label, color_theme):
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
    p.circle(source=source, x=df[dfx], y=df[dfy], radius=1, fill_color=color_theme[df_label], fill_alpha=.5, line_color='black', line_alpha=.25)

    # Define HoverTool
    p.select_one(HoverTool).tooltips = [
        ('Sensor', '@S')
    ]

    return p

df['index'] = tindex 

matplotlib_plot()

# p = row(bokeh_plot(df, 'index', 'Solar Azimuth', clustering.labels_, color_theme))#, plot(df, 'Solar Alt', 'SP2_2', 'r', color_theme))
# show(p)