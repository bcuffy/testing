import numpy as np
import pandas as pd

import cufflinks as cf

import plotly
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go

tls.set_credentials_file(username='bcuffy', api_key='rsg7eF66w2YBnZUERT10')

df = pd.read_csv("CITA_ML_Ext_Cond_Imputed_stacked.csv")
feature_df = df.drop(['Solar Azimuth', 'Solar Alt', 'c_time','o_time','d_time','Yr','Mo','Day'], 1)

# layout = dict(title = 'Chart from PD',  xaxis=dict(title='x-axis'), yaxis=dict(title='y-axis'))
f = feature_df[['STemp']]
# feature_df.iplot(filename='cf_simple_chart', layout=layout)

data = [go.Bar(x=feature_df['STemp'], y=feature_df['T_in'])]
layout = dict(title = 'Bar Chart',  xaxis=dict(title='x-axis'), yaxis=dict(title='y-axis'))
feature_df.iplot(data, filename='bar_chart', layout=layout)