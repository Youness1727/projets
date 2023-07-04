#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
# Graphics
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Clustring
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[2]:


def k_kmeans(df, scaler):
    # Mise à l'échelle des donnée
    df_scaled = scaler.fit_transform(df)
    n = len(df_scaled)
    # Metrics pour determiner le nombre optimal de clusters  
    fig, ax = plt.subplots(1,2, figsize = (20,5))
    
    # KMeans
    model = KMeans()
    # Instanciation des visualiseurs
    visualizer_coude = KElbowVisualizer(model, k=(2,11), timings = False, ax = ax[0])#, ax = ax[0]
    # Entraînement des données
    visualizer_coude.fit(df_scaled)
    # Nombre optimal de clusters
    k_opt = visualizer_coude.elbow_value_

    # Visuel des clusters avec leur score silhouette
    visualizer_silhouette = SilhouetteVisualizer(KMeans(k_opt), colors='yellowbrick', ax = ax[1])
     # Entraînement des données
    visualizer_silhouette.fit(df_scaled)
    
    # Titre et labels des axes 
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('Distortion Score')
    ax[0].set_title('Distortion Score Elbow') 
    
    # Titre et labels des axes 
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('Silhouette score')
    ax[1].set_title(f'Score Silhouette pour {n} valeurs en {k_opt} clusters') 


# In[1]:


def kmeans_visualizer(df ,k ,scaler, col_iso):
   # Scaler
   df_scaled = scaler.fit_transform(df)

   # Mise à l'échelle [0,1]
   df_rad = MinMaxScaler().fit_transform(df)
   df_rad = pd.DataFrame(df_rad, columns = df.columns, index = df.index)
   
   # KMeans       
   kmeans = KMeans(k, random_state = 42)
   kmeans.fit(df_scaled)
   labels = kmeans.labels_
   
   # Carte Choropleth pour visualiser les pays selon leur appartenance de cluster
   fig = go.Figure(data=go.Choropleth(
                                       locations=col_iso, 
                                       z = labels, 
                                       colorscale = 'spectral',
                                       colorbar_title = "Groupes pays",
                                    ))

   fig.update_layout(
                      title_text = 'Classification KMeans pays'
                      )
   fig.show()

   # Visualisation des clusters
   print('Clusters:')
   for n in range(k):
       X_km = df.copy()
       X_km['clusters'] = labels
       Result_km = X_km[X_km.clusters == n].index
       print('\nCluster n° ', n,' :\n', Result_km)
       
       
    # Données Radar
   df_rad['kmeans_clusters'] = labels
   df_rad = df_rad.groupby('kmeans_clusters').mean()
   print('\n')
   display(df_rad)    
       
       
   
    # figure Radar
   if k<=6:
       fig = make_subplots(rows=3, cols=2, specs=[[{'type':'Scatterpolar'}, {'type':'Scatterpolar'}],
                                              [{'type':'Scatterpolar'}, {'type':'Scatterpolar'}],
                                              [{'type':'Scatterpolar'}, {'type':'Scatterpolar'}]])
       for i in range(k):
           q, r = divmod(i,2)
           fig.add_trace(go.Scatterpolar(
                               r=df_rad.iloc[i].values,
                               theta=df_rad.columns,
                               fill='toself',
                               name="Groupe - %s"%df_rad.index[i],
                               showlegend=True, opacity=0.5,
                               #fillcolor=list(colors.values())[i], 
                                          ), row=q+1, col=r+1)
           fig.update_layout(
                               polar=dict( radialaxis=dict(visible=True, range=[0, 1])),
                               title="Diagramme Radar pour clusters K-Means",
                               height=1200, width=1000
                                )
       fig.show()
   
   if k > 6:
       fig = make_subplots(rows=3, cols=2, specs=[[{'type':'Scatterpolar'}, {'type':'Scatterpolar'}],
                                              [{'type':'Scatterpolar'}, {'type':'Scatterpolar'}],
                                              [{'type':'Scatterpolar'}, {'type':'Scatterpolar'}]])
       for i in range(6):
           q, r = divmod(i,2)
           fig.add_trace(go.Scatterpolar(
                               r=df_rad.iloc[i].values,
                               theta=df_rad.columns,
                               fill='toself',
                               name="Groupe - %s"%df_rad.index[i],
                               showlegend=True, opacity=0.5,
                               #fillcolor=list(colors.values())[i], 
                                          ), row=q+1, col=r+1)
           fig.update_layout(
                               polar=dict( radialaxis=dict(visible=True, range=[0, 1])),
                               title="Diagramme Radar pour clusters K-Means",
                               height=1200, width=1000
                                )
       fig.show()   
       
       fig = make_subplots(rows=3, cols=2, specs=[[{'type':'Scatterpolar'}, {'type':'Scatterpolar'}],
                                              [{'type':'Scatterpolar'}, {'type':'Scatterpolar'}],
                                              [{'type':'Scatterpolar'}, {'type':'Scatterpolar'}]])
       for i in range(k - 6):
           q, r = divmod(i,2)
           fig.add_trace(go.Scatterpolar(
                               r=df_rad.iloc[i+6].values,
                               theta=df_rad.columns,
                               fill='toself',
                               name="Groupe - %s"%df_rad.index[i+6],
                               showlegend=True, opacity=0.5,
                               #fillcolor=list(colors.values())[i], 
                                          ), row=q+1, col=r+1)
           fig.update_layout(
                               polar=dict( radialaxis=dict(visible=True, range=[0, 1])),
                               title="Diagramme Radar pour clusters K-Means",
                               height=1200, width=1000
                                )
       fig.show()

def kmeans_labels(df, k, scaler) :
    # Scaler
   df_scaled = scaler.fit_transform(df)
   # KMeans       
   kmeans = KMeans(k, random_state = 42)
   kmeans.fit(df_scaled)
   labels = kmeans.labels_
   return labels