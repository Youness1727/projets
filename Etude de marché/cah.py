#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# CAH
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet 
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


def k_cah(df, scaler):
    df_scaled = scaler.fit_transform(df)
    # Matrice de distances
    Z = linkage(df_scaled, method= 'ward', metric= 'euclidean')
    # Nombre optimal de cluster
    last = Z[-10:, 2][::-1]
    val = []
    for i in range(1, len(last)-1): val.append(last[i]-last[i+1])
    
    # Nombre optimal de cluster
    Nb_clust_cah = np.argmax(val) + 3
    
    # Diagramme d'inertie
    plt.step(range(2, len(last)+2), last)
    plt.axvline(Nb_clust_cah, color='red', ls='--', label ='Nombre obtimal de clusters = %.f' % Nb_clust_cah)
    # titres et labels
    plt.title("Diagramme d'inertie")  
    plt.xlabel('k')
    plt.ylabel('Inertie')
    fig.show()


# In[ ]:


def dendrogramme(df, k, scaler) :
    # Données transformées
    df_scaled = scaler.fit_transform(df)
    # Matrice des liens
    Z = linkage(df_scaled, method='ward', metric= 'euclidean')
    # Distance entre deux groupes
    Z1, Z2 = Z[-k][2], Z[-k + 1][2]
    t = random.uniform(Z1, Z2)
    # Dendrogramme
    plt.figure(figsize =(20,20))
    plt.title('CAH avec matérialisation des %.f groupes' % k, fontsize = 15)
    dendrogram(Z, labels = df.index, color_threshold = t)
    plt.show()


# In[3]:


def cah_visualizer(df ,k ,scaler, col_iso):
   global labels
   # Scaler
   df_scaled = scaler.fit_transform(df)
   
   # CAH      
   Z = linkage(df_scaled, method='ward', metric= 'euclidean')
   Z1, Z2 = Z[-k][2], Z[-k + 1][2]
   t = random.uniform(Z1, Z2)
   labels = fcluster(Z, t = t, criterion='distance')
      
   # Visualisation des zones
   fig = go.Figure(data=go.Choropleth(
                                       locations=col_iso, 
                                       z = labels, 
                                       colorscale = 'spectral',
                                       colorbar_title = "Groupes pays",
                                    ))

   fig.update_layout(
                      title_text = 'Classification CAH pays'
                      )
   fig.show()
   
   # Visualisation des clusters
   print('Clusters:')
   for n in range(k):
       X_cah = df.copy()
       X_cah['clusters'] = labels
       Result_cah = X_cah[X_cah.clusters == n+1].index
       print('\nCluster n° ', n+1,' :\n', Result_cah.values)
          
   
   # Visualisation des données radar
   # Mise à l'échelle [0,1]
   df_rad = MinMaxScaler().fit_transform(df)
   df_rad = pd.DataFrame(df_rad, columns = df.columns, index = df.index)
     
   # Données Radar
   df_rad['cah_clusters'] = labels
   df_rad = df_rad.groupby('cah_clusters').mean()
   df_rad = df_rad.iloc[:,:-1]
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
                               title="Diagrammes Radar pour clusters CAH",
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
                               title="Diagrammes Radar pour clusters CAH",
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
                               title="Diagrammes Radar pour clusters CAH",
                               height=1200, width=1000
                                )
       fig.show()


# In[ ]:


def cah_labels(df, k, scaler):
    # Mise à l'échelle
    df_scaled = scaler.fit_transform(df)
    # Matrice des liens
    Z = linkage(df_scaled, method= 'ward', metric= 'euclidean')
    # distance entre deux groupes
    Z1, Z2 = Z[-k][2], Z[-k + 1][2]
    t_score = random.uniform(Z1, Z2)
    # labels CAH
    return fcluster(Z, t = t_score, criterion='distance')

