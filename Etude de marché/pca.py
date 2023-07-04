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
import warnings
warnings.simplefilter("ignore")
# PCA
from sklearn.decomposition import PCA
# KMeans
from sklearn.cluster import KMeans
# Pipeline
from sklearn.pipeline import make_pipeline


# In[2]:


# Palette des couleurs pour clusters 
palette = {1: '#49A462', 2: '#E52335', 3: '#E5B823', 4: '#23A3E5', 5: '#FD6D72', 6: '#23E580', 7: '#E523DB',
           8: '#4923E5', 9: '#545014', 10: '#6CE523'}


# In[3]:


def scree_plot(df, scaler_pca, t_var):
    
    # Mise à l'échelle des données 
    l = df.shape[1]
    df_scaled = scaler_pca.fit_transform(df)
    
    # Instanciation PCA 
    pca = PCA()
    pca.fit(df_scaled)
    
    # Données PCA
    var_exp = pca.explained_variance_
    var_exp_ratio = pca.explained_variance_ratio_
    
    # Nombre de composantes cumulants t_var % de variance
    n_comp = np.argmax(var_exp_ratio.cumsum() >= t_var)
    
    # Tableau résumant les résultats de la PCA
    dict_pca = { 
                # Valeurs propres ou variances expliquées par les nouveaux axes
                'Variance expliquée' : var_exp, 
                # Ratios de vriances expliquées
                '% variance expliquée' : var_exp_ratio * 100,
                # Cumul des variances expliquée
                '% cumul variance expliquée' : var_exp_ratio.cumsum() * 100
                 }
    data_pca = pd.DataFrame.from_dict(dict_pca, orient = 'index', columns = ['F'+ str(i + 1) for i in range(l)]).round(2)
    
    # Affichage du tableau
    display(data_pca)
    print('\n\n')
    
    # Scree Plot (Les nouvelles dimensions et leurs variances (en %))
    if t_var > 1 or t_var <= 0 : print('Erreur : Le taux de variance cumulée doit être compris entre 0 et 1.')
    else : 
        fig, ax = plt.subplots(figsize=(10,5))
        # Diagramme des éboulies  
        ax.plot(range(1, l+1),var_exp_ratio.cumsum(), marker ='o', label = 'Variance expliquée', c ='r')
        ax.bar(range(1, l+1), var_exp_ratio, label = 'Cumul de la variance expliquée')

        plt.ylabel('Pourcentage de variance expliquée')
        plt.xlabel('index des composantes principales')
        plt.title(f" Sree Plot (Scaler : {scaler_pca}) \n ({n_comp} composantes cumulants plus de {t_var * 100} % de la variance)", fontsize = 14)
        
        plt.axvline(n_comp, c='grey')
        plt.legend(loc='best', fontsize = 12)
        plt.tight_layout()
        
        fig.show()


# In[5]:


def pca_axes(df, scaler_pca, n_comp):
    # Mise à l'échelles des données
    df_scaled = scaler_pca.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    # PCA
    pca= PCA(n_comp)
    df_reduced = pca.fit_transform(df_scaled)
    df_pca = pca.components_
    var_ratio = pca.explained_variance_ratio_
    n = len(df)
    
    pca_labels = [f"F{i+1} ({var:.1f}%)" for i, var in enumerate(var_ratio * 100)]
    df_pca = pd.DataFrame(df_pca.T, columns = pca_labels, index = df.columns)
    
    # Coefficient de corrélation des variables avec les différentes composantes
    df_pca_corr = df_pca 
    # Qualité de la représentation des variables par l'ACP - Corrélation au carré = COS^2
    df_pca_cos2 = df_pca_corr.apply(lambda x : x*x)
    #Poids d'une variable dans la definition d'une composante principale
    df_pca_cos2_copy = df_pca_cos2.copy()
    s = df_pca_cos2_copy.sum()
    for j in range(df_pca_cos2_copy.shape[1]):
        for i in range(df_pca_cos2_copy.shape[0]):
            df_pca_cos2_copy.iloc[i,j] = (df_pca_cos2_copy.iloc[i,j]/s[j])
    # HeatMap des poids
    #fig = plt.figure(figsize = (12,6))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5), sharex=False, sharey=False)
    axes = axes.ravel() 
    sns.heatmap(df_pca_cos2_copy, annot=True, linewidth=.5,center=0, cmap='Spectral', ax = axes[1])
    sns.heatmap(df_pca_corr, annot=True, linewidth=.5,center=0, cmap='Spectral', ax = axes[0])
    axes[0].set_title('Coefficient de corrélation entre variables/axes')
    axes[1].set_title('Poids des variables/axes')
    fig.show()


# In[ ]:


def pca_visualizer(df, scaler,n_comp, comp1, comp2, k = 1, clust = False, text = True, quality = False):
    # definition des couleurs par cluster
    colors = {}
    for i, color in enumerate(palette.values()) : 
        if i < k : colors[i] = color
    
    # Mise à l'échelle des données
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled)
    
    # PCA
    pca= PCA(n_comp)
    pca.fit(df_scaled)
    
    # Données réduites
    df_reduced = pca.transform(df_scaled)
    # Matrice des chargements
    df_pca = pca.components_
    
    # KMeans, centroïdes et labels
    km = KMeans(k, random_state = 42)
    km.fit(df_reduced)
    centers_reduced = km.cluster_centers_
    labels = km.labels_
    
    # Associations couleurs/labes
    df_copy = df_scaled.copy()
    df_copy['clusters'] = labels
    
    # Qualité de représentation des individus sur le plan
    X_rep = df_reduced.copy()
    X_rep = pd.DataFrame(X_rep, index = df.index)
    
    # Transformation en cosinus carré
    X_rep_cos2 = X_rep.apply(lambda x : x*x) 
    
    # somme des cos2
    s = X_rep_cos2.T.sum() 
    
    # cos2 sur la somme des cos2
    for j in range(X_rep_cos2.shape[1]):
            for i in range(X_rep_cos2.shape[0]):
                X_rep_cos2.iloc[i,j] = X_rep_cos2.iloc[i,j]/s[j]
   
    
    # BiPlot
    # Valeurs maximales et minimales pour les axes
    max = df_reduced.max().max()
    max = max + 1
    
    # Adaptation pour l'affichage des vecteurs propres par un coefficient 
    dfpca = df_pca.T.copy() * max 
    
    var = pca.explained_variance_ratio_ * 100

    # Condition d'affichage des clusters
    if clust == True :
        c_ind = df_copy['clusters'].map(colors) # Association des couleurs et des labels 
        c_ctr = colors.values() 
    else : 
        c_ind = '#41ADFF'
        c_ctr = '#41ADFF'

    # Représentation des données sur le les plans formés par les composantes principales deux à deux
    #for j in range(0,n_comp,2):
        
    plt.figure(figsize =(20,20))
    plt.title("Biplot")
       
        
    # Projection des données sur le plan formé par deux axes
    R = X_rep_cos2.iloc[:,comp1] + X_rep_cos2.iloc[:,comp2]
    x, y = df_reduced[:,comp1], df_reduced[:,comp2] 
    
    # Accianement de l'option qualité de représentation
    if quality == True : s = R*1200
    else : s = 100
    
    
    # Projection des données   
    plt.scatter(x, y, c = c_ind, s = s)
    #for i in range(len(df)):
        #if R[i]>0.5:
            #plt.scatter(df_reduced[i,comp1], df_reduced[i,comp2], c = 'orange', s = R[i]*1200)
        #else : 
            #plt.scatter(df_reduced[i,comp1], df_reduced[i,comp2], c = c_ind[i], s = R[i]*1200)
        
    # projection des centroides de clusters
    # Condition d'affichage des clusters
    if clust == True :
        plt.scatter(centers_reduced[:,comp1], centers_reduced[:,comp2] , c = c_ctr , s=300, marker="v")
        
    # Condition d'affichage des intitulés 
    if text == True :
        for i, (x, y) in enumerate(zip(x,y)):
            plt.text(x, y, df.index[i], fontsize='10', ha = 'left', va = 'top')
    
    # Représentation des vecteurs-variables formants les composantes principales deux à deux
    for i, (x1, y1) in enumerate(zip(dfpca[:,comp1], dfpca[:,comp2])) :
        # Orientation du text
        if dfpca[i,j] < 0 : ha = 'right'
        else : ha = 'left'
        # Affichage des variables
        plt.arrow(0, 0, x1, y1, head_width = 0.03, width = 0.003, color = 'red')
        plt.text(x1, y1, df.columns[i], fontsize='12', c = 'red', ha = ha, va = 'top')
    
    # Axes centrals
    plt.axhline(0, c="grey")
    plt.axvline(0, c="grey")
        
    # Cerle unité agrandi par max
    plt.gca().add_artist(plt.Circle((0,0), max, color='red',fill=False))

    # Labels des axes
    plt.xlabel(f"PC {comp1} ({var[comp1]:.1f}%)")
    plt.ylabel(f"PC {comp2} ({var[comp2]:.1f}%)")

    # Limites des axes
    plt.xlim([-max, max])
    plt.ylim([-max, max])

