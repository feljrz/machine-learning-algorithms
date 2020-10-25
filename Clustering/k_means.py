#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:12:48 2020

@author: felipe
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def start_centroids(k, maximun):
    centroids = np.random.rand(k,2)*maximun
    return centroids

def graph_kmeans(clusters):
    for df in clusters:
        k_mean = df['distance'].mean(axis=0)
        k= np.max(df['group'].unique()) + 1
        plt.plot(k, k_mean,'-o')
    plt.show()

        
    return k_mean
    
def grap_scatter(clusters):
    clrs=['r','g','b','c','m','y','k']

    for df in clusters:
        df['group'] = df.group.astype(int)
        k= np.max(df['group'].unique())
        #print(k)
        
    
        for i in range(k+1):
            indices = df.loc[df['group'] == i].index.to_list()
            df.loc[indices, 'color'] = clrs[i]
        
        plt.scatter(df['x'], df['y'], color=df['color'])
        plt.show()
    
    
def closest_centroid(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return distances, np.argmin(distances, axis=0)


def update_centroid(points, centroids, groups):
    new_centroid = []
    for i in range(centroids.shape[0]):
        new_centroid.append(points[groups == i].mean(axis=0))
    return np.array(new_centroid)

def k_means(points, epochs, n_centroids):
    flag = 0
    k = 0
    dict_k = dict()
    lista = []
    
    

    for i in np.arange(n_centroids):
        k +=1
        centroids = start_centroids(k, 10)

        for i in np.arange(100):
            print(i)
            
            distance, groups = closest_centroid(points, centroids)
            
            new_centroids = update_centroid(points, centroids, groups)
            difference = new_centroids - centroids
        
            if (difference > 0.001).all():
                centroids = new_centroids
            else:
                flag += 1
                break

        clusters = np.append(points, groups.reshape(-1,1), axis=1)
        
       
        dis_point = np.choose(clusters[:,-1].T.astype(int), distance).reshape(-1,1)
        df = np.append(clusters, dis_point, axis=1)
        df = pd.DataFrame(df)
        df.columns = ['x', 'y', 'group', 'distance']
        lista.append(df)
    return lista
            

points = np.random.rand(200,2)*10 #Pontos X e y
clusters = k_means(points, 100, 7)

grap_scatter(clusters)
graph_kmeans(clusters)
