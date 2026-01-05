#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:42:23 2025

@author: zachkaye


# # Filename:   full_plot.py
# # Package:    instron_analysis
# # Author:     Zach Kaye
# # Created:    
# # Modified:   28 August 2025
# # Python Version: 3.9
# # Description: Contains functions for running Instron_Code jupyter notebook, 
# # trims slack and analyzes modulus and max tenacity
# #
# # Copyright:  Copyright 2023, TomTex Inc., All rights reserved.
# # Requires:   matplotlib, numpy, time

"""

import matplotlib.pyplot as plt
import numpy as np
import time

def plot_data(df_list, 
              title: str, 
              rep, 
              sample_thickness_pairs, 
              cmap = plt.get_cmap('viridis'),
              figsize = [14,10],
              save = False
              ):
    
    '''
    Plots the complete set of instron data. If there is only one DataFrame to
    plot, enclose it in []. Title, repetition number and "sample_thickness_pairs"
    metadata (from the process function) are all required for this function. 
    Optional arguments include cmap (set of colors), figure size 
    and the option to save the figure.
    '''
    
    N = len(sample_thickness_pairs)
    number_list = np.repeat(np.linspace(0,rep,N),rep)
    colors = [cmap(i) for i in number_list]
    marker_list = ['o','+','*','p','s','D','v','h','H']
    if N > len(marker_list):
        marker_list += marker_list
    
    markers = np.repeat(marker_list, rep)
    
    fig,ax = plt.subplots(figsize = figsize)
    fig.patch.set_facecolor('xkcd:white')
    
    for df,c,marker,label in zip(df_list, 
                                 colors,
                                 markers,
                                 [name for name,d in sample_thickness_pairs]):
        
        mask = df['Load (MPa)'] >= -0.5
        df = df[mask]
        
        df.plot(ax = ax,
                x = 'Elongation',
                y = 'Load (MPa)',
                marker = marker,
                label = label,
                color = c)
        
    ax.set_title(title)
    ax.set_ylabel('Load (MPa)')
    ax.set_xlabel('Elongation (%)')
    
    handles,labels = ax.get_legend_handles_labels()
    plt.legend(handles[::rep], labels[::rep], bbox_to_anchor = [1.15,0.5])
    
    plt.tight_layout()
    if save:
        plt.savefig(time.strftime('%y%m%d') + '_' + title)
        
    plt.show()