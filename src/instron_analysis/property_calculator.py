#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:09:12 2025

@author: zachkaye

# # Filename:   property_calculator.py
# # Package:    instron_analysis
# # Author:     Zach Kaye
# # Created:    
# # Modified:   28 August 2025
# # Python Version: 3.9
# # Description: Contains functions for running Instron_Code jupyter notebook, 
# # trims slack and analyzes modulus and max tenacity
# #
# # Copyright:  Copyright 2023, TomTex Inc., All rights reserved.
# # Requires:   numpy, pandas, pyEarth, regex, scipy.integrate


"""

import pandas as pd
import scipy.integrate
from pyearth import Earth
import re
from decimal import Decimal

def xy_pointFinder(x,y,point):
    '''
    Converts numbers in data and model DataFrames to decimals so hinges can be
    properly mapped onto original data x,y coordinates.
    '''
    PRECISION = 2
    
    x_dec = x.apply(lambda num: Decimal(f'{num:.{PRECISION}f}'))
    point_dec = Decimal(f'{point:.{PRECISION}f}')
    
    mask = x_dec == point_dec
    
    x_point = x.loc[mask].item()
    y_point = y.loc[mask].item()
    
    return x_point, y_point


def mars_model(x,y):

    '''
    Creates the MARS fit for raw data.
    Load in raw data DataFrame's Elongation and Load columns for x and y.
    Returns a DataFrame of the model and a list of the intercepts.
    Intercepts here are points along the x-axis at which the model hinges.
    Straight lines are drawn between intercept points.
    '''    
    
    model = Earth(max_terms = 40,
                 max_degree = 1,
                 enable_pruning = True)
    
    model.fit(x,y)
    
    y_hat = model.predict(x)
    model_df = pd.DataFrame({
        'y':y_hat,
        'x':x
    })
    
    intercepts = str(model.basis_).splitlines()[::2]
    inters = [re.split('[()-]',i) for i in intercepts]
    inters = inters[1:]
    
    inters = sorted([float(tup[1]) for tup in inters])

    return model_df, inters


def trim_end(df):
    delta = df.diff()
    idx = delta['Load (MPa)'].idxmin()
    
    return idx


def trim(x, y_hat, inters):

    '''
    Removes excess slack picked up in instron test that may interfere 
    with modulus calculation. Load in x and y from the model and l
    ist of intercepts. Returns the index that is used later to slice raw data 
    DataFrame to omit region of slack.
    '''
    
    mod_filter = {}
    x_points = []
    y_points = []
    
    for i,hinge in enumerate(inters[:2]):
        x_point, y_point = xy_pointFinder(x, y_hat, hinge)
        
        x_points.append(x_point)
        y_points.append(y_point)

        if i == 0:
            m_youngs = (y_point-y_hat.iloc[0])/(x_point-x.iloc[0])
        else:
            m_youngs = (y_points[i] - y_points[i-1])/(x_points[i]-x_points[i-1])
    
        mod_filter[m_youngs] = [x_point, y_point]
    
    m_youngs_final = max(list(mod_filter.keys()))
    for key in mod_filter.keys():
        if key < m_youngs_final:
            
            idx = x.loc[x == mod_filter[key][0]].index
            break
        else:
            idx = 0
            break

    return idx
    

def find_modulus(x, y_hat, elastic_intercept):

    '''
    Calculates modulus from origin and point of first intercept.
    Load in model x and y and intercept at the end of elastic region 
    (should be first intercept in list). Returns youngs modulus, and 
    coordinates of intercept point.
    '''
    
    x_point, y_point = xy_pointFinder(x, y_hat, elastic_intercept)
    
    #x_point = x.loc[x.round(1) == np.float64(elastic_intercept).round(1)].to_list()[0]
    #y_point = [j for i,j in zip(x,y_hat) if i == x_point][0]
    
    m_youngs = (y_point-y_hat.iloc[0])/(elastic_intercept-x.iloc[0])

    return m_youngs, [x_point, y_point]


def offset_yield(x, x2p, y_hat, second_intercept, youngs, coords):

    '''
    Calculates the yield point using the 0.2% offset method.
    Load in model x, "E.2%" from original DataFrame, model y, 
    intercept after elastic, youngs modulus, and coordinates of 
    first intercept point. Returns the x and y of the intersection of the 
    offset line and line between first and second intercepts. This x and y is 
    the yield strain and yield stress.
    '''
    
    x_point2, y_point2 = xy_pointFinder(x, y_hat, second_intercept)

    m = (y_point2-coords[1])/(x_point2-coords[0])
    b = -m*x_point2+y_point2

    b_offset = -youngs*x2p.iloc[0]

    x_offset = (b-b_offset)/(youngs-m)
    y_offset = m*x_offset+b

    return x_offset, y_offset


def adjust_df(df,EGL,width,thickness):
    
    '''
    Added high and low mask to remove outliers from raw data.
    Applies trim to the front and back of tensile curve in 
    case of any anomalies in order to produce the best MARS fitting 
    in the analyze function. Also adds columns for Elongation, Load and 0.2% offset
    Returns the adjusted DataFrame.
    '''
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast = 'float')
    
    df['Elongation'] = (df['Position (mm)']/EGL)*100
    df['E.2%'] = df['Elongation'] + 0.2
    
    df['Force (N)'] = df['Force (N)'] - df['Force (N)'].iloc[0]
    
    df['Load (MPa)'] = df['Force (N)'] / (width * thickness)
    
    low_mask = df['Load (MPa)'] >= -0.5
    high_mask = df['Load (MPa)'] <= df['Load (MPa)'].mean()*3
    
    df = df[low_mask]
    df = df[high_mask]
    
    x = df['Elongation']
    y = df['Load (MPa)']
    model_df, inters = mars_model(x,y)
    
        
    idx_end = trim_end(df)
    df = df.iloc[:idx_end]

    try:
        idx = trim(x, model_df.y, inters)
    except IndexError:
        print('index error on sample')
        idx = 0
        pass
    
    if idx > 0:
        df = df.iloc[idx[0]:]
        df.reset_index(inplace = True)
        l_o = df['Position (mm)'].iloc[0]
        df['Elongation'] = (df['Position (mm)'] - l_o)/(l_o + EGL) * 100
        df['E.2%'] = df['Elongation'] + 0.2
        
        f_o_load = df['Load (MPa)'].iloc[0]
        df['Load (MPa)'] = df['Load (MPa)'] - f_o_load
        
        f_o_N = df['Force (N)'].iloc[0]
        df['Force (N)'] = df['Force (N)'] - f_o_N
        
    return df


def analyze(df,EGL,width,thickness):

    '''
    This function uses the functions mars_model, trim, find_modulus 
    and offset_yield. Takes the DataFrame of original data. Returns the
    youngs modulus, yield strain, yield stress, final elongation, 
    max tenacity, and toughness.
    '''
    
    df = adjust_df(df, EGL, width, thickness)
    
    x = df['Elongation']
    x2p = df['E.2%']
    y = df['Load (MPa)']

    model_df, inters = mars_model(x,y)
    
    modulus, elastic_endpoint = find_modulus(x, model_df.y, inters[0])
    yield_strain, yield_stress = offset_yield(x, x2p, model_df.y, inters[1], modulus, elastic_endpoint)
    
    max_elongation = model_df.x.max()
    max_tenacity = model_df.y.max()
    #max_tenacity = y.max()
    #max_elongation = x.max()

    toughness = scipy.integrate.simpson(y = model_df['y'].values,
                                     x = model_df['x'].values)

    return modulus, yield_strain, yield_stress, max_elongation, max_tenacity, toughness


def data_table(df_list, sample_thickness_pairs, EGL_list, widths):
    
    '''
    Applies the analyze function to a list of DataFrames from an entire
    instron data file that has been formatted. Returns DataFrames of individual
    tests, one of averages, and one of standard deviations.
    '''
    
    thickness = [d for name,d in sample_thickness_pairs]
    names = [name for name,d in sample_thickness_pairs]
    
    if len([EGL_list]) == 1:
        EGL_list = [EGL_list] * len(df_list)
    if len([widths]) == 1:
        widths = [widths] * len(df_list)
        
    cols = ['modulus',
            'yield strain',
            'yield stress',
            'elongation',
            'tenacity',
            'toughness']
    
    ind_data_table = pd.DataFrame(map(analyze, df_list, EGL_list, widths, thickness),
                              columns = cols,
                              index = names)
    
    averages = ind_data_table.groupby(ind_data_table.index).mean()
    stds = ind_data_table.groupby(ind_data_table.index).std()
 
    averages = averages.round(decimals = 3)
    stds = stds.round(decimals = 3)
    
    
    return ind_data_table, averages, stds