#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:09:12 2025

@author: zachkaye

# # Filename:   property_calculator.py
# # Package:    instron_analysis
# # Author:     Zach Kaye
# # Created:    
# # Modified:   27 August 2026
# # Python Version: 3.9
# # Description: Contains functions for running Instron_Code jupyter notebook, 
# # trims slack and analyzes modulus and max tenacity
# #
# # Copyright:  Copyright 2023, TomTex Inc., All rights reserved.
# # Requires:   numpy, pandas, pyEarth, regex, scipy.integrate


"""
import pandas as pd
import numpy as np
import scipy.integrate
from scipy.signal import savgol_filter, find_peaks
import re

HAS_PYEARTH = False
try:
    from pyearth import Earth
    HAS_PYEARTH = True
except ImportError:
    HAS_PYEARTH = False

def xy_pointFinder(x: pd.Series, y: pd.Series, target_x: float) -> tuple[float, float]:
    
    '''
    Finds (x,y) coordinates by using minimum difference from model to target value
    '''
    
    nearest_idx = (x - target_x).abs().idxmin()
    return float(x.loc[nearest_idx]), float(y.loc[nearest_idx])


def mars_model(x,y):

    '''
    Creates the MARS fit for raw data to extract spline intercepts. There is a fallback
    if pyearth is missing or fails to fit data.
    '''    
    
    if HAS_PYEARTH:
        try:
            model = Earth(max_terms = 40,
                          max_degree = 1,
                          enable_pruning = True)
            model.fit(x,y)
            y_hat = pd.Series(model.predict(x), index = x.index)
            model_df = pd.DataFrame({'x': x,
                                     'y': y_hat})
            
            intercepts = str(model.basis_).splitlines()[::2]
            inters = []
            for item in intercepts[1:]:
                parts = re.split(r'[()-]', item)
                for part in parts:
                    try:
                        val = float(part)
                        if x.min() < val < x.max():
                            inters.append(val)
                    except ValueError:
                            continue
            inters = sorted(list(set(inters)))
            if inters:
                return model_df, inters
        except Exception:
            pass
            
        model_df = pd.DataFrame({'x': x,
                                 'y': y})
        inters = [float(x.quantile(0.01)), float(x.quantile(0.05))]
        
        return model_df, inters
        
def find_uyt(strain: pd.Series,
             stress: pd.Series,
             window_length: int = 51,
             polyorder: int = 3,
             start_search_fraction = 0.005,
             end_search_fraction = 0.55
        ) -> tuple[float, float]:
    '''
    Identifies end point of yeild transition (ultimate yield transition, UYT)
    and beginning of permenant deformation using Savitzky-Golay derivative detection.
    '''
    
    strain = np.array(strain)
    stress = np.array(stress)
    
    d_strain = np.mean(np.diff(strain))
    smooth_stress = savgol_filter(stress, window_length = window_length,
                                  polyorder = polyorder, deriv = 0)
    ds_de = savgol_filter(stress, window_length = window_length,
                                  polyorder = polyorder, deriv = 1, delta = d_strain)
    
    start_idx = int(len(strain) * start_search_fraction)
    end_idx = int(len(strain) * end_search_fraction)
    if end_idx <= start_idx:
        end_idx = len(strain)
    
    ds_de_search = ds_de[start_idx:end_idx]
    smooth_search = smooth_stress[start_idx:end_idx]
    
    peaks, properties = find_peaks(smooth_search, prominence = 0.01 * np.ptp(smooth_search))
    
    if len(peaks) > 0:
        uyt_idx = start_idx + peaks[0]
        return float(strain[uyt_idx]), float(stress[uyt_idx])
    
    zero_crossings = np.where(np.diff(np.sign(ds_de_search)) < 0)[0]
    if len(zero_crossings) > 0:
        uyt_idx = start_idx + zero_crossings[0]
        return float(strain[uyt_idx]), float(stress[uyt_idx])
    
    neg_abs_ds_de = -np.abs(ds_de_search)
    deriv_peaks, _ = find_peaks(neg_abs_ds_de, prominence=0.02 * np.ptp(neg_abs_ds_de))
    if len(deriv_peaks) > 0:
        uyt_idx = start_idx + deriv_peaks[0]
        return float(strain[uyt_idx]), float(stress[uyt_idx])

    max_in_window = start_idx + int(np.argmax(smooth_search))
    return float(strain[max_in_window]), float(stress[max_in_window])
    
##add no uyt point here instead of max_in_window


def trim_end(df: pd.DataFrame()) -> int:
    
    '''
    Finds idx where break occurs.
    '''
    
    stress = df['Stress (MPa)']
    if len(stress) == 0:
        return 0
    
    max_idx = stress.idxmax()
    post_peak = stress.loc[max_idx:]
    
    delta = post_peak.diff()
    min_delta_idx = delta.idxmin()
    
#    if pd.notna(min_delta_idx) and min_delta_idx in df.index:
#        idx_pos = df.index.get_loc(min_delta_idx)
#        return max(1, idx_pos)
        
    return min_delta_idx

def trim(x: pd.Series, y_hat: pd.Series, inters: list[float]) -> int:
    
    '''
    Finds starting integer row index to trim initial slack.
    '''
    
    if not inters or len(inters) < 2 or len(x) == 0:
        return 0

    # Extract coordinates for Knot 1 and Knot 2
    x_pt1, y_pt1 = xy_pointFinder(x, y_hat, inters[0])
    x_pt2, y_pt2 = xy_pointFinder(x, y_hat, inters[1])
    x_start, y_start = float(x.iloc[0]), float(y_hat.iloc[0])

    # Segment 0 slope: Origin -> Knot 1
    dx1 = x_pt1 - x_start
    m1 = (y_pt1 - y_start) / dx1 if abs(dx1) > 1e-6 else 0.0

    # Segment 1 slope: Knot 1 -> Knot 2
    dx2 = x_pt2 - x_pt1
    m2 = (y_pt2 - y_pt1) / dx2 if abs(dx2) > 1e-6 else 0.0

    # Slack condition: Initial region (m1) is flatter than elastic region (m2)
    if m1 < m2:
        matches = x[x == x_pt1].index
        if len(matches) > 0:
            return int(matches[0])

    # If m1 >= m2, the test started directly in the elastic region (no slack)
    return 0


def find_modulus(x: pd.Series, y_hat: pd.Series, inters: list[float]) -> tuple[float, float, list[float]]:
    
    '''
    Calculates Young's modulus from origin to first spline detected, the elastic intercept.'
    '''
    
    if not inters:
        k = float(x.quantile(0.01))
        x_pt, y_pt = xy_pointFinder(x, y_hat, k)
        return (y_pt/x_pt if x_pt > 0 else 0.0), k, [x_pt, y_pt]
    
    x_start, y_start = float(x.iloc[0]), float(y_hat.iloc[0])
    best_m = -1.0
    best_knot = inters[0]
    best_coords = [0.0, 0.0]
    
    for knot in inters[:3]:
        x_pt, y_pt = xy_pointFinder(x, y_hat, knot)
        dx = x_pt - x_start
        if dx > 1e-6:
            m = (y_pt - y_start) / dx
            if m > best_m:
                best_m = m
                best_knot = knot
                best_coords = [x_pt, y_pt]
                
    return float(best_m), float(best_knot), best_coords


def offset_yield(x: pd.Series,
                 x2p: pd.Series,
                 y_hat: pd.Series,
                 second_intercept: float,
                 youngs: float,
                 coords: list[float]
                 ) -> tuple[float,float]:
   
    '''
    Calculates the yield point using the 0.2% offset method.
    Exact coordinates are extracted by finding the intersection of the offset slope
    and line connecting the first (end of elastic) and second (upper bound of yield 
    transition) knots detected by the MARS model.
    '''
    
    x_pt2, y_pt2 = xy_pointFinder(x, y_hat, second_intercept)

    dx = x_pt2 - coords[0]
    m = (y_pt2 - coords[1])/dx if abs(dx) > 1e-6 else 0.0
    b = y_pt2 - m*x_pt2
    b_offset = -youngs*float(x2p.iloc[0])

    denom = youngs - m
    if abs(denom) > 1e-6:
        x_offset = (b-b_offset)/denom
        y_offset = m*x_offset + b
    else:
        x_offset, y_offset = coords[0], coords[1]
        
    return float(x_offset), float(y_offset)


def adjust_df(df: pd.DataFrame,
              EGL: float,
              width: float,
              thickness:float
              ) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    
    '''
    Normalizes raw force (N) and position (mm) into stress/strain, filters outliers, and trims
    slack/rupture.
    '''
    
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')
    
    
    df['Elongation'] = (df['Position (mm)']/EGL)*100
    df['E.2%'] = df['Elongation'] + 0.2
    df['Force (N)'] = df['Force (N)'] - df['Force (N)'].iloc[0]
    df['Stress (MPa)'] = df['Force (N)'] / (width * thickness)
    
    df = df[(df['Stress (MPa)'] >= -0.5) & (df['Stress (MPa)'] <= df['Stress (MPa)'].mean() * 3)].copy()
    
    x = df['Elongation']
    y = df['Stress (MPa)']
    model_df, inters = mars_model(x,y)
        
    idx_end = trim_end(df)
    df = df.iloc[:idx_end]

    try:
        start_idx = trim(x, model_df.y, inters)
    except Exception:
        start_idx = 0
        pass
    
    if start_idx > 0:
        df = df.loc[start_idx:].reset_index(drop = True)
        l_o = df['Position (mm)'].iloc[0]
        df['Elongation'] = (df['Position (mm)'] - l_o)/(l_o + EGL) * 100
        df['E.2%'] = df['Elongation'] + 0.2

        df['Stress (MPa)'] = df['Stress (MPa)'] - df['Stress (MPa)'].iloc[0]
        df['Force (N)'] = df['Force (N)'] - df['Force (N)'].iloc[0]
        
    df = df.reset_index(drop = True)
    model_df, inters = mars_model(df['Elongation'], df['Stress (MPa)'])

        
    return df.reset_index(drop = True), model_df, inters


def analyze(df: pd.DataFrame,
            EGL: float,
            width: float,
            thickness: float
            ) -> tuple[float, float, float, float, float, float, float]:

    '''
    Computes full mechanical property readouts for a single specimen.
    '''
    
    df_adj, model_df, inters = adjust_df(df, EGL, width, thickness)
    
    x = df_adj['Elongation']
    x2p = df_adj['E.2%']
    y = df_adj['Stress (MPa)']
    
    modulus, elastic_knot, elastic_endpoint = find_modulus(x, model_df['y'], inters)
    
    remaining_knots = [k for k in inters if k > elastic_knot]
    second_knot = remaining_knots[0] if remaining_knots else float(x.quantile(0.05))
    
    yield_strain, yield_stress = offset_yield(x, x2p, model_df['y'], second_knot, modulus, elastic_endpoint)
    
    max_elongation = float(model_df['x'].max())
    uts = float(model_df['y'].max()) #ultimate tensile strength
    
    uy_strain, uy_stress = find_uyt(x, y) #ultimate yield transition
   
    try:
        toughness = float(scipy.integrate.simpson(y = model_df['y'].values,
                                     x = model_df['x'].values))
    except AttributeError:
        toughness = float(scipy.integrate.simps(y = model_df['y'].values,
                                     x = model_df['x'].values))

    return modulus*100, yield_strain, yield_stress, max_elongation, uts, uy_strain, uy_stress, toughness


def data_table(df_list: list[pd.DataFrame],
               sample_thickness_pairs: list[tuple[str, float]],
               EGL_list,
               widths
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    '''
    Processes specimen tests and returns inidividual, mean and std DataFrames
    '''
    
    target_len = len(df_list)
    def _normalize(val, name):
        if isinstance(val, (int, float)):
            return [float(val)]*target_len
        elif isinstance(val, (list, tuple)):
            if len(val) == 1:
                return list(val) * target_len
            elif len(val) == target_len:
                return list(val)
        raise ValueError(f"Invalid length for {name}: expected {target_len}")
        
    egls = _normalize(EGL_list, 'EGL_list')
    w_list = _normalize(widths, 'widths')    
    
    thicknesses = [d for _,d in sample_thickness_pairs]
    names = [n for n,_ in sample_thickness_pairs]
    
    cols = ['modulus',
            'yield strain',
            'yield stress',
            'elongation',
            'UTS',
            'UY strain',
            'UY stress',
            'toughness']
    
    records = [analyze(df, egl, w, d) for df, egl, w, d in zip(df_list, egls, w_list, thicknesses)]
    
    ind_data_table = pd.DataFrame(records, columns = cols, index = names)
    averages = ind_data_table.groupby(ind_data_table.index).mean().round(3)
    stds = ind_data_table.groupby(ind_data_table.index).std().round(3)   
    
    return ind_data_table, averages, stds