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
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Sequence, Optional, Tuple

from .property_calculator import(
    adjust_df,
    find_modulus,
    offset_yield,
    find_uyt
)

marker_list = ['o','+','*','p','s','D','v','h','H','X','d','>','8']

def plot_data(df_list: Sequence[pd.DataFrame],
              sample_thickness_pairs: Sequence[Tuple[str, float]],
              x_col: str = 'Elongation',
              y_col: str = 'Stress (MPa)',
              title: str = 'Tensile Stress-Strain Curves',
              x_label: str = 'Elongation (%)',
              y_label: str = 'Engineering Stress (MPa)',
              palette: Optional[str] = 'viridis',
              show_markers: bool = True,
              markevery: int = 1200,
              figsize: Tuple[float,float] = (14,10),
              dpi: int = 300,
              save_path: Optional[Union[str, Path]] = None,
              ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    '''
    Plots individual specimen curves grouped together by formulation.
    '''
    
    if ax is None:
        fig,ax = plt.subplots(figsize = figsize, dpi = dpi)
    else:
        fig = ax.get_figure()
        
    names = [name for name,_ in sample_thickness_pairs[:len(df_list)]]
    unique_names = list(dict.fromkeys(names))
    
    cmap = plt.get_cmap(palette, len(unique_names))
    colormap = {name: cmap(i) for i,name in enumerate(unique_names)}
    marker_map = {name: marker_list[i % len(marker_list)] for i,name in enumerate(unique_names)}
    
    seen_labels = set()
    
        
    for df,name in zip(df_list,names):
        if x_col not in df.columns or y_col not in df.columns:
            continue
    
        color = colormap[name]
        label = name if name not in seen_labels else None
        marker = marker_map[name]
        seen_labels.add(name)
        
        kwargs = {'color': color, 'linewidth': 1.5, 'alpha': 0.8}
        if show_markers:
            kwargs.update({'marker': marker, 'markevery': markevery, 'markersize': 5})
            
        ax.plot(df[x_col], df[y_col], label = label, **kwargs)              
        
    ax.set_title(title, fontsize = 12, fontweight = 'bold', pad = 12)
    ax.set_xlabel(x_label, fontsize = 11, labelpad = 8)
    ax.set_ylabel(y_label, fontsize = 11, labelpad = 8)
    ax.legend(
        title = 'Sample group',
        frameon = True,
        facecolor = 'white',
        framealpha = 0.9,
        bbox_to_anchor = (1.02, 1),
        loc = 'upper left'
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi = dpi, bbox_inches = 'tight')
        
    return fig, ax


def plot_summary_data(df_list: Sequence[pd.DataFrame],
              sample_thickness_pairs: Sequence[Tuple[str, float]],
              x_col: str = 'Elongation',
              y_col: str = 'Stress (MPa)',
              title: str = 'Sample average stress-strain (Mean ± STD)',
              x_label: str = 'Elongation (%)',
              y_label: str = 'Engineering Stress (MPa)',
              palette: Optional[str] = 'viridis',
              show_markers: bool = True,
              markevery: int = 200,
              figsize: Tuple[float,float] = (14,10),
              dpi: int = 300,
              num_interp_points: int = 1000,
              save_path: Optional[Union[str, Path]] = None,
              ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    '''
    Plots summary data with shaded bands to display standard deviations.
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()
    
    names = [name for name, _ in sample_thickness_pairs[:len(df_list)]]
    grouped_dfs = {}
    
    for df,name in zip(df_list,names):
        if x_col in df.columns and y_col in df.columns:
            grouped_dfs.setdefault(name, []).append(df)

    cmap = plt.get_cmap(palette, len(df_list))

    for idx, (sample_name, test_list) in enumerate(grouped_dfs.items()):
        color = cmap(idx)
        
        max_x = min(df[x_col].max() for df in test_list)
        grid_x = np.linspace(0, max_x, num_interp_points)
        
        interp_y_list = [np.interp(max_x, df[x_col], df[y_col]) for df in test_list]
        
        y_matrix = np.vstack(interp_y_list)
        mean_y = np.mean(y_matrix, axis = 0)
        std_y = np.std(y_matrix, axis = 0)
        
        ax.plot(grid_x, mean_y, color = color, linewidth = 2.0, label = sample_name)
        ax.fill_between(grid_x, mean_y - std_y, mean_y + std_y, color = color, alpha = 0.2, edgecolor = 'none')
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(x_label, fontsize=11, labelpad=8)
    ax.set_ylabel(y_label, fontsize=11, labelpad=8)
    #ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Sample Group', frameon=True, facecolor='white', framealpha=0.9, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plot_modulus_fit(
        df: pd.DataFrame,
        EGL: float,
        width: float,
        thickness: float,
        sample_name: str = 'Specimen',
        zoom_elastic: bool = True,
        zoom_max_strain: Optional[float] = None,
        figsize: Tuple[float, float] = (8, 5),
        dpi: int = 300,
        save_path: Optional[Union[str, Path]] = None,
        ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    '''
    Check plot to assess calculated modulus fit.
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    else:
        fig = ax.get_figure()
        
    df_adj, model_df, inters = adjust_df(df, EGL, width, thickness)

    x = df_adj['Elongation']
    x2p = df_adj['E.2%']
    y = df_adj['Stress (MPa)']
    
    modulus, elastic_knot, elastic_endpoint = find_modulus(x, model_df['y'], inters)
    
    remaining_knots = [k for k in inters if k > elastic_knot]
    second_knot = remaining_knots[0] if remaining_knots else float(x.quantile(0.05))
    
    yield_strain, yield_stress = offset_yield(x, x2p, model_df['y'], second_knot, modulus, elastic_endpoint)
    uy_strain, uy_stress = find_uyt(x, y)
    
    #drawing modulus curve from MARS calculations
    x_mod_max = elastic_endpoint[0] * 2.0 if elastic_endpoint[0] > 0 else float(x.quantile(0.15))
    x_mod = np.linspace(0,x_mod_max,100)
    y_mod = modulus * x_mod
    
    # Generate 0.2% offset yield line
    x_off = np.linspace(0.2, max(yield_strain * 1.3, 0.5), 100)
    y_off = modulus * (x_off - 0.2)
    
    # Plot curves
    ax.plot(x, y, label='Experimental Data', color='royalblue', alpha=0.7, linewidth=1.2)
    ax.plot(model_df['x'], model_df['y'], label='MARS Spline Fit', color='gray', linestyle=':', alpha=0.8)
    ax.plot(x_mod, y_mod, label=f"Young's Modulus ({modulus:.3f} MPa/%)", color='crimson', linestyle='--', linewidth=2)
    ax.plot(x_off, y_off, label='0.2% Offset Line', color='darkorange', linestyle='--', linewidth=1.5)
    
    # Plot key point markers
    ax.plot(elastic_endpoint[0], elastic_endpoint[1], 'ro', markersize=6, label='Elastic Limit Knot')
    ax.plot(yield_strain, yield_stress, 'go', markersize=6, label=f'0.2% Yield ({yield_strain:.1f}%, {yield_stress:.2f} MPa)')
    ax.plot(uy_strain, uy_stress, 'mo', markersize=6, label=f'Ultimate Yield ({uy_strain:.1f}%, {uy_stress:.2f} MPa)')

    ax.set_title(f'Modulus & Yield Fit Inspection — {sample_name}', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Elongation (%)', fontsize=11, labelpad=8)
    ax.set_ylabel('Engineering Stress (MPa)', fontsize=11, labelpad=8)
    
    if zoom_elastic:
        x_limit = zoom_max_strain if zoom_max_strain else max(elastic_endpoint[0] * 2.2, yield_strain * 1.8, uy_strain * 1.3, 10.0)
        y_limit = max(yield_stress * 1.8, uy_stress * 1.4, elastic_endpoint[1] * 1.8, 0.2)
        ax.set_xlim(-0.2, x_limit)
        ax.set_ylim(-0.02, y_limit)

    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='best',bbox_to_anchor=(1.05,1))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig, ax
    
    