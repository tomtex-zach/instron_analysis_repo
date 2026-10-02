#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:20:05 2025

@author: zachkaye

# # Filename:   raw_data_formatter.py
# # Package:    instron_analysis
# # Author:     Zach Kaye
# # Created:    
# # Modified:   26 August 2026
# # Python Version: 3.9
# # Description: Contains functions for running Instron_Code jupyter notebook, trims slack and analyzes modulus and max tenacity
# #
# # Copyright:  Copyright 2026, TomTex Inc., All rights reserved.
# # Requires:   pathlib, pandas

"""

from pathlib import Path
import pandas as pd


def parse_instron_raw(filepath: str | Path) -> list[pd.DataFrame]:
    '''
    Parse multi-specimen Instron data file without modifying files on disk.
    Handles files missing extension and comma method for adding tests.
    
    Returns list of individual specimen tests as DataFrames
    '''
    
    path = Path(filepath)
    if not path.exists() and path.with_suffix('.csv').exists():
        path = path.with_suffix('.csv')
        
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {filepath} or {path.with_suffix('.csv')}")
        
    with open(path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
        lines = f.readlines()
        
    specimen_dfs = []
    current_rows = []
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        
        parts = [p.strip() for p in line_str.split(',')]
        
        try:
            pos = float(parts[0])
            #Retrieve the last non-empty entry across shifted column blocks
            force_str = next(p for p in reversed(parts[1:]) if p != '')
            force = float(force_str)
            
            #A 0.00 in the position column signals a new test
            if pos == 0.00 and current_rows and current_rows[-1][0] != 0.00:
                df = pd.DataFrame(current_rows, columns = ['Position (mm)',
                                                           'Force (N)'])
                if len(df) > 4:
                    specimen_dfs.append(df)
                current_rows = []
                
            current_rows.append((pos, force))
        except (ValueError, StopIteration):
            continue
        
    if current_rows:
        df = pd.DataFrame(current_rows, columns = ['Position (mm)',
                                                   'Force (N)'])
        if len(df) > 4:
            specimen_dfs.append(df)
            
    return specimen_dfs


def parse_coupon_metadata(filepath: str | Path) -> list[tuple[str, float]]:
    '''
    Parse coupon metadata csv into sample name and thickness pairs.
    Handles comma-delimited replicate strings in column 'd'.
    
    Returns list of sample names paired with sample thickness
    '''
    
    path = Path(filepath)
    if not path.exists() and path.with_suffix('.csv').exists():
        path = path.with_suffix('.csv')
        
    coupon_df = pd.read_csv(path)
    pairs = []
    
    for _,row in coupon_df.iterrows():
        sample_name = str(row['sample_name'].strip())
        raw_d = str(row['d'].strip())
        
        thicknesses = [float(v.strip()) for v in raw_d.split(',') if v.strip()]
        for t in thicknesses:
            pairs.append((sample_name, t))
            
    return pairs


def load_and_process(data_filepath: str | Path, coupon_filepath: str | Path):
    '''
    Combined entry for test data and metadata.
    
    Returns list of test results in DataFrames and list of thickness pairs
    '''
    
    specimen_dfs = parse_instron_raw(data_filepath)
    thickness_pairs = parse_coupon_metadata(coupon_filepath)
    
    return specimen_dfs, thickness_pairs