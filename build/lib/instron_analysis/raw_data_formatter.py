#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:20:05 2025

@author: zachkaye

# # Filename:   raw_data_formatter.py
# # Package:    instron_analysis
# # Author:     Zach Kaye
# # Created:    
# # Modified:   28 August 2025
# # Python Version: 3.9
# # Description: Contains functions for running Instron_Code jupyter notebook, trims slack and analyzes modulus and max tenacity
# #
# # Copyright:  Copyright 2025, TomTex Inc., All rights reserved.
# # Requires:   tempfile, csv, shutil

"""

from tempfile import NamedTemporaryFile
import csv
import shutil
import pandas as pd


def format_file(filename):
    
    '''
    Function takes in the raw data file from the Instron machine.
    This code is built for raw data format as of August 2025. Older tests
    will not work in this code. No returns.
    '''
    
    data_filename = filename
    backup_filename = 'backup_' + filename
    
    tempFile = NamedTemporaryFile('w+t', newline= '', delete = False)
          
    try:
        shutil.copy(data_filename, backup_filename)
        print(f'Original file {data_filename} duplicated to {backup_filename}')
    except FileNotFoundError:
        print(f'Error: Original file {data_filename} not found')
        exit()
        
    with open(data_filename, 'r', newline= '') as csvFile, tempFile:        
        reader = csv.reader(csvFile, delimiter = ',', quotechar = '"')
        writer = csv.writer(tempFile, delimiter = ',', quotechar = '"')
        rows = list(reader)
    
        count = 0
        for i,row in enumerate(rows):
            if i > 0 and row[0] == rows[i-1][0]:
                pass
            elif row[0] == '0.00':
                writer.writerow([f'test number: {count}'])
                writer.writerow([row[0], row[-1]])
                count += 1
            else:
                writer.writerow([row[0],row[-1]])
    
    shutil.move(tempFile.name, data_filename)
    
    
def process(filename, coupon_filename):
    
    '''
    Initial processing of the raw data file and coupon data into workable
    DataFrames usable for the rest of the code. Returns list of DataFrames for
    individual tests and list of tuples containing each sample and their thickness.
    '''
    
    instron_data_filename = filename
    coupon_data_filename = coupon_filename
    
    test_data = pd.read_csv(instron_data_filename,
                            header = 2,
                            names = ['Position (mm)', 'Force (N)'])
    
    coupon_data = pd.read_csv(coupon_data_filename)
    
    coupon_data['d'] = coupon_data['d'].apply(lambda x: x.split(','))
    coupon_data['d'] = coupon_data['d'].apply(pd.to_numeric)
    
    sample_thickness_pairs = []
    for i,row in coupon_data.iterrows():
        for j in row['d']:
            sample_thickness_pairs.append([row['sample_name'], j])
            
    indx = [0]
    
    for i in test_data[test_data['Position (mm)'].str.contains('test number:')].index:
        indx.append(i)
    indx.append(test_data.index[-1])

    dfs = []
    for i in range(len(indx[:-1])):
        if i == 0:
            df = test_data[indx[i]:indx[i+1]].reset_index(drop = True)
            dfs.append(df)
        elif i == indx[-1]:
            df = test_data[indx[i]:].reset_index(drop = True)
            dfs.append(df)
        else:
            df = test_data[indx[i]+1:indx[i+1]].reset_index(drop = True)
            dfs.append(df)
            
    dfs = [df for df in dfs if len(df) > 4]
    
    return dfs, sample_thickness_pairs