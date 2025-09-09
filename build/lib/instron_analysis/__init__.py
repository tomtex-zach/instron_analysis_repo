#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:30:05 2025

This package contains functions for formatting instron data files, adjusting them
by calculating Elongation (%), Load (MPa) etc and calculating properties such as 
Youngs modulus, yield stress/strain, max elongation/tenacity and toughness.

A quick description of the functions:
    
    format_file -- input filepath of raw data, file is duplicated as the "backup"
        and data file is replaced correct format for the rest of the code.
    process -- input filepath to formatted data file and coupon metadata and
        get list of DataFrames for each individual test, as well as list of tuples
        linking sample names to their thicknesses.
    mars_model, trim_end, trim, find_modulus, offset_yield all help to calculate
        properties. These are used in the following functions and probably have no
        use being used separately.
    adjust_df -- adds columns for Elongation and Load and applies trimming functions
    analyze -- input adjusted DataFrame and other coupon data to get table of
        physical readouts
    data_table -- input list of DataFrames (all tests in session) to get results
        from each test, the averages and standard deviations.

@author: zachkaye
"""

from .raw_data_formatter import(
        format_file,
        process
    )

from .property_calculator import(
        mars_model,
        trim_end,
        trim,
        find_modulus,
        offset_yield,
        adjust_df,
        analyze,
        data_table
    )

from .full_plot import(
        plot_data
    )