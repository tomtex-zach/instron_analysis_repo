#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
instron_analysis package
"""

from .raw_data_formatter import (
    parse_instron_raw,
    parse_coupon_metadata,
    load_and_process
)

from .property_calculator import (
    xy_pointFinder,
    mars_model,
    find_uyt,
    trim_end,
    trim,
    find_modulus,
    offset_yield,
    adjust_df,
    analyze,
    data_table
)

from .full_plot import (
    plot_data,
    plot_summary_data,
    plot_modulus_fit
)

__version__ = "0.2.0"

__all__ = [
    # Data Ingestion & File Parsing
    "parse_instron_raw",
    "parse_coupon_metadata",
    "load_and_process",
    # Mechanical Property Calculations
    "xy_pointFinder"
    "mars_model",
    "find_uyt",
    "trim_end",
    "trim",
    "find_modulus",
    "offset_yield",
    "adjust_df",
    "analyze",
    "data_table",
    # Plotting & Visualizations
    "plot_data",
    "plot_summary_data",
    "plot_modulus_fit",
]