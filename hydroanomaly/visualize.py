"""
Simple Time Series Visualization

This module provides simple functions to visualize time series data.
That's it - nothing else!
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================================
def plot_timeseries(data: pd.DataFrame, title: str = "Time Series Data", save_file: str = None) -> None:
    """
    Create a simple time series plot.
    
    Args:
        data (pd.DataFrame): DataFrame with datetime index and numeric columns
        title (str): Title for the plot
        save_file (str): Optional filename to save the plot
        
    Example:
        >>> plot_timeseries(turbidity_data, "Turbidity Over Time", "turbidity_plot.png")
    """
    
    if data.empty:
        print("No data to plot")
        return
    
    print(f"Creating plot: {title}")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot each column
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            plt.plot(data.index, data[column], label=column, linewidth=1.5, alpha=0.8)
    
    # Format plot
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add legend if multiple columns
    if len(data.columns) > 1:
        plt.legend()
    
    plt.tight_layout()
    
    # Save if requested
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_file}")
    
    plt.show()
    print("Plot created successfully!")

# ==============================================================================================
def plot_turbidity(turbidity_data: pd.DataFrame, save_file: str = None) -> None:
    """
    Create a turbidity-specific plot with appropriate formatting.
    
    Args:
        turbidity_data (pd.DataFrame): DataFrame with turbidity values
        save_file (str): Optional filename to save the plot
    """
    
    if turbidity_data.empty:
        print("No turbidity data to plot")
        return
    
    print("Creating turbidity plot")
    
    plt.figure(figsize=(12, 6))
    
    # Plot turbidity
    column_name = turbidity_data.columns[0]
    plt.plot(turbidity_data.index, turbidity_data.iloc[:, 0], 
             color='brown', linewidth=1.5, alpha=0.8)
    
    # Add threshold lines for water quality assessment
    plt.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Moderate (10 NTU)')
    plt.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='High (25 NTU)')
    
    # Format plot
    plt.title('Turbidity Time Series', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Turbidity (NTU)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.tight_layout()
    
    # Save if requested
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Turbidity plot saved as {save_file}")
    
    plt.show()
    print("Turbidity plot created!")

# ==============================================================================================
def plot_sentinel(sentinel_data: pd.DataFrame, save_file: str = None) -> None:
    """
    Create a Sentinel satellite data plot.
    
    Args:
        sentinel_data (pd.DataFrame): DataFrame with Sentinel band values
        save_file (str): Optional filename to save the plot
    """
    
    if sentinel_data.empty:
        print("No Sentinel data to plot")
        return
    
    print("Creating Sentinel bands plot")
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for different bands
    band_colors = {
        'B2': 'blue',    # Blue band
        'B3': 'green',   # Green band  
        'B4': 'red',     # Red band
        'B8': 'darkred', # NIR band
        'NDVI': 'darkgreen'
    }
    
    # Plot each band
    for column in sentinel_data.columns:
        color = band_colors.get(column, 'black')
        plt.plot(sentinel_data.index, sentinel_data[column], 
                label=column, color=color, linewidth=2, marker='o', markersize=4)
    
    # Format plot
    plt.title('Sentinel Satellite Data', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Digital Number / Index Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.tight_layout()
    
    # Save if requested
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Sentinel plot saved as {save_file}")
    
    plt.show()
    print("Sentinel plot created!")

# ==============================================================================================
def plot_comparison(data1: pd.DataFrame, data2: pd.DataFrame, 
                   label1: str = "Dataset 1", label2: str = "Dataset 2",
                   title: str = "Data Comparison", save_file: str = None) -> None:
    """
    Create a comparison plot of two time series datasets.
    
    Args:
        data1 (pd.DataFrame): First dataset
        data2 (pd.DataFrame): Second dataset
        label1 (str): Label for first dataset
        label2 (str): Label for second dataset
        title (str): Plot title
        save_file (str): Optional filename to save the plot
    """
    
    if data1.empty and data2.empty:
        print("No data to plot")
        return
    
    print(f"Creating comparison plot: {title}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot first dataset
    if not data1.empty:
        ax1.plot(data1.index, data1.iloc[:, 0], color='blue', linewidth=1.5, alpha=0.8)
        ax1.set_title(f'{label1}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value', fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # Plot second dataset  
    if not data2.empty:
        ax2.plot(data2.index, data2.iloc[:, 0], color='red', linewidth=1.5, alpha=0.8)
        ax2.set_title(f'{label2}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Value', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax2.set_xlabel('Date', fontsize=12)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as {save_file}")
    
    plt.show()
    print("Comparison plot created!")


# Simple aliases
plot = plot_timeseries
visualize = plot_timeseries
