"""
HydroAnomaly: Simple Water Data Analysis Package

A simple Python package with just 3 modules:
1. USGS turbidity data retrieval (returns data and site coordinates)
2. Sentinel satellite bands retrieval  
3. Time series visualization
4. Machine learning anomaly detection (One-Class SVM and Isolation Forest)

That's it - nothing else!
"""

__version__ = "1.2.9"
__author__ = "Ehsan Kahrizi (Ehsan.kahrizi@usu.edu)"

# Import the 3 simple modules
from .usgs_turbidity import get_turbidity, get_usgs_turbidity
from .sentinel_bands import get_sentinel_bands, get_satellite_data, get_sentinel, get_sentinel_bands_gee, show_sentinel_ndwi_map
from .visualize import plot_timeseries, plot_turbidity, plot_sentinel, plot_comparison, plot, visualize
from .ml import run_oneclass_svm, run_isolation_forest


# Export everything
__all__ = [
    # USGS turbidity functions
    'get_turbidity',
    'get_usgs_turbidity',
    
    # Sentinel functions
    'get_sentinel_bands_gee',
    'get_sentinel_bands',
    'get_satellite_data', 
    'get_sentinel',
    'show_sentinel_ndwi_map',

    # Visualization functions
    'plot_timeseries',
    'plot_turbidity',
    'plot_sentinel', 
    'plot_comparison',
    'plot',
    'visualize',
    
    # Machine learning functions
    'run_oneclass_svm',
    'run_isolation_forest'  
]

print(f"HydroAnomaly v{__version__} - Simple Water Data Package")
print("Available functions:")
print("   • get_turbidity() - Get USGS turbidity data and site coordinates")
print("   • get_sentinel_bands() - Get satellite data") 
print("   • plot_timeseries() - Visualize data")
print("Try: help(hydroanomaly.get_turbidity) for examples")
