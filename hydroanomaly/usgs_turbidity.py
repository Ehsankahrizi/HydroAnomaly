"""
Simple USGS Turbidity Data Retrieval

This module provides one simple function to get turbidity data from USGS stations.
That's it - nothing else!
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import numpy as np

# Function for retrive data ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_turbidity(site_number: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get turbidity data from a USGS station.
    
    Args:
        site_number (str): USGS site number (e.g., "294643095035200")
        start_date (str): Start date as "YYYY-MM-DD" 
        end_date (str): End date as "YYYY-MM-DD"
        
    Returns:
        tuple: (pd.DataFrame, (latitude, longitude)) or (empty DataFrame, (None, None)) if not found.
        * Note: pd.DataFrame: Time series data with datetime index and turbidity values
        
    Example:
        >>> data = get_turbidity("294643095035200", "2023-01-01", "2023-12-31")
        >>> print(f"Got {len(data)} turbidity measurements")
    """

    # --- Validate inputs ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print(f"Getting turbidity data for site {site_number}")
    print(f"Date range: {start_date} to {end_date}")
    
    # --- Retrieve site metadata (lat/lon) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    site_url = (
        f"https://waterservices.usgs.gov/nwis/site/"
        f"?sites={site_number}"
        f"&format=rdb")
    try:
        site_resp = requests.get(site_url, timeout=15)
        if site_resp.status_code != 200:
            print(f"Could not get site metadata: {site_resp.status_code}")
            lat, lon = None, None
        else:
            df_meta = pd.read_csv(StringIO(site_resp.text), sep="\t", comment="#")
            df_meta = df_meta.dropna(axis=1, how="all")
            lat, lon = None, None
            if not df_meta.empty:
                lat = float(df_meta["dec_lat_va"].iloc[0]) if "dec_lat_va" in df_meta.columns else None
                lon = float(df_meta["dec_long_va"].iloc[0]) if "dec_long_va" in df_meta.columns else None
    except Exception as e:
        print(f"Error getting site coordinates: {e}")
        lat, lon = None, None
    
    
    # --- Retrieve turbidity data (Build USGS API URL for turbidity (parameter code 63680))------------------------------------------------------------------------------------------------------------------------------------------------------------------
    url = (
        f"https://waterservices.usgs.gov/nwis/iv/"
        f"?sites={site_number}"
        f"&parameterCd=63680"  # Turbidity parameter code
        f"&startDT={start_date}"
        f"&endDT={end_date}"
        f"&format=rdb")
    
    try:
        # Get data from USGS
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"No data found: API returned status {response.status_code}.")
            print("Data for the specified site or parameters does not exist.")
            return pd.DataFrame(), (lat, lon)
        
        # Parse the response
        data = _parse_usgs_response(response.text)
        
        if len(data) == 0:
            print("No data found for the specified parameters or date range.")
            return pd.DataFrame(), (lat, lon)
        
        print(f"Retrieved {len(data)} turbidity measurements")
        return data, (lat, lon)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Data for the specified site or parameters does not exist.")
        return pd.DataFrame(), (lat, lon)


# Function for parse and cleaning Turbidity Time Series from USGS API Respons as DataFrame ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _parse_usgs_response(content: str) -> pd.DataFrame:
    """Parse USGS response and extract turbidity data."""
    
    if "No sites found" in content or "No data" in content:
        return pd.DataFrame()
    
    try:
        # Read tab-separated data
        data = pd.read_csv(StringIO(content), sep='\t', comment='#')
        
        # Clean up
        data = data.dropna(axis=1, how='all')
        data.columns = data.columns.str.strip()
        
        # Find datetime and turbidity columns
        datetime_cols = [col for col in data.columns if 'datetime' in col.lower()]
        turbidity_cols = [col for col in data.columns if '63680' in col]
        
        if not datetime_cols or not turbidity_cols:
            return pd.DataFrame()
        
        # Extract relevant columns
        result = data[[datetime_cols[0], turbidity_cols[0]]].copy()
        result.columns = ['datetime', 'turbidity']
        
        # Convert data types
        result['datetime'] = pd.to_datetime(result['datetime'], errors='coerce')
        result['turbidity'] = pd.to_numeric(result['turbidity'], errors='coerce')
        
        # Remove missing data
        result = result.dropna()
        
        # Set datetime as index
        result = result.set_index('datetime')
        
        return result
        
    except Exception:
        return pd.DataFrame()


# Simple alias for backwards compatibility
get_usgs_turbidity = get_turbidity
