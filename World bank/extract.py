# ============================================
# extract.py - EXTRACT data from World Bank API
# ============================================
# This script calls the World Bank API and returns
# raw data as a pandas DataFrame.

import requests
import pandas as pd
from config import COUNTRIES, INDICATORS, START_YEAR, END_YEAR, API_BASE_URL


def extract_indicator(indicator_code, indicator_name):
    """
    Extracts data for ONE indicator from the World Bank API
    for all configured countries and date range.
    
    Parameters:
        indicator_code: str - World Bank indicator code (e.g., "NY.GDP.MKTP.CD")
        indicator_name: str - Friendly name we give it (e.g., "gdp_current_usd")
    
    Returns:
        pandas DataFrame with the raw data
    """
    # Build the list of countries separated by semicolon
    countries_str = ";".join(COUNTRIES)
    
    # Build the API URL
    url = (
        f"{API_BASE_URL}/country/{countries_str}"
        f"/indicator/{indicator_code}"
        f"?date={START_YEAR}:{END_YEAR}"
        f"&format=json"
        f"&per_page=10000"
    )
    
    print(f"📡 Extracting: {indicator_name} ({indicator_code})...")
    print(f"   URL: {url}")
    
    # Make the API call
    response = requests.get(url)
    
    # Check if the call was successful
    if response.status_code != 200:
        print(f"   ❌ Error: Status code {response.status_code}")
        return pd.DataFrame()
    
    # Parse the JSON response
    data = response.json()
    
    # World Bank API returns a list with 2 elements:
    # [0] = metadata (pagination info)
    # [1] = actual data
    if len(data) < 2 or data[1] is None:
        print(f"   ⚠️ No data returned for {indicator_name}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.json_normalize(data[1])
    
    # Add friendly indicator name
    df["indicator_name"] = indicator_name
    
    records = len(df)
    print(f"   ✅ Extracted {records} records")
    
    return df


def extract_all():
    """
    Extracts ALL configured indicators and combines them
    into a single DataFrame.
    
    Returns:
        pandas DataFrame with all raw data
    """
    print("=" * 50)
    print("🚀 STARTING EXTRACTION")
    print("=" * 50)
    
    all_data = []
    
    for code, name in INDICATORS.items():
        df = extract_indicator(code, name)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("❌ No data was extracted!")
        return pd.DataFrame()
    
    # Combine all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("=" * 50)
    print(f"✅ EXTRACTION COMPLETE: {len(combined_df)} total records")
    print(f"   Indicators: {len(INDICATORS)}")
    print(f"   Countries: {len(COUNTRIES)}")
    print(f"   Date range: {START_YEAR}-{END_YEAR}")
    print("=" * 50)
    
    return combined_df


# If you run this file directly, it extracts and shows a preview
if __name__ == "__main__":
    df = extract_all()
    if not df.empty:
        print("\n📋 Preview of extracted data:")
        print(df.head(10))
        print(f"\n📊 Columns: {list(df.columns)}")
