# ============================================
# transform.py - TRANSFORM the raw data
# ============================================
# This script takes the raw extracted data and:
# 1. Selects only the columns we need
# 2. Renames columns to clean names
# 3. Fixes data types
# 4. Removes null values
# 5. Pivots indicators into separate columns

import pandas as pd


def transform(raw_df):
    """
    Transforms raw World Bank API data into a clean,
    analysis-ready DataFrame.
    
    Parameters:
        raw_df: pandas DataFrame - Raw data from extract.py
    
    Returns:
        pandas DataFrame - Clean, transformed data
    """
    print("=" * 50)
    print("🔧 STARTING TRANSFORMATION")
    print("=" * 50)
    
    # -----------------------------------------
    # Step 1: Select only the columns we need
    # -----------------------------------------
    columns_needed = [
        "country.id",        # Country ISO code (e.g., "CO")
        "country.value",     # Country name (e.g., "Colombia")
        "date",              # Year
        "value",             # The actual data value
        "indicator_name",    # Our friendly name
    ]
    
    df = raw_df[columns_needed].copy()
    print(f"   📌 Step 1: Selected {len(columns_needed)} columns")
    
    # -----------------------------------------
    # Step 2: Rename columns to clean names
    # -----------------------------------------
    df = df.rename(columns={
        "country.id": "country_code",
        "country.value": "country_name",
        "date": "year",
        "value": "value",
        "indicator_name": "indicator",
    })
    print("   📌 Step 2: Renamed columns")
    
    # -----------------------------------------
    # Step 3: Fix data types
    # -----------------------------------------
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    print("   📌 Step 3: Fixed data types")
    
    # -----------------------------------------
    # Step 4: Remove rows with null values
    # -----------------------------------------
    rows_before = len(df)
    df = df.dropna(subset=["value"])
    rows_removed = rows_before - len(df)
    print(f"   📌 Step 4: Removed {rows_removed} null rows")
    
    # -----------------------------------------
    # Step 5: Pivot - each indicator becomes a column
    # -----------------------------------------
    df_pivot = df.pivot_table(
        index=["country_code", "country_name", "year"],
        columns="indicator",
        values="value",
    ).reset_index()
    
    # Flatten column names (remove multi-level index)
    df_pivot.columns.name = None
    
    print(f"   📌 Step 5: Pivoted to wide format")
    
    # -----------------------------------------
    # Step 6: Sort by country and year
    # -----------------------------------------
    df_pivot = df_pivot.sort_values(["country_name", "year"]).reset_index(drop=True)
    print(f"   📌 Step 6: Sorted data")
    
    # -----------------------------------------
    # Step 7: Round numeric values
    # -----------------------------------------
    numeric_cols = df_pivot.select_dtypes(include=["float64"]).columns
    df_pivot[numeric_cols] = df_pivot[numeric_cols].round(2)
    print(f"   📌 Step 7: Rounded numeric values")
    
    print("=" * 50)
    print(f"✅ TRANSFORMATION COMPLETE: {len(df_pivot)} rows, {len(df_pivot.columns)} columns")
    print(f"   Columns: {list(df_pivot.columns)}")
    print("=" * 50)
    
    return df_pivot


# If you run this file directly, it shows what the transformation does
if __name__ == "__main__":
    from extract import extract_all
    
    raw = extract_all()
    if not raw.empty:
        clean = transform(raw)
        print("\n📋 Preview of transformed data:")
        print(clean.head(10))
        print(f"\n📊 Shape: {clean.shape}")
