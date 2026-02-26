# ============================================
# load.py - LOAD data into PostgreSQL
# ============================================
# This script takes the clean DataFrame and loads
# it into a PostgreSQL table.

import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_URL


def load(df):
    """
    Loads a clean DataFrame into PostgreSQL.
    
    Parameters:
        df: pandas DataFrame - Clean data from transform.py
    
    What it does:
        1. Connects to PostgreSQL
        2. Creates/replaces the table 'economic_indicators'
        3. Loads all the data
        4. Verifies the load was successful
    """
    print("=" * 50)
    print("📦 STARTING LOAD TO POSTGRESQL")
    print("=" * 50)
    
    # -----------------------------------------
    # Step 1: Create database connection
    # -----------------------------------------
    try:
        engine = create_engine(DB_URL)
        print(f"   📌 Step 1: Connected to PostgreSQL")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("   💡 Make sure PostgreSQL is running and config.py is correct")
        return False
    
    # -----------------------------------------
    # Step 2: Load DataFrame into PostgreSQL
    # -----------------------------------------
    table_name = "economic_indicators"
    
    try:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",  # Replaces table if it exists
            index=False,          # Don't save the DataFrame index
        )
        print(f"   📌 Step 2: Loaded {len(df)} rows into '{table_name}'")
    except Exception as e:
        print(f"   ❌ Load error: {e}")
        return False
    
    # -----------------------------------------
    # Step 3: Verify the load
    # -----------------------------------------
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            print(f"   📌 Step 3: Verified - {count} rows in PostgreSQL")
            
            # Show a sample
            sample = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
            print(f"\n   📋 Sample from PostgreSQL:")
            print(sample.to_string(index=False))
    except Exception as e:
        print(f"   ⚠️ Verification error: {e}")
    
    print("=" * 50)
    print(f"✅ LOAD COMPLETE: {len(df)} rows loaded to '{table_name}'")
    print(f"   🔌 Connect Power BI to: localhost > world_bank_db > {table_name}")
    print("=" * 50)
    
    engine.dispose()
    return True


# If you run this file directly with sample data
if __name__ == "__main__":
    from extract import extract_all
    from transform import transform
    
    raw = extract_all()
    if not raw.empty:
        clean = transform(raw)
        load(clean)
