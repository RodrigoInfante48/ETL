# ============================================
# config.py - Database Configuration
# ============================================
# Edit these values with your PostgreSQL credentials

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "world_bank_db",
    "user": "postgres",          # <-- Change to your PostgreSQL user
    "password": "4301077Reic."  # <-- Change to your PostgreSQL password
}

# Build the connection string for SQLAlchemy
DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# ============================================
# World Bank API Configuration
# ============================================

# Countries to extract (ISO 2-letter codes)
COUNTRIES = [
    "CO",  # Colombia
    "BR",  # Brazil
    "MX",  # Mexico
    "AR",  # Argentina
    "CL",  # Chile
    "PE",  # Peru
    "EC",  # Ecuador
    "US",  # United States
    "CN",  # China
    "DE",  # Germany
]

# Economic indicators to extract
INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "SP.POP.TOTL": "population_total",
    "FP.CPI.TOTL.ZG": "inflation_annual_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
}

# Date range
START_YEAR = 2000
END_YEAR = 2023

# API Base URL
API_BASE_URL = "https://api.worldbank.org/v2"
