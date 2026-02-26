# 🌍 World Bank ETL Pipeline

ETL pipeline that extracts economic indicators from the World Bank API, transforms the data with Python (pandas), and loads it into PostgreSQL.

## 🏗️ Architecture

```
World Bank API → Python (Extract) → pandas (Transform) → PostgreSQL (Load) → Power BI (Visualize)
```

## 📊 Indicators Extracted

| Code | Indicator |
|------|-----------|
| NY.GDP.MKTP.CD | GDP (current US$) |
| SP.POP.TOTL | Population, total |
| FP.CPI.TOTL.ZG | Inflation, consumer prices (annual %) |
| SL.UEM.TOTL.ZS | Unemployment (% of total labor force) |
| NY.GDP.PCAP.CD | GDP per capita (current US$) |

## 🌎 Countries

Latin America focus: Colombia, Brazil, Mexico, Argentina, Chile, Peru, Ecuador + USA, China, Germany as global benchmarks.

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** - Data transformation
- **requests** - API extraction
- **sqlalchemy** - PostgreSQL connection
- **seaborn / matplotlib** - Exploratory analysis
- **PostgreSQL** - Data warehouse
- **Power BI** - Stakeholder dashboards

## 📁 Project Structure

```
world-bank-etl/
├── README.md
├── requirements.txt
├── config.py              # Database configuration
├── extract.py             # Extract data from World Bank API
├── transform.py           # Clean and transform data
├── load.py                # Load data into PostgreSQL
├── run_pipeline.py        # Run full ETL pipeline
└── explore.py             # Exploratory analysis with seaborn
```

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/world-bank-etl.git
cd world-bank-etl
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup PostgreSQL
Create a database called `world_bank_db`:
```sql
CREATE DATABASE world_bank_db;
```

### 4. Configure database connection
Edit `config.py` with your PostgreSQL credentials.

### 5. Run the pipeline
```bash
python run_pipeline.py
```

## 📈 Connect Power BI
1. Open Power BI Desktop
2. Get Data → PostgreSQL database
3. Server: `localhost`, Database: `world_bank_db`
4. Select the `economic_indicators` table
5. Build your dashboard!

## Author
Built by [YOUR NAME] as a portfolio ETL project.
