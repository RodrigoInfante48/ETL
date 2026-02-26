# 📧 Email Report Automation

Automated Python script that queries PostgreSQL, generates a summary report, and sends it via email with a CSV attachment.

## 🏗️ Architecture

```
PostgreSQL → Python (Query + Report) → Email with CSV attachment → Recipients
```

## 🔧 What It Does

1. Connects to PostgreSQL and pulls the latest economic data
2. Generates a summary report (top GDP countries, highest inflation, etc.)
3. Exports the data as a CSV file
4. Sends an email with the report + CSV attached
5. Can be scheduled to run automatically (daily, weekly, monthly)

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** - Data querying and manipulation
- **sqlalchemy** - PostgreSQL connection
- **smtplib** - Email sending (built-in Python)
- **schedule** - Task scheduling

## 📁 Project Structure

```
email-automation/
├── README.md
├── requirements.txt
├── config.py              # Email and DB configuration
├── generate_report.py     # Query DB and create report
├── send_email.py          # Send email with attachment
├── run_automation.py      # Main script (manual or scheduled)
```

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/email-automation.git
cd email-automation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Gmail App Password
You need a Gmail App Password (NOT your regular password):
1. Go to https://myaccount.google.com/apppasswords
2. Generate a new app password for "Mail"
3. Copy the 16-character password
4. Paste it in `config.py`

### 4. Configure settings
Edit `config.py` with your email and database credentials.

### 5. Run manually
```bash
python run_automation.py
```

### 6. Run on schedule (optional)
```bash
python run_automation.py --schedule
```

## 📬 Sample Email Output

**Subject:** 📊 World Bank Economic Report - 2026-02-19

**Body:**
- Top 3 GDP countries with latest values
- Countries with highest inflation
- Unemployment comparison
- CSV file attached with full dataset

## Author
Built by [YOUR NAME] as a portfolio automation project.
