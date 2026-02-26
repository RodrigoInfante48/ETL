from urllib.parse import quote_plus

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "world_bank_db",
    "user": "----------------",
    "password": "---------------",
}

DB_URL = f"postgresql://{DB_CONFIG['user']}:{quote_plus(DB_CONFIG['password'])}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

EMAIL_CONFIG = {
    "sender_email": "------------------",
    "sender_password": "------------------",
    "recipients": ["------------------", "------------------"],
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
}
REPORT_CONFIG = {
    "csv_filename": "economic_indicators_report.csv",
    "email_subject": "📊 World Bank Economic Report",
}
