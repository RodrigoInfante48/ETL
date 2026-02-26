from urllib.parse import quote_plus

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "world_bank_db",
    "user": "postgres",
    "password": "4301077Reic.",
}

DB_URL = f"postgresql://{DB_CONFIG['user']}:{quote_plus(DB_CONFIG['password'])}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

EMAIL_CONFIG = {
    "sender_email": "roesinf2@gmail.com",
    "sender_password": "bvvo dgcp whkz ffzi",
    "recipients": ["roesinf2@gmail.com", "valentinaburbano2002@gmail.com"],
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
}
REPORT_CONFIG = {
    "csv_filename": "economic_indicators_report.csv",
    "email_subject": "📊 World Bank Economic Report",
}