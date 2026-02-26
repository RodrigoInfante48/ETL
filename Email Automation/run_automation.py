# ============================================
# run_automation.py - RUN THE EMAIL AUTOMATION
# ============================================
# Two modes:
#   python run_automation.py            → Run once (send report now)
#   python run_automation.py --schedule → Run every Monday at 8:00 AM

import sys
import time
from datetime import datetime

# Optional: only import schedule if needed
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

from generate_report import generate_report
from send_email import send_email


def run_once():
    """Generate report and send email once"""
    start_time = datetime.now()
    
    print("\n" + "📧" * 25)
    print("   EMAIL REPORT AUTOMATION")
    print(f"   Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("📧" * 25 + "\n")
    
    # Step 1: Generate the report
    summary, csv_file = generate_report()
    
    # Step 2: Send the email
    success = send_email(summary, csv_file)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "🏁" * 25)
    if success:
        print("   ✅ AUTOMATION COMPLETE - Email sent!")
    else:
        print("   ⚠️ AUTOMATION COMPLETE - Email failed")
    print(f"   Duration: {duration:.1f} seconds")
    print("🏁" * 25 + "\n")
    
    return success


def run_scheduled():
    """Run the report every Monday at 8:00 AM"""
    if not SCHEDULE_AVAILABLE:
        print("❌ 'schedule' package not installed. Run: pip install schedule")
        return
    
    print("⏰ SCHEDULER ACTIVE")
    print("   Report will be sent every Monday at 8:00 AM")
    print("   Press Ctrl+C to stop\n")
    
    schedule.every().monday.at("08:00").do(run_once)
    
    # You can also use:
    # schedule.every().day.at("08:00").do(run_once)        # Every day
    # schedule.every().friday.at("17:00").do(run_once)     # Every Friday 5PM
    # schedule.every(2).hours.do(run_once)                 # Every 2 hours
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    if "--schedule" in sys.argv:
        run_scheduled()
    else:
        run_once()
