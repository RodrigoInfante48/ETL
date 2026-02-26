"""
* Description: Sends an HTML email via Gmail SMTP with charts embedded
*              in the body and a CSV attachment.
* Project: World Bank Economic Reporting
* Author: Rodrigo Infante
* Modified: 2026-02-24
* Dependencies:
*   - smtplib, email.mime (standard library)
*   - config.py (EMAIL_CONFIG, REPORT_CONFIG)
*
* Notes:
*   - Requires Gmail App Password (NOT your regular Gmail password).
*   - HTML body with base64-embedded charts — no external image hosting needed.
*   - Create App Password: https://myaccount.google.com/apppasswords
"""

from __future__ import annotations

import os
import smtplib
from datetime import date

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import EMAIL_CONFIG, REPORT_CONFIG


def send_email(html_body: str, attachment_path: str) -> bool:
    """
    Send an HTML email with embedded charts and a CSV attachment.

    Parameters:
        html_body: Full HTML string (charts embedded as base64)
        attachment_path: Path to CSV file to attach

    Returns:
        True if sent successfully, False otherwise.
    """
    print("📧 Sending email...")

    sender    = EMAIL_CONFIG["sender_email"]
    password  = EMAIL_CONFIG["sender_password"]
    recipients = EMAIL_CONFIG["recipients"]

    if not recipients:
        print("   ❌ No recipients configured.")
        return False

    # Build message
    msg = MIMEMultipart("related")
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = f"{REPORT_CONFIG['email_subject']} — {date.today()}"

    # Attach HTML body
    msg_alternative = MIMEMultipart("alternative")
    msg.attach(msg_alternative)
    msg_alternative.attach(MIMEText(html_body, "html", "utf-8"))

    # Attach CSV
    if os.path.isfile(attachment_path):
        try:
            with open(attachment_path, "rb") as f:
                attachment = MIMEBase("application", "octet-stream")
                attachment.set_payload(f.read())
            encoders.encode_base64(attachment)
            attachment.add_header(
                "Content-Disposition",
                f"attachment; filename={REPORT_CONFIG['csv_filename']}",
            )
            msg.attach(attachment)
            print(f"   📎 Attached: {REPORT_CONFIG['csv_filename']}")
        except Exception as e:
            print(f"   ⚠️ Could not attach CSV: {e}")
    else:
        print(f"   ⚠️ CSV not found at {attachment_path}, sending without attachment.")

    # Send via Gmail SMTP
    try:
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"   ✅ Email sent to: {', '.join(recipients)}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("   ❌ Authentication failed!")
        print("   💡 Use a Gmail App Password: https://myaccount.google.com/apppasswords")
        return False

    except Exception as e:
        print(f"   ❌ Error sending email: {type(e).__name__}: {e}")
        return False
