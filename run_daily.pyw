"""
Run Daily Check - รันวันละครั้งตอนเปิดคอม
เช็ค log ว่าวันนี้รันแล้วหรือยัง
"""
import os
import sys
from datetime import datetime

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "run_log.txt")

def already_ran_today():
    """เช็คว่าวันนี้รันแล้วหรือยัง"""
    if not os.path.exists(LOG_FILE):
        return False
    
    try:
        with open(LOG_FILE, 'r') as f:
            last_run = f.read().strip()
        
        today = datetime.now().strftime('%Y-%m-%d')
        return last_run == today
    except:
        return False

def log_today():
    """บันทึกว่าวันนี้รันแล้ว"""
    today = datetime.now().strftime('%Y-%m-%d')
    with open(LOG_FILE, 'w') as f:
        f.write(today)

def main():
    # เช็คว่าวันนี้รันแล้วหรือยัง
    if already_ran_today():
        print(f"Already ran today. Skipping...")
        return
    
    # Change to script directory
    os.chdir(SCRIPT_DIR)
    
    # Run daily check
    try:
        from daily_check_v2 import check_today_v2
        result = check_today_v2()
        
        # บันทึก log ว่ารันแล้ว
        log_today()
        print(f"Daily check completed and logged.")
        
    except Exception as e:
        # Log error
        error_log = os.path.join(SCRIPT_DIR, "error_log.txt")
        with open(error_log, 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
