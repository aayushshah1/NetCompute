import threading
import time
import os
import sys

def run_dashboard():
    os.system("python dashboard.py")

def run_master():
    # Wait for dashboard to start up
    time.sleep(2)
    os.system("python master.py")

if __name__ == "__main__":
    # Start dashboard in a separate thread
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    # Start master in the main thread
    run_master()