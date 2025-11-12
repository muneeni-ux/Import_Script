#!/usr/bin/env python3
import sys
import signal

def timeout_handler(signum, frame):
    print("TIMEOUT: Connection attempt took too long")
    sys.exit(1)

# Set a 5-second timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)

print("Importing mysql.connector...")
try:
    import mysql.connector
    print("✓ Imported")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("Creating connection config...")
config = {
    'host': '127.0.0.1',  # Try IP instead of hostname
    'user': 'root',
    'password': 'admin',
    'connection_timeout': 3,
    'autocommit': False
}
print(f"Config: {config}")

print("Attempting connection...")
try:
    conn = mysql.connector.connect(**config)
    print("✓ Connected!")
    conn.close()
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    signal.alarm(0)  # Cancel alarm
