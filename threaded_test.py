#!/usr/bin/env python3
import sys
import threading
import time

connected = False
error_msg = None

def connect_thread():
    global connected, error_msg
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='admin',
            connection_timeout=3
        )
        connected = True
        conn.close()
    except Exception as e:
        error_msg = str(e)

print("Attempting MySQL connection with 5-second timeout...")

thread = threading.Thread(target=connect_thread, daemon=True)
thread.start()
thread.join(timeout=5)

if connected:
    print("✓ Successfully connected to MySQL!")
    sys.exit(0)
elif error_msg:
    print(f"✗ Connection failed: {error_msg}")
    sys.exit(1)
else:
    print("✗ Connection attempt timed out after 5 seconds")
    print("\nPossible issues:")
    print("1. MySQL server is not running")
    print("2. Username/password is incorrect")
    print("3. MySQL is not listening on localhost:3306")
    print("4. Firewall is blocking the connection")
    sys.exit(1)
