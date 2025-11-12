#!/usr/bin/env python3
import sys
print("Testing MySQL connector...")

try:
    import mysql.connector
    print("✓ mysql.connector imported")
except ImportError as e:
    print(f"✗ Could not import mysql.connector: {e}")
    sys.exit(1)

print("\nAttempting to connect...")
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin"
    )
    print("✓ Connected to MySQL!")
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    print(f"✓ MySQL version: {version}")
    cur.close()
    conn.close()
except mysql.connector.Error as err:
    if err.errno == 2003:
        print(f"✗ Can't connect to MySQL Server on 'localhost' (errno {err.errno})")
        print("  Make sure MySQL server is running and accessible")
    elif err.errno == 1045:
        print(f"✗ Access denied for user 'root'@'localhost' (errno {err.errno})")
        print("  Check username and password")
    else:
        print(f"✗ MySQL error {err.errno}: {err.msg}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
