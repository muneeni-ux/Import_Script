#!/usr/bin/env python3
import sys
import mysql.connector
from mysql.connector import errorcode

print("Testing MySQL connector with detailed output...")
print()

config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin'
}

print(f"Config: {config}")
print()

try:
    print("Connecting...")
    conn = mysql.connector.connect(**config)
    print("✓ Connected!")
    
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    print(f"MySQL Version: {version}")
    cur.close()
    conn.close()
    
except mysql.connector.Error as err:
    print(f"ERROR {err.errno}: {err.msg}")
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
