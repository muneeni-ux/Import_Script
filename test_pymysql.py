#!/usr/bin/env python3
"""Test PyMySQL connection"""
import sys
import pymysql

print("Connecting with PyMySQL...")
print("Config: host='127.0.0.1', user='root', password='admin'")
print()

try:
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='admin'
    )
    print("✓ Connected!")
    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    print(f"MySQL Version: {version}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
