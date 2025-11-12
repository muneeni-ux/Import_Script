#!/usr/bin/env python3
"""Debug script to test the import step by step"""

import sys
import mysql.connector

MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "admin"
MYSQL_DATABASE = "testdb"

print("Step 1: Test MySQL connection...")
try:
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    print("✓ Connected to MySQL")
    conn.close()
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    sys.exit(1)

print("\nStep 2: Test database creation...")
try:
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    conn.commit()
    print(f"✓ Database '{MYSQL_DATABASE}' exists")
    cur.close()
    conn.close()
except Exception as e:
    print(f"✗ Failed to create database: {e}")
    sys.exit(1)

print("\nStep 3: Test table creation...")
try:
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cur = conn.cursor()
    CREATE_STMT = r"""CREATE TABLE IF NOT EXISTS `q1_data` (
      id INT AUTO_INCREMENT PRIMARY KEY,
      `Row_ID` INT,
      `Order_ID` VARCHAR(50),
      `Order_Date` VARCHAR(50),
      `Ship_Date` VARCHAR(50),
      `Ship_Mode` VARCHAR(50),
      `Customer_ID` VARCHAR(50),
      `Customer_Name` VARCHAR(50),
      `Segment` VARCHAR(50),
      `Country` VARCHAR(50),
      `City` VARCHAR(50),
      `State` VARCHAR(50),
      `Postal_Code` DOUBLE,
      `Region` VARCHAR(50),
      `Product_ID` VARCHAR(50),
      `Category` VARCHAR(50),
      `Sub_Category` VARCHAR(50),
      `Product_Name` VARCHAR(152),
      `Sales` DOUBLE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""
    cur.execute(CREATE_STMT)
    conn.commit()
    print("✓ Table 'q1_data' created or already exists")
    cur.close()
    conn.close()
except Exception as e:
    print(f"✗ Failed to create table: {e}")
    sys.exit(1)

print("\nStep 4: Test row insertion...")
try:
    import pandas as pd
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        autocommit=False
    )
    cur = conn.cursor()
    
    # Test insert with a single row
    sql = """INSERT INTO `q1_data` 
            (`Row_ID`, `Order_ID`, `Order_Date`, `Ship_Date`, `Ship_Mode`, 
             `Customer_ID`, `Customer_Name`, `Segment`, `Country`, `City`, `State`, 
             `Postal_Code`, `Region`, `Product_ID`, `Category`, `Sub_Category`, `Product_Name`, `Sales`) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    
    test_row = (
        1, "CA-2017-152156", "08/11/2017", "11/11/2017", "Second Class",
        "CG-12520", "Claire Gute", "Consumer", "United States", "Henderson",
        "Kentucky", 42420.0, "South", "FUR-BO-10001798", "Furniture", "Bookcases",
        "Bush Somerset Collection Bookcase", 261.96
    )
    
    cur.execute(sql, test_row)
    conn.commit()
    print("✓ Test row inserted successfully")
    
    # Verify it's there
    cur.execute("SELECT COUNT(*) FROM q1_data")
    count = cur.fetchone()[0]
    print(f"  Total rows in table: {count}")
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"✗ Failed to insert: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
