"""
Auto-generated script to import the uploaded Q4 data into a MySQL table.
Replace the MySQL connection placeholders with your credentials, then run:

    python import_Q4_to_mysql.py

This script will:
 - create a table named "q4_data" (if not exists) inferred from the Excel headers,
 - insert all rows from the file "Q4 Dataset.csv" (sheet: (n/a)),
 - print progress.

Make sure you have installed:
    pip install pandas pymysql openpyxl
"""

import pandas as pd
import pymysql
import os

# === CONFIGURE THESE BEFORE RUNNING ===
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "admin"
MYSQL_DATABASE = "testdb"

FILE_PATH = r"C:\Users\dell\OneDrive\Dokumente\Data Analytics\Data Analytics\heart_disease_uci.csv"
SHEET_NAME = None
TABLE_NAME = "q4_data"

df = None  # Will be loaded in main()

# inferred CREATE TABLE statement
CREATE_STMT = r"""CREATE TABLE IF NOT EXISTS `q4_data` (
  pk INT AUTO_INCREMENT PRIMARY KEY,
  `id` INT,
  `age` INT,
  `sex` VARCHAR(50),
  `dataset` VARCHAR(50),
  `cp` VARCHAR(50),
  `trestbps` INT,
  `chol` INT,
  `fbs` VARCHAR(50),
  `restecg` VARCHAR(50),
  `thalch` INT,
  `exang` VARCHAR(50),
  `oldpeak` DECIMAL(10,2),
  `slope` VARCHAR(50),
  `ca` VARCHAR(50),
  `thal` VARCHAR(50),
  `num` INT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""

def load_data():
    """Load CSV or Excel file, with fallback to script directory."""
    global df, FILE_PATH
    
    if not os.path.exists(FILE_PATH):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible = [p for p in os.listdir(script_dir) if p.lower().endswith(('.csv', '.xlsx', '.xls'))]
        if possible:
            fallback = os.path.join(script_dir, possible[0])
            print(f"NOTE: configured FILE_PATH not found; using fallback: {fallback}")
            FILE_PATH = fallback
        else:
            raise FileNotFoundError(f"Configured FILE_PATH not found and no data files found in {script_dir}")
    
    if FILE_PATH.lower().endswith(".csv"):
        df = pd.read_csv(FILE_PATH)
    else:
        if SHEET_NAME:
            df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
        else:
            df = pd.read_excel(FILE_PATH)
    
    print(f"✓ Loaded data from {FILE_PATH} — shape={df.shape}")
    print(df.head(3).to_string(index=False))
    
    # sanitize column names
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    
    return df

def connect():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        autocommit=False
    )

def ensure_database_and_table():
    try:
        conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD)
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.select_db(MYSQL_DATABASE)
        # create table if not exists
        cur.execute(CREATE_STMT)
        conn.commit()
        print("✓ Database and table ready.")
        cur.close()
        conn.close()
    except pymysql.Error as e:
        print(f"Error setting up database/table: {e}")
        raise

def insert_rows():
    global df
    if df is None:
        print("ERROR: DataFrame is None. Data was not loaded.")
        return
    
    conn = connect()
    cur = conn.cursor()
    cols = df.columns.tolist()
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join([f"`{c}`" for c in cols])
    sql = f"INSERT INTO `{TABLE_NAME}` ({col_list}) VALUES ({placeholders})"
    print(f"Inserting {len(df)} rows...")

    # prepare values and convert NaN/empty strings to None
    values = []
    for row in df.itertuples(index=False, name=None):
        row_vals = []
        for i, x in enumerate(row):
            # Convert NaN and empty strings/whitespace to None
            if pd.isna(x) or (isinstance(x, str) and x.strip() == ''):
                row_vals.append(None)
            else:
                row_vals.append(x)
        values.append(tuple(row_vals))

    try:
        if not values:
            print("No rows to insert (empty dataframe).")
            return

        cur.executemany(sql, values)
        conn.commit()
        # rowcount is the number of affected rows
        inserted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(values)
        print(f"✓ Inserted {inserted} rows into {TABLE_NAME}")
    except pymysql.Error as e:
        conn.rollback()
        print(f"MySQL error during insert: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Q4 Data Import to MySQL")
    print("=" * 60)
    
    try:
        # Load data first
        df = load_data()
        
        # Prepare database and table
        print("\n✓ Preparing database and table...")
        ensure_database_and_table()
        
        # Insert data
        print()
        insert_rows()
        
        print("=" * 60)
        print("✓ Import complete!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)