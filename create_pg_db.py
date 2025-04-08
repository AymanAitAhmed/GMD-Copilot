import csv
import psycopg2
import csv
from psycopg2.extras import execute_batch


# Configuration
db_host = '127.0.0.1'
db_port = 5432
db_user = 'gmd'
db_password = '1234'
db_name = 'gmd_iot'
csv_file = '0'

def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


def insert_data(cursor, table_name, rows, headers):
    placeholders = ','.join(['%s'] * len(headers))
    columns = '"' + '", "'.join(headers) + '"'
    sql = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'

    execute_batch(cursor, sql, rows)

try:
    # 1. Connect to PostgreSQL (create the database if it doesn't exist)
    conn = psycopg2.connect(host=db_host, database=db_name, port=db_port, user=db_user, password=db_password)
    print("connected")
    conn.autocommit = True  # Important: Set autocommit to True for creating the database

    cur = conn.cursor()

    # Check if the database exists. If not, create it.
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
    db_exists = cur.fetchone()

    if not db_exists:
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Database '{db_name}' created successfully.")
    else:
        print(f"Database '{db_name}' already exists.")

    conn.close()  # Close the initial connection

    # 2. Connect to the newly created (or existing) database
    conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
    cur = conn.cursor()

    headers, rows = read_csv(csv_file)
    types = ["INTEGER","INTEGER","VARCHAR(29)","VARCHAR(29)"]
    # 4. Create the table (adapt the column names and data types as needed)
    table_name = "table_states"  # Choose a name for your table

    # Construct the CREATE TABLE statement dynamically based on CSV columns
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    for col_name, col_type in zip(headers, types):

        create_table_query += f'"{col_name}" {col_type}, '

    create_table_query = create_table_query[:-2] + ")"  # Remove the trailing comma and space
    cur.execute(create_table_query)


    insert_data(cur,table_name,rows,headers)

    conn.commit()
    print(f"Table '{table_name}' created successfully.")

    conn.commit()
    print(f"Data from '{csv_file}' inserted into '{table_name}' successfully.")


    cur.close()
    conn.close()
    print("PostgreSQL database and table created and populated successfully!")

except psycopg2.Error as e:
    if conn:
        conn.rollback() # Rollback in case of error to avoid partial updates
    print(f"Error: {e}")
except Exception as e:
    if conn:
        conn.rollback()
    print(f"An unexpected error occurred: {e}")
finally:
    if conn:
        conn.close()
