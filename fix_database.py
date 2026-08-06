import sqlite3
import os
import csv

def fix():
    # 1. Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(base_dir, 'database')
    db_path = os.path.join(db_dir, 'properties.db')
    
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created directory: {db_dir}")

    print(f"Target Database: {db_path}")

    # 2. Connect and Create Table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Creating table 'properties'...")
    cursor.execute('DROP TABLE IF EXISTS properties')
    cursor.execute('''
    CREATE TABLE properties (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        property_type TEXT, property_name TEXT, location_area TEXT, location TEXT,
        total_price REAL, area_sqft REAL, price_sqft REAL, bhk INTEGER, bath INTEGER,
        face TEXT, outside_view TEXT, furnished_status TEXT, property_status TEXT,
        property_link TEXT, property_img_link TEXT, property_description TEXT,
        status TEXT DEFAULT 'available'
    )
    ''')
    conn.commit()

    # 3. Import Data
    csv_files = ['plots_dataset.csv', 'flats_dataset.csv', 'house_and_villa_dataset.csv']
    for file_name in csv_files:
        file_path = os.path.join(db_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    cursor.execute('INSERT INTO properties (property_type, location) VALUES (?, ?)', 
                                   (row.get('property_type', 'Unknown'), row.get('location', 'Unknown')))
                    count += 1
                print(f"Successfully imported {count} rows from {file_name}")
        else:
            print(f"Warning: {file_name} not found in {db_dir}")

    conn.commit()
    
    # 4. Final Verification
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='properties'")
    if cursor.fetchone():
        print("\n✅ SUCCESS: The 'properties' table now exists!")
    else:
        print("\n❌ ERROR: Table creation failed.")
    
    conn.close()

if __name__ == "__main__":
    fix()
