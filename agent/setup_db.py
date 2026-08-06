import sqlite3
import csv
import os
import logging

logger = logging.getLogger("setup_db")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# Values in a source CSV's status/availability column that mean the unit is
# already sold and should never be shown to a buyer.
SOLD_VALUES = {"sold", "sold out", "soldout", "booked", "not available", "unavailable", "closed"}


def safe_float(value, default=0.0):
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == '':
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_availability(value):
    """
    Maps whatever the source CSV uses for availability ('Sold', 'Booked',
    'Available', blank, etc.) into exactly 'sold' or 'available'. Defaults to
    'available' if the dataset doesn't have this column at all, or the value
    is unrecognized - so we never accidentally hide a property that's
    actually for sale.
    """
    if not value:
        return "available"
    v = str(value).strip().lower()
    return "sold" if v in SOLD_VALUES else "available"


def setup_database():
    # 1. Setup Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.getenv("DATABASE_DIR") or os.path.join(root_dir, 'database')
    db_path = os.path.join(db_dir, 'properties.db')

    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    logger.info("Target database: %s", db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")  # matches the runtime app's connection mode
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS properties')

        # Unified schema for all property types
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

        def import_csv(file_name, prop_type_default):
            file_path = os.path.join(db_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning("Skipping %s (not found)", file_name)
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Fail loudly instead of silently zeroing every row on a header mismatch
                logger.info("%s columns: %s", file_name, reader.fieldnames)
                expected_any = ['total_price', 'area(sq-ft)', 'area_sqft', 'location', 'property_location']
                if reader.fieldnames and not any(col in reader.fieldnames for col in expected_any):
                    logger.warning(
                        "None of the expected columns %s found in %s. Check for a column-name "
                        "mismatch before trusting this import.", expected_any, file_name
                    )

                data = []
                skipped_price = 0
                sold_count = 0
                for row in reader:
                    price = safe_float(row.get('total_price'))
                    area = safe_float(row.get('area(sq-ft)', row.get('area_sqft')))
                    if price == 0.0 and row.get('total_price') not in (None, '', '0'):
                        skipped_price += 1

                    # Respect an existing sold/availability column if the source
                    # CSV has one, under any of these common header names, so a
                    # property already sold is never imported as available.
                    raw_availability = (
                        row.get('status') or row.get('availability') or
                        row.get('sold_status') or row.get('is_sold') or
                        row.get('availability_status')
                    )
                    availability = normalize_availability(raw_availability)
                    if availability == "sold":
                        sold_count += 1

                    data.append((
                        row.get('property_type') or prop_type_default,
                        row.get('property', ''),
                        row.get('location_area', ''),
                        row.get('location') or row.get('property_location', ''),
                        price, area,
                        safe_float(row.get('price_sqft', row.get('price(sqft)'))),
                        safe_int(row.get('bhk')), safe_int(row.get('bath')),
                        row.get('face', ''), row.get('outside_view', ''),
                        row.get('furnished_status', ''), row.get('property_status', ''),
                        row.get('property_link', ''), row.get('property_img_link', ''),
                        row.get('property_description', ''),
                        availability,
                    ))

                cursor.executemany('''
                INSERT INTO properties (
                    property_type, property_name, location_area, location, total_price,
                    area_sqft, price_sqft, bhk, bath, face, outside_view,
                    furnished_status, property_status, property_link, property_img_link,
                    property_description, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', data)
                logger.info("Successfully imported %d item(s) from %s", len(data), file_name)
                if skipped_price:
                    logger.warning("%d row(s) in %s had an unparsable total_price and were stored as 0.", skipped_price, file_name)
                if sold_count:
                    logger.info(
                        "%d of %d propert(y/ies) in %s were marked sold in the source data and will "
                        "NOT be shown to buyers.", sold_count, len(data), file_name
                    )

        import_csv('plots_dataset.csv', 'Plot')
        import_csv('flats_dataset.csv', 'Flat')
        import_csv('house_and_villa_dataset.csv', 'House/Villa')

        conn.commit()
        logger.info("✅ Database is now full of real property data!")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        setup_database()
    except Exception as e:
        logger.error("❌ Database setup failed: %s", e, exc_info=True)
        raise SystemExit(1)