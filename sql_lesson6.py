# %% 

import sqlite3
import pandas as pd

# Create SQLite database
conn = sqlite3.connect("microbiome_db.sqlite")

# Recreate patients table
conn.execute('''
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    patient_name TEXT, 
    age INTEGER, 
    city TEXT
);
''')

conn.executemany('''
INSERT INTO patients (patient_name, age, city) VALUES (?, ?, ?);
''', [
    ('Alice Smith', 32, 'Seattle'),
    ('Bob Johnson', 28, 'Boston'),
    ('Carol Williams', 35, 'San Francisco'),
    ('David Miller', 40, 'Chicago')
])

# Recreate samples table
conn.execute('''
CREATE TABLE samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    collection_date TEXT, 
    status TEXT, 
    patient_id INTEGER, 
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);
''')

conn.executemany('''
INSERT INTO samples (collection_date, status, patient_id) VALUES (?, ?, ?);
''', [
    ('2025-03-18', 'processed', 1),
    ('2025-03-19', 'pending', 2),
    ('2025-03-20', 'processed', 3),
    ('2025-03-21', 'processed', 4),
    ('2025-03-22', 'pending', 1)
])

conn.commit()
print("Database recreated successfully!")
# Expand samples table to simulate a larger dataset
for _ in range(100):  # Repeat 100 times
    conn.execute('''
        INSERT INTO samples (collection_date, status, patient_id)
        SELECT collection_date, status, patient_id FROM samples;
    ''')

conn.commit()
print("Samples table expanded!")
# -------------------------------------
# STEP 2: Run unoptimized query
# -------------------------------------

import time
import pandas as pd

# STEP 2: Run unoptimized query
start = time.time()

query = '''
SELECT *
FROM samples
WHERE status = 'processed';
'''

df = pd.read_sql_query(query, conn)

end = time.time()
print(f"Query took {end - start:.4f} seconds")


# -------------------------------------
# STEP 3: Add index and rerun query
# -------------------------------------

# Create an index on the 'status' column
conn.execute('CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);')
conn.commit()
print("Index on 'status' column created.")

# Time the same query again
start = time.time()

query = '''
SELECT *
FROM samples
WHERE status = 'processed';
'''

df_indexed = pd.read_sql_query(query, conn)

end = time.time()
print(f"Query with index took {end - start:.4f} seconds")

