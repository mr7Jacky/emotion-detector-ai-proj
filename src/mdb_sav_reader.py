import pandas_access as mdb
import pandas as pd

db_filename = 'isear_databank.mdb'

# Listing the tables.

for tbl in mdb.list_tables(db_filename):
    df = mdb.read_table(db_filename, tbl)
    df.to_csv(tbl+'.csv')    
# Read a small table.


db = pd.read_spss('ISEAR SPSS Databank.sav')
db.to_csv('isear.csv')

