# Description: convert CSV to sql database

import pandas as pd
import sqlite3 as sl

# Load your dataset
data = pd.read_csv('spotify.csv', index_col=0)

# Drop rows with missing values
data = data.dropna()
# Drop duplicates
data = data.drop_duplicates(subset=['track_name', 'artists'], keep='first')


# Connect to Database
conn = sl.connect('spotifyMusic.db')
data.to_sql('songs', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

# print("Data has been stored in the database.")