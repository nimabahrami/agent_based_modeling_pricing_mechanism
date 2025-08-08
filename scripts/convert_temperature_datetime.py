import pandas as pd

# Load the data
infile = 'data/processed/temperature_data.csv'
df = pd.read_csv(infile)

# Convert 'dateandtime' to standard datetime format
# Remove timezone and 'T', keep as 'YYYY-MM-DD HH:MM:SS'
df['period_end'] = pd.to_datetime(df['period_end']).dt.strftime('%Y-%m-%d %H:%M:%S')

# Save back to CSV (overwrite original)
df.to_csv(infile, index=False)

print('Converted dateandtime column to standard datetime format.') 