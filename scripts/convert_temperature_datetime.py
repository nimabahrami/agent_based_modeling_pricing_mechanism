import pandas as pd

path = 'data/processed/temperature_data.csv'
df = pd.read_csv(path)
df['period_end'] = pd.to_datetime(df['period_end']).dt.strftime('%Y-%m-%d %H:%M:%S')
df.to_csv(infile, index=False)
