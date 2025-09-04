import numpy as np
import pandas as pd

from src.config import BASE
from src.utils import rsi


df = pd.read_csv(BASE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Date': 'date'})

# Relação óleo/farelo
oleo_farelo = df[['date', 'boc1', 'smc1']]
oleo_farelo = oleo_farelo.dropna()
oleo_farelo['oleo_tons'] = oleo_farelo['boc1'] * 22.0462
oleo_farelo['farelo_tons'] = oleo_farelo['smc1'] * 1.1023 
oleo_farelo['oleo/farelo'] = oleo_farelo['oleo_tons'] / oleo_farelo['farelo_tons'] 
oleo_farelo = oleo_farelo.sort_values(by='date', ascending=True)

# Relação óleo/palma
oleo_palma = df[['date', 'boc1', 'fcpoc1', 'myr=']]
oleo_palma = oleo_palma.dropna()
oleo_palma['oleo_tons'] = oleo_palma['boc1'] * 22.0462
oleo_palma['palma_tons'] = oleo_palma['fcpoc1'] / oleo_palma['myr=']
oleo_palma['oleo/palma'] = oleo_palma['oleo_tons'] / oleo_palma['palma_tons'] 
oleo_palma = oleo_palma.sort_values(by='date', ascending=True)

# Relação óleo/diesel
oleo_diesel = df[['date', 'boc1', 'hoc1']]
oleo_diesel = oleo_diesel.dropna()
oleo_diesel['oleo_tons'] = oleo_diesel['boc1'] * 22.0462
oleo_diesel['diesel_tons'] = oleo_diesel['hoc1'] / 0.003785 / 0.782
oleo_diesel['oleo/diesel'] = oleo_diesel['oleo_tons'] / oleo_diesel['diesel_tons'] 
oleo_diesel = oleo_diesel.sort_values(by='date', ascending=True)

# Óleo e indicadores técnicos
oleo_quote = df[['date', 'boc1']]
oleo_quote = oleo_quote.dropna()
oleo_quote = rsi(oleo_quote, ticker_col='boc1')

# Flat USD óleo
oleo_flat_usd = df[['date', 'boc1', 'so-premp-c1']]
oleo_flat_usd = oleo_flat_usd.dropna()
oleo_flat_usd['oleo_flat_usd'] = (oleo_flat_usd['boc1'] + (oleo_flat_usd['so-premp-c1']/100)) * 22.0462
oleo_flat_usd = rsi(oleo_flat_usd, ticker_col='oleo_flat_usd')
