import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def CapEx(PPE_current, PPE_previous, amortisation):
    capex = (PPE_current - PPE_previous) + amortisation
    return capex

df = pd.read_csv("Dane/dane_wedlug_wskaznikow.csv", encoding="utf-8")

row_netto = df[df['wskaźnik'].str.contains('NP Wynik finansowy netto', na=False)]
row_brutto = df[df['wskaźnik'].str.contains('POS Wynik na sprzedaży', na=False)]

lata_cols = [str(year) for year in range(2005, 2025)]

def clean_currency(x):
    if isinstance(x, str):
        x = x.strip()
        if x == 'bd': return np.nan
        x = x.replace(' ', '').replace('\xa0', '').replace(',', '.')
    return float(x)

dane_netto = row_netto[lata_cols].iloc[0].apply(clean_currency)
dane_brutto = row_brutto[lata_cols].iloc[0].apply(clean_currency)

lata = [int(col) for col in lata_cols]

def get_valid_cagr(data_series):
    valid_data = data_series.dropna()
    if valid_data.empty: return 0
    start_val = valid_data.iloc[0]
    end_val = valid_data.iloc[-1]
    periods = len(valid_data) - 1
    if periods <= 0: return 0
    return CapEx(start_val, end_val, periods)

cagr_netto = get_valid_cagr(dane_netto)
cagr_brutto = get_valid_cagr(dane_brutto)

for rok in lata_cols:
    df[rok] = df[rok].astype(str).str.replace("\xa0", "", regex=True)
    df[rok] = df[rok].str.replace(" ", "", regex=True)
    df[rok] = df[rok].str.replace(",", ".", regex=False)
    df[rok] = pd.to_numeric(df[rok], errors="coerce")

plt.figure(figsize=(18, 8))

for _, row in df.iterrows():
    plt.plot(lata_cols, row[lata_cols], marker="o", label=row["numer PKD"])

plt.title("NP Wynik finansowy netto")
plt.xlabel("Rok")
plt.ylabel("Wynik finansowy netto")
plt.xticks(rotation=45)
plt.legend(title="Numer PKD", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()