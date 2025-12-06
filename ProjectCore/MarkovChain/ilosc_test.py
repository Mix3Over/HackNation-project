import pandas as pd

df_raw = pd.read_csv("data/wsk_fin.csv", sep=";")
df_raw["pkd_section"] = df_raw["PKD"].astype(str).str.extract(r"^(\d{2})")
print(sorted(df_raw["pkd_section"].dropna().unique()))
print("Liczba działów w surowych danych:", df_raw["pkd_section"].nunique())


df_idx = pd.read_csv("output/index_branż.csv", sep=";")
print(sorted(df_idx["pkd_section"].unique()))
print("Liczba działów w indeksie:", df_idx["pkd_section"].nunique())
