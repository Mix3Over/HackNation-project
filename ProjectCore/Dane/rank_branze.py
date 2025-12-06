import pandas as pd

df = pd.read_csv("output/index_branż.csv", sep=";")

# sortowanie po głównym indeksie - od największego
df_sorted = df.sort_values("main_index", ascending=False)

# zapis posortowanej wersji
df_sorted.to_csv("output/index_branż_posortowane.csv", sep=";", index=False, encoding="utf-8-sig")

# wypisz Top 10 do konsoli
cols = ["pkd_section", "pkd_name", "main_index", "size_index", "growth_index", "class"]
print(df_sorted[cols].head(10))
