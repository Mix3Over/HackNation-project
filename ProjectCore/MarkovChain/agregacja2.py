import pandas as pd

# Nowe mapowanie SEK_x na branże
SEK_TO_SECTOR = {
    "SEK_A": "Rolnictwo",
    "SEK_B": "Górnictwo i kopalnictwo",
    "SEK_C": "Przetwórstwo przemysłowe",
    "SEK_D": "Wytwarzanie i zaopatrywanie w energię el., gaz, wodę",
    "SEK_E": "Budownictwo",
    "SEK_F": "Handel",
    "SEK_G": "Transport",
    "SEK_H": "Pośrednictwo finansowe",
    "SEK_I": "Obsługa nieruchomości i firm",
    "SEK_J": "Edukacja",
    "SEK_K": "Ochrona zdrowia",
    "SEK_L": "Pozostała działalność usługowa",
    "SEK_M": "Hotele i restauracje",
    "SEK_N": "Zakończyły działalność",
}

def read_and_aggregate(file_path: str) -> pd.DataFrame:
    # Wczytanie pliku CSV
    df = pd.read_csv(file_path, sep=";", decimal=",")

    # Pierwsza kolumna → "SEK"
    df.rename(columns={df.columns[0]: "SEK"}, inplace=True)

    # Filtrujemy tylko znane SEK
    df = df[df["SEK"].isin(SEK_TO_SECTOR.keys())]

    # Mapowanie na nazwy branż
    df["Sector"] = df["SEK"].map(SEK_TO_SECTOR)

    # Konwersja kolumn lat na float
    years = [c for c in df.columns if c != "SEK" and c != "Sector"]
    for y in years:
        df[y] = pd.to_numeric(df[y], errors="coerce")

    # Grupowanie po branżach i sumowanie
    sector_df = df.groupby("Sector", as_index=False)[years].sum()

    # Zaokrąglenie do 2 miejsc po przecinku
    sector_df[years] = sector_df[years].round(2)

    # Sortowanie według listy states
    states = [
        "Rolnictwo",
        "Górnictwo i kopalnictwo",
        "Przetwórstwo przemysłowe",
        "Wytwarzanie i zaopatrywanie w energię el., gaz, wodę",
        "Budownictwo",
        "Handel",
        "Transport",
        "Hotele i restauracje",
        "Pośrednictwo finansowe",
        "Obsługa nieruchomości i firm",
        "Edukacja",
        "Ochrona zdrowia",
        "Pozostała działalność usługowa",
        "Zakończyły działalność",
    ]
    sector_df["Sector"] = pd.Categorical(sector_df["Sector"], categories=states, ordered=True)
    sector_df = sector_df.sort_values("Sector").reset_index(drop=True)

    return sector_df

# Wczytanie i agregacja
sector_df = read_and_aggregate("Data/main_index_predictions.csv")

# Zapis do nowego pliku CSV
sector_df.to_csv("Data/aggregated_sectors.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

print("Agregacja zakończona. Plik zapisany jako Data/aggregated_sectors.csv")
