import re
import pandas as pd


def createDataPerSector(wskazniki):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df = pd.read_csv(
        "Dane/wsk_fin.xlsx - dane.csv",
        encoding="utf-8",
        dtype={"numer PKD": str}
    )

    df.columns = df.columns.str.strip()
    df["wskaźnik"] = df["wskaźnik"].str.strip()
    df["nazwa PKD"] = df["nazwa PKD"].str.strip()
    df["numer PKD"] = df["numer PKD"].str.strip()

    wybrane_wskazniki = wskazniki

    filtrowane_dane = df[df["wskaźnik"].isin(wybrane_wskazniki)]

    #filtrowane_dane = filtrowane_dane[filtrowane_dane["numer PKD"].str.upper() != "OG"]

    kolumny_lata = [str(rok) for rok in range(2005, 2025)]
    kolumny_do_wyswietlenia = ["numer PKD", "nazwa PKD", "wskaźnik"] + kolumny_lata
    wynik = filtrowane_dane[kolumny_do_wyswietlenia]

    wynik.to_csv("Dane/dane_wedlug_CAGR.csv", index=False, encoding="utf-8")