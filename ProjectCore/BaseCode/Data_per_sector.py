import re

import pandas as pd


def CreateDataPerSector(wskazniki):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    input_file = "Data/wsk_fin.xlsx - dane.csv"

    output_file = "Data/dane_wedlug_wskaznikow.csv"

    df = pd.read_csv(
        input_file,
        encoding="utf-8",
        dtype={"numer PKD": str}
    )

    df.columns = df.columns.str.strip()
    df["wskaźnik"] = df["wskaźnik"].str.strip()
    df["nazwa PKD"] = df["nazwa PKD"].str.strip()
    df["numer PKD"] = df["numer PKD"].str.strip()

    df = df[df["wskaźnik"].isin(wskazniki)]

    df = df[df["numer PKD"].str.startswith("SEK_")]

    lata = [str(rok) for rok in range(2005, 2025)]
    kolumny = ["numer PKD", "nazwa PKD", "wskaźnik"] + lata

    wynik = df[kolumny]

    wynik.to_csv(output_file, index=False, encoding="utf-8")