import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_per_sector import createDataPerSector


def bigCagr():
    createDataPerSector(["GS Przychody ogółem "])

    df = pd.read_csv("Dane/dane_wedlug_wskaznikow.csv", encoding="utf-8")

    kolumny_lata = [str(rok) for rok in range(2005, 2025)]

    for rok in kolumny_lata:
        df[rok] = df[rok].astype(str).str.replace("\xa0", "", regex=True)
        df[rok] = df[rok].str.replace(" ", "", regex=True)
        df[rok] = df[rok].str.replace(",", ".", regex=False)
        df[rok] = pd.to_numeric(df[rok], errors="coerce")

    plt.figure(figsize=(18, 8))

    for _, row in df.iterrows():
        plt.plot(kolumny_lata, row[kolumny_lata], marker="o", label=row["numer PKD"])

    plt.title("NP Wynik finansowy netto")
    plt.xlabel("Rok")
    plt.ylabel("Wynik finansowy netto")
    plt.xticks(rotation=45)
    plt.legend(title="Numer PKD", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

bigCagr()