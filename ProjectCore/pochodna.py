import pandas as pd
import matplotlib.pyplot as plt
from data_per_sector import createDataPerSector

def pochodna():

    createDataPerSector([
        "EN Liczba jednostek gospodarczych",
        "NP Wynik finansowy netto (zysk netto)",
        "GS (I) Przychody netto ze sprzedaży i zrównane z nimi",
        "TC Koszty ogółem",
        "INV Zapasy"
    ])

    plik_wejsciowy = "Dane/dane_wedlug_wskaznikow.csv"
    df = pd.read_csv(plik_wejsciowy, sep=",", encoding="utf-8")

    wskaźniki = [
        "EN Liczba jednostek gospodarczych",
        "NP Wynik finansowy netto (zysk netto)",
        "GS (I) Przychody netto ze sprzedaży i zrównane z nimi",
        "TC Koszty ogółem",
        "INV Zapasy"
    ]

    df = df[df['wskaźnik'].isin(wskaźniki)]

    def czysc_liczby(x):
        try:
            return float(str(x).replace("\xa0", "").replace(" ", "").replace(",", "."))
        except:
            return None

    lata = [str(i) for i in range(2005, 2025)]
    for rok in lata:
        df[rok] = df[rok].apply(czysc_liczby)

    df_index = df.set_index(['numer PKD', 'wskaźnik'])

    np_og = df_index.loc[("OG", "NP Wynik finansowy netto (zysk netto)"), lata].copy()
    en_og = df_index.loc[("OG", "EN Liczba jednostek gospodarczych"), lata].copy()

    for rok in lata:
        df_index.loc[("OG", "NP Wynik finansowy netto (zysk netto)"), rok] = np_og[rok]
        #df_index.loc[("OG", "EN Liczba jednostek gospodarczych"), rok] = en_og[rok]

    df = df_index.reset_index()

    plik_wyjsciowy = "Dane/pochodne.csv"
    df.to_csv(plik_wyjsciowy, index=False, encoding="utf-8")

    plt.figure(figsize=(18, 8))

    for _, row in df.iterrows():
        wartosci = row[lata].astype(float).values
        pochodna = [0] + list(wartosci[1:] - wartosci[:-1])
        plt.plot(lata, pochodna, marker="o", label=row["numer PKD"] + " - " + row["wskaźnik"])

    plt.title("Tempo zmian wskaźników w czasie (pochodna)")
    plt.xlabel("Rok")
    plt.ylabel("Zmiana wskaźnika w stosunku do poprzedniego roku")
    plt.xticks(rotation=45)
    plt.legend(title="Sektor i wskaźnik", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
