import pandas as pd
from BaseCode.Data_per_sector import CreateDataPerSector


def derivative():
    wskazniki = [
        "EN Liczba jednostek gospodarczych",
        "NP Wynik finansowy netto (zysk netto)",
        "GS (I) Przychody netto ze sprzedaży i zrównane z nimi",
        "TC Koszty ogółem",
        "INV Zapasy"
    ]

    CreateDataPerSector(wskazniki)

    input_file = "Data/dane_wedlug_wskaznikow.csv"
    output_file = "Data/pochodne.csv"

    df = pd.read_csv(input_file, encoding="utf-8")

    def clean_number(x):
        try:
            return float(
                str(x)
                .replace("\xa0", "")
                .replace(" ", "")
                .replace(",", ".")
            )
        except:
            return 0.0

    lata = [str(i) for i in range(2005, 2025)]
    for rok in lata:
        df[rok] = df[rok].apply(clean_number)

    grouped = df.groupby("numer PKD")[lata].sum()

    wynik_rows = []

    for sek, row in grouped.iterrows():
        values = row.values
        derivative = [0] + list(values[1:] - values[:-1])

        new_row = {
            "numer PKD": sek
        }

        for i, rok in enumerate(lata):
            new_row[rok] = derivative[i]

        wynik_rows.append(new_row)

    wynik_df = pd.DataFrame(wynik_rows)
    wynik_df.to_csv(output_file, index=False, encoding="utf-8")
