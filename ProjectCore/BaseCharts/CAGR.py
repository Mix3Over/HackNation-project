import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from BaseCode.Data_per_sector import CreateDataPerSector


def bigCagr():
    CreateDataPerSector(["NP Wynik finansowy netto (zysk netto)", "GS (I) Przychody netto ze sprzedaży i zrównane z nimi", "IO Wartość nakładów inwestycyjnych"])

    def calculate_cagr(start_val, end_val, periods):
        # Zabezpieczenie przed zerami i wartościami ujemnymi (wymagane dla CAGR)
        if start_val <= 0 or end_val <= 0 or periods <= 0:
            return 0
        return (end_val / start_val) ** (1 / periods) - 1

    # 2. Wczytanie danych
    try:
        df = pd.read_csv("Data/dane_wedlug_wskaznikow.csv", encoding="utf-8")
    except FileNotFoundError:
        print("Nie znaleziono pliku CSV.")
        return

    # Definiujemy lata, które nas interesują
    # range(2005, 2025) daje liczby od 2005 do 2024 włącznie.
    kolumny_lata = [str(rok) for rok in range(2005, 2025)]

    # --- POPRAWKA 1: CZYSZCZENIE DANYCH MUSI BYĆ NA POCZĄTKU ---
    # Musimy zamienić napisy na liczby PRZED obliczeniami
    for rok in kolumny_lata:
        if rok in df.columns:
            df[rok] = df[rok].astype(str).str.replace("\xa0", "", regex=True)
            df[rok] = df[rok].str.replace(" ", "", regex=True)
            df[rok] = df[rok].str.replace(",", ".", regex=False)
            df[rok] = pd.to_numeric(df[rok], errors="coerce")  # 'bd' zamieni się na NaN

    wyniki_macierzy = []

    # --- POPRAWKA 2: ITERACJA PO UNIKALNYCH NAZWACH SEKTORÓW ---
    sektory = df["nazwa PKD"].unique()  # Pobieramy listę nazw sektorów

    print("Rozpoczynam obliczenia...")

    for sektor in sektory:
        if sektor == "OGÓŁEM":
            continue  # Często warto pominąć sumę ogólną

        df_sektor = df[df["nazwa PKD"] == sektor]

        # Pobieramy wiersze dla 3 wskaźników
        row_sprzedaz = df_sektor[
            df_sektor["wskaźnik"].str.contains("GS \(I\)", regex=True, na=False)
        ]
        row_zysk = df_sektor[
            df_sektor["wskaźnik"].str.contains("NP Wynik finansowy netto", na=False)
        ]
        row_inwestycje = df_sektor[
            df_sektor["wskaźnik"].str.contains("IO Wartość nakładów", na=False)
        ]

        # Sprawdzamy czy mamy komplet danych (czy wiersze istnieją)
        if row_sprzedaz.empty or row_zysk.empty or row_inwestycje.empty:
            continue

        try:
            # Pobieramy dane liczbowe (pierwszy wiersz z danego sektora dla danego wskaźnika)
            sprzedaz = row_sprzedaz[kolumny_lata].iloc[0]
            zysk = row_zysk[kolumny_lata].iloc[0]
            inwestycje = row_inwestycje[kolumny_lata].iloc[0]

            # --- POPRAWKA 3: LOGIKA OBLICZEŃ ---

            # 1. OŚ X: CAGR (np. 5 letni: 2019-2024 lub 2018-2023)
            # Upewniamy się, że kolumny istnieją. Weźmy np. ostatnie 5 lat dostępnych danych.
            rok_koniec = "2023"  # Lub '2024' jeśli dane są pełne
            rok_start = "2018"
            okres = 5

            if rok_koniec in sprzedaz and rok_start in sprzedaz:
                cagr_sprzedazy = calculate_cagr(
                    sprzedaz[rok_start], sprzedaz[rok_koniec], okres
                )
            else:
                cagr_sprzedazy = 0

            # 2. OŚ Y: Rentowność (ROS) - średnia z ostatnich 3 lat
            lata_do_sredniej = kolumny_lata[-3:]  # Ostatnie 3 dostępne lata

            suma_zysk = zysk[lata_do_sredniej].sum()
            suma_przychody = sprzedaz[lata_do_sredniej].sum()

            if suma_przychody == 0:
                avg_ros = 0
            else:
                avg_ros = suma_zysk / suma_przychody

            # 3. WIELKOŚĆ: Intensywność Inwestycji
            suma_inwestycje = inwestycje[lata_do_sredniej].sum()

            if suma_przychody == 0:
                avg_capex = 0
            else:
                avg_capex = suma_inwestycje / suma_przychody

            # Filtrujemy tylko jeśli dane mają sens matematyczny
            if -0.5 < avg_ros < 0.5 and -0.5 < cagr_sprzedazy < 0.5:
                wyniki_macierzy.append(
                    {
                        "Sektor": sektor,
                        "Wzrost_Przychodow": cagr_sprzedazy,
                        "Rentownosc": avg_ros,
                        "Inwestycje": avg_capex,
                    }
                )

        except Exception:
            # Warto wydrukować błąd, żeby wiedzieć co się stało
            # print(f"Błąd dla sektora {sektor}: {e}")
            continue

    # Tworzymy DataFrame
    df_macierz = pd.DataFrame(wyniki_macierzy)

    # Sprawdzamy czy cokolwiek się policzyło
    if df_macierz.empty:
        print("Błąd: Brak danych po przetworzeniu. Sprawdź format pliku CSV.")
        return

    df_macierz = df_macierz.sort_values(by=["Rentownosc"], ascending=False).reset_index(
        drop=True
    )

    # --- RYSOWANIE WYKRESU ---
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(right=0.60, left=0.08, top=0.9, bottom=0.1)

    # Skalowanie bąbelków
    sizes = df_macierz["Inwestycje"] * 5000
    sizes = [max(80, s) for s in sizes]  # Minimalna wielkość kropki to 10

    # Rysujemy WSZYSTKO jedną komendą (scatter plot), a nie w pętli plot()
    scatter = plt.scatter(
        df_macierz["Wzrost_Przychodow"],
        df_macierz["Rentownosc"],
        s=sizes,
        alpha=0.8,
        c=df_macierz["Rentownosc"],
        cmap="RdYlGn",
        edgecolors='black',
        linewidth=0.5)

    # Linie pomocnicze
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)

    # Podpisy
    plt.title(
        "Macierz Branż: Rentowność vs Wzrost (Wielkość = Inwestycje)", fontsize=16
    )
    plt.xlabel("Wzrost Przychodów (CAGR)", fontsize=12)
    plt.ylabel("Rentowność Sprzedaży (ROS)", fontsize=12)

    # Etykiety dla najciekawszych punktów
    legenda_tekst = []

    for i, row in df_macierz.iterrows():
        numer = i + 1  # Numeracja od 1

        # Wstawiamy numer w środek kropki
        plt.text(
            row["Wzrost_Przychodow"],
            row["Rentownosc"],
            str(numer),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
        )

        # Przygotowujemy tekst do legendy
        # Skracamy nazwę sektora jeśli jest bardzo długa
        nazwa = str(row["Sektor"]).replace("DZIAŁALNOŚĆ ", "").capitalize()
        if len(nazwa) > 40:
            nazwa = nazwa[:37] + "..."

        linia_legendy = f"{numer}. {nazwa} (ROS: {row['Rentownosc']:.1%})"
        legenda_tekst.append(linia_legendy)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:.0%}".format(x)))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    plt.grid(True, linestyle=":", alpha=0.6)
    tekst_caly = "\n".join(legenda_tekst)
    plt.text(
        1.05,
        1.0,
        "LEGENDA (Ranking wg Rentowności):",
        transform=plt.gca().transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    plt.text(
        1.05,
        0.96,
        tekst_caly,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        linespacing=1.6,
    )
    plt.show()


# Uruchomienie
if __name__ == "__main__":
    bigCagr()