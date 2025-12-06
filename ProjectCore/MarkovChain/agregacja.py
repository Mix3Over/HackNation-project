import unicodedata

import pandas as pd


def _norm(text: str) -> str:
    """Uproszczony tekst do dopasowań (bez polskich znaków, wielkie litery)."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", str(text)) \
                     .encode("ascii", "ignore") \
                     .decode("ascii")
    return text.upper()


def pkd_name_to_sector(pkd_name: str) -> str:
    """
    Mapuje pełną nazwę działalności PKD (kolumna `pkd_name`)
    na jeden z 13 sektorów makro.
    """
    n = _norm(pkd_name)

    # Rolnictwo, leśnictwo, rybactwo
    if any(kw in n for kw in ["UPRAWY", "ROLNE", "HODOWLA", "LESNICTWO",
                              "LOWIECTWO", "RYBACTWO"]):
        return "Rolnictwo"

    # Górnictwo i kopalnictwo
    if any(kw in n for kw in ["GORNIC", "KOPALNICTWO", "WYDOBYWANIE", "KOKSU"]):
        return "Gornictwo_i_kopalnictwo"

    # Energia, gaz, woda, ścieki, odpady
    if any(kw in n for kw in [
        "ENERGIE ELEKTRYCZNA", "ENERGII ELEKTRYCZNEJ", "GAZ",
        "PARA WODNA", "GORACA WODA", "DOSTARCZANIE WODY",
        "SCIEKOW", "GOSPODARKA ODPADAMI", "REKULTYWACJA"
    ]):
        return "Energia_gaz_woda"

    # Budownictwo
    if any(kw in n for kw in [
        "ROBOTY BUDOWLANE", "BUDOWA OBIEKTOW",
        "WZNOSZENIEM BUDYNKOW", "INZYNIERII LADOWEJ I WODNEJ"
    ]):
        return "Budownictwo"

    # Handel (hurt/detal + pojazdy)
    if "HANDEL" in n:
        return "Handel"

    # Transport, magazynowanie, poczta, kurierzy
    if any(kw in n for kw in [
        "TRANSPORT", "MAGAZYNOWANIE", "KURIERSKA", "POCZTOWA"
    ]):
        return "Transport"

    # Hotele i restauracje (zakwaterowanie + wyżywienie + turystyka blisko hotelarstwa)
    if any(kw in n for kw in [
        "ZAKWATEROWANIE",
        "USLUGOWA ZWIAZANA Z WYZYWIENIEM",
        "HOTELE", "RESTAURACJE", "GASTRONICZNA"
    ]):
        return "Hotele_i_restauracje"

    # Pośrednictwo finansowe: finanse, ubezpieczenia, fundusze, usługi pomocnicze
    if any(kw in n for kw in [
        "FINANSOWA DZIALALNOSC USLUGOWA",
        "USLUGI FINANSOWE",
        "FUNDUSZE EMERYTALNE",
        "UBEZPIECZENIA", "REASEKURACJA",
        "WSPOMAGAJACA USLUGI FINANSOWE",
        "RYNKU FINANSOWEGO"
    ]):
        return "Posrednictwo_finansowe"

    # Obsługa nieruchomości i firm (L + M + N: nieruchomości, usługi dla biznesu)
    if any(kw in n for kw in [
        "NIERUCHOMOSCI",
        "HEAD OFFICES", "FIRM CENTRALNYCH",
        "ARCHITEKTURY I INZYNIERII",
        "BADANIA I ANALIZY TECHNICZNE",
        "REKLAMA", "BADANIE RYNKU",
        "OBSLOGA BIURA",
        "WSPOMAGAJACA PROWADZENIE DZIALALNOSCI GOSPODARCZEJ",
        "DZIALALNOSC PROFESJONALNA, NAUKOWA I TECHNICZNA",
        "ZATRUDNIENIEM",
        "WYNAJEM I DZIERZAWA",
        "PRAWNICZA", "RACHUNKOWO-KSIEGOWA", "DORADZTWO PODATKOWE"
    ]):
        return "Obsluga_nieruchomosci_i_firm"

    # Edukacja
    if "EDUKACJA" in n:
        return "Edukacja"

    # Ochrona zdrowia + pomoc społeczna + weterynaria
    if any(kw in n for kw in [
        "OPIEKA ZDROWOTNA",
        "POMOC SPOLECZNA",
        "WETERYNARYJNA"
    ]):
        return "Ochrona_zdrowia"

    # Przetwórstwo przemysłowe - wszystkie pozostałe "PRODUKCJA ..."
    if n.startswith("PRODUKCJA ") or " PRODUKCJA " in n:
        return "Przetworstwo_przemyslowe"

    # Domyślnie: pozostałe usługi (IT, media, kultura, sport, organizacje, gospodarstwa domowe itd.)
    return "Pozostala_dzialalnosc_uslugowa"


def aggregate_by_sector_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przyjmuje DataFrame z kolumnami:
    - 'year'
    - 'pkd_name'
    - 'revenue', 'net_profit', 'investments',
      'firms', 'profitable_firms', 'bank_debt', 'total_liab'
    i zwraca agregację do 13 sektorów.
    """
    df = df.copy()
    df["sector"] = df["pkd_name"].apply(pkd_name_to_sector)

    sum_cols = [
        "revenue",
        "net_profit",
        "investments",
        "firms",
        "profitable_firms",
        "bank_debt",
        "total_liab",
    ]

    # upewniamy się, że dane są liczbowe
    for c in sum_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grouped = (
        df.groupby(["year", "sector"], as_index=False)[sum_cols]
          .sum(min_count=1)
    )

    # wskaźniki sektorowe
    grouped["profit_margin"] = grouped["net_profit"] / grouped["revenue"]
    grouped["profitable_share"] = grouped["profitable_firms"] / grouped["firms"]
    grouped["leverage_bank_to_rev"] = grouped["bank_debt"] / grouped["revenue"]
    grouped["leverage_tot_to_rev"] = grouped["total_liab"] / grouped["revenue"]
    grouped["invest_intensity"] = grouped["investments"] / grouped["revenue"]

    return grouped


df = pd.read_csv(
    "./output/index_branż_posortowane.csv",
    sep=";", decimal=",", encoding="utf-8-sig", engine="python"
)

sector_df = aggregate_by_sector_name(df)
print(sector_df)
print(sector_df["sector"].unique())  # powinno dać 13 sektorów
