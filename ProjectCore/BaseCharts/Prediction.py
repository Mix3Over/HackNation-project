import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_predictions(file_path, indexName):
    # Wczytaj CSV z separatorem ";"
    try:
        df = pd.read_csv(file_path, sep=";", engine="python")
    except Exception as e:
        print("Błąd wczytywania pliku:", e)
        return

    # Ujednolicenie nazw kolumn
    df.columns = df.columns.str.lower().str.strip()

    # Sprawdź wymagane kolumny
    required = {"pkd_section", "year", indexName}
    if not required.issubset(df.columns):
        print("Brakuje wymaganych kolumn:", required - set(df.columns))
        return

    # Uporządkowana tabela
    df_table = df[["pkd_section", "year", indexName]].copy()
    df_table = df_table.sort_values(["pkd_section", "year"]).reset_index(drop=True)

    print("\n=== Dane oryginalne ===")
    print(df_table)

    # --- TWORZENIE PREDYKCJI ---
    all_sections = df_table["pkd_section"].unique()
    new_rows = []

    for sec in all_sections:
        sec_data = df_table[df_table["pkd_section"] == sec]

        years = sec_data["year"].values
        values = sec_data[indexName].values

        # Regresja liniowa: y = a*x + b
        a, b = np.polyfit(years, values, 1)

        last_year = years.max()
        future_years = [last_year + 1, last_year + 2, last_year + 3]

        for fy in future_years:
            pred_value = a * fy + b
            new_rows.append([sec, fy, pred_value])

    df_pred = pd.DataFrame(new_rows, columns=["pkd_section", "year", indexName])

    # Scal dane oryginalne + predykcje
    df_full = pd.concat([df_table, df_pred], ignore_index=True)
    df_full = df_full.sort_values(["pkd_section", "year"]).reset_index(drop=True)

    print("\n=== Dane z predykcjami ===")
    print(df_full)

    # --- RYSOWANIE 20 SEKCJI NA JEDNYM WYKRESIE ---
    sections = sorted(df_full["pkd_section"].unique())[:20]

    plt.figure(figsize=(12, 7))

    for sec in sections:
        sec_data = df_full[df_full["pkd_section"] == sec]
        plt.plot(sec_data["year"], sec_data[indexName], marker="o", label=f"{sec}")

    plt.title("Pierwsze 20 sekcji PKD – Main Index z predykcjami (+3 lata)")
    plt.xlabel("Year")
    plt.ylabel("Main Index")
    plt.grid(True)
    plt.legend(title="PKD Section")
    plt.tight_layout()
    plt.show()

    return df_full
