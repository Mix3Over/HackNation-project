import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def forecast_economy():
    print("--- ROZPOCZYNAM DZIAŁANIE MODELU ---")

    # --- 1. INTELIGENTNE WCZYTYWANIE PLIKU ---
    filename = "wsk_fin.xlsx - dane.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Lista miejsc do sprawdzenia
    search_paths = [
        os.path.join(current_dir, "Data", filename),
        os.path.join(current_dir, "..", "Data", filename),
        os.path.join(current_dir, "Dane", filename),
        os.path.join(current_dir, "..", "Dane", filename),
        os.path.join(current_dir, filename),  # Sprawdź też w tym samym folderze
    ]

    file_path = None
    for path in search_paths:
        if os.path.exists(path):
            file_path = path
            print(f"✅ Znaleziono plik danych: {path}")
            break

    if file_path is None:
        print("\n❌ BŁĄD KRYTYCZNY: Nie znaleziono pliku csv.")
        print(f"Szukano pliku '{filename}' w folderach 'Data', 'Dane' i bieżącym.")
        return  # Przerywamy działanie, ale po wypisaniu błędu

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as e:
        print(f"❌ Błąd podczas otwierania pliku: {e}")
        return

    # --- 2. CZYSZCZENIE DANYCH ---
    print("Czyszczenie danych...")
    df.columns = df.columns.str.strip()
    # Czyścimy spacje wewnątrz tekstów
    if "wskaźnik" in df.columns:
        df["wskaźnik"] = (
            df["wskaźnik"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )
    if "nazwa PKD" in df.columns:
        df["nazwa PKD"] = (
            df["nazwa PKD"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )

    # Parametry predykcji
    target_sector = "OGÓŁEM"
    target_indicator_fragment = "GS (I)"  # Przychody ze sprzedaży (szukamy fragmentu)

    # Filtrowanie
    mask_sector = df["nazwa PKD"] == target_sector
    mask_indicator = df["wskaźnik"].str.contains(
        target_indicator_fragment, regex=False, case=False
    )

    row = df[mask_sector & mask_indicator]

    if row.empty:
        print(
            f"\n❌ BŁĄD: Nie znaleziono danych dla sektora '{target_sector}' i wskaźnika '{target_indicator_fragment}'."
        )
        print("Dostępne sektory (pierwsze 5):", df["nazwa PKD"].unique()[:5])
        return

    # --- 3. PRZYGOTOWANIE SZEREGU CZASOWEGO ---
    years = [str(y) for y in range(2005, 2025)]
    raw_values = row[years].iloc[0]

    clean_values = []
    clean_years = []

    for y in years:
        val = (
            str(raw_values[y])
            .strip()
            .replace("\xa0", "")
            .replace(" ", "")
            .replace(",", ".")
        )
        try:
            val_float = float(val)
            clean_values.append(val_float)
            clean_years.append(int(y))
        except ValueError:
            continue  # Pomijamy 'bd'

    if len(clean_values) < 5:
        print("\n❌ BŁĄD: Za mało danych liczbowych do nauki modelu (mniej niż 5 lat).")
        return

    ts_df = pd.DataFrame({"Year": clean_years, "Value": clean_values})

    # --- 4. INŻYNIERIA CECH (LAGS) ---
    lags = 3
    for i in range(1, lags + 1):
        ts_df[f"Lag_{i}"] = ts_df["Value"].shift(i)

    model_df = ts_df.dropna().reset_index(drop=True)

    X = model_df[[f"Lag_{i}" for i in range(1, lags + 1)]]
    y = model_df["Value"]
    years_model = model_df["Year"]

    # --- 5. TRENING I TEST (Ostatnie 4 lata jako test) ---
    split_index = len(model_df) - 4
    if split_index < 2:
        print("Za mało danych na podział trening/test. Uczę na całości.")
        split_index = len(model_df)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    years_test = years_model.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n--- WYNIKI MODELU ---")
    if not X_test.empty:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Test na latach {years_test.min()}-{years_test.max()}")
        print(f"Średni błąd (MAE): {mae:,.2f} PLN")
        print(f"Dopasowanie (R2): {r2:.2f}")
    else:
        print("Brak zbioru testowego (wszystkie dane użyte do nauki).")

    # --- 6. PROGNOZA NA 2025 ---
    # Bierzemy ostatnie znane 3 wartości (2024, 2023, 2022) żeby przewidzieć 2025
    last_known_values = ts_df["Value"].iloc[-lags:].values[::-1]
    prediction_2025 = model.predict([last_known_values])[0]

    print(f"\n>>> PROGNOZA {target_sector} NA ROK 2025: {prediction_2025:,.2f} PLN <<<")

    # --- 7. WIZUALIZACJA ---
    plt.figure(figsize=(12, 6))

    # Dane historyczne
    plt.plot(
        ts_df["Year"],
        ts_df["Value"],
        label="Historia",
        marker="o",
        color="#1f77b4",
        linewidth=2,
    )

    # Dane testowe (predykcja wsteczna) - tylko jeśli mamy test
    if not X_test.empty:
        plt.plot(
            years_test,
            y_pred,
            label="Weryfikacja modelu",
            marker="x",
            linestyle="--",
            color="orange",
        )

    # Przyszłość
    plt.scatter(
        [2025], [prediction_2025], color="red", s=150, zorder=5, label="Prognoza 2025"
    )

    plt.title(f"Prognoza: {target_sector} - Przychody (GS I)", fontsize=16)
    plt.xlabel("Rok")
    plt.ylabel("Wartość (PLN)")
    plt.xticks(list(range(2005, 2026)), rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Wyłączenie notacji naukowej na osi Y
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)

    plt.tight_layout()

    print("Generowanie wykresu...")
    plt.savefig("prognoza_2025.png") # Zapis do pliku
    plt.show() # Wyświetlenie okna

if __name__ == "__main__":
    forecast_economy()