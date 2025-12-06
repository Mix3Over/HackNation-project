"""
Indeks Branż – prototyp z wyborem normalizacji i wag indeksów.

Funkcjonalność:
1. Wczytuje dane finansowe wg PKD z:
   - data/wsk_fin.csv   (separator ';', UTF-8)
   - data/krz_pkd.csv   (separator ',', ISO-8859-1, używany do kontroli spójności)
   - data/mapowanie_pkd.xlsx (opcjonalne mapowanie PKD 2007 -> PKD 2025)

2. Agreguje dane do poziomu działu PKD (2 cyfry) i lat.

3. Dla każdego działu PKD (pkd_section) i roku CURRENT_YEAR:
   - liczy wskaźniki finansowe,
   - liczy dynamiki (CAGR),
   - liczy zmienność marży (std z 3 lat).

4. Buduje indeksy cząstkowe:
   - size_index, growth_index, profit_index, risk_index, outlook_index.

5. Buduje indeksy syntetyczne:
   - current_index, future_index, main_index.

6. Klasyfikuje branże:
   - „Gwiazdy wzrostu”,
   - „Trudna teraźniejszość, dobra przyszłość”,
   - „Stabilne, ale bez dynamiki”,
   - „Branże ryzykowne / schyłkowe”.

7. Zapisuje wynik do:
   output/index_branż.csv
"""

import os
import re
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------
# USTAWIENIA GLOBALNE – TUTAJ MOŻESZ ZMIENIAĆ
# -------------------------------------------------

DATA_DIR = "data"
OUTPUT_DIR = "output"

FILE_WSK_FIN = os.path.join(DATA_DIR, "wsk_fin.csv")
FILE_KRZ_PKD = os.path.join(DATA_DIR, "krz_pkd.csv")
FILE_MAP_PKD = os.path.join(DATA_DIR, "mapowanie_pkd.xlsx")

# Rok analizowany:
CURRENT_YEAR = 2024
BASE_YEAR_FOR_CAGR = 2021  # do CAGR 3-letniego

# METODA NORMALIZACJI:
#   "minmax" -> 0–100 (jak wcześniej)
#   "zscore" -> z = (x - mean) / std (z-score)
NORMALIZATION_METHOD = "minmax"  # albo "zscore"
# NORMALIZATION_METHOD = "zscore"

# DOMYŚLNE WAGI INDEXÓW (możesz zmienić dowolne)
DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Filar: skala
    "size": {
        "revenue": 0.5,
        "firms": 0.5,
    },
    # Filar: wzrost
    "growth": {
        "cagr_revenue": 0.5,
        "cagr_net_profit": 0.5,
    },
    # Filar: rentowność
    "profit": {
        "profit_margin": 0.5,
        "profitable_share": 0.5,
    },
    # Filar: ryzyko
    "risk": {
        "leverage_bank_to_rev": 1 / 3,
        "leverage_tot_to_rev": 1 / 3,
        "profit_margin_std_3y": 1 / 3,
    },
    # Filar: perspektywy
    "outlook": {
        "invest_intensity": 0.5,
        "cagr_investments": 0.5,
    },
    # Indeks bieżącej kondycji
    "current": {
        "size_index": 0.20,
        "growth_index": 0.25,
        "profit_index": 0.25,
        "risk_index": 0.30,
    },
    # Indeks perspektyw
    "future": {
        "growth_index": 0.30,
        "outlook_index": 0.40,
        "risk_index": 0.30,
    },
    # Główny indeks
    "main": {
        "current_index": 0.60,
        "future_index": 0.40,
    },
}

# Jeżeli chcesz własne wagi, możesz zdefiniować tu słownik o tej samej strukturze,
# np. CUSTOM_WEIGHTS = {"size": {"revenue": 0.7, "firms": 0.3}}
# i zostawić None, jeśli mają zostać wartości domyślne.
CUSTOM_WEIGHTS: Optional[Dict[str, Dict[str, float]]] = None


# -------------------------------------------------
# FUNKCJE POMOCNICZE
# -------------------------------------------------

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_polish_number(x):
    """Polski zapis liczby (spacje, przecinki) -> float; 'bd' itp. -> NaN."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if not isinstance(x, str):
        return np.nan

    s = x.strip()
    if s == "" or s.lower() in ["bd", "bd.", "brak", "nan"]:
        return np.nan

    s = s.replace("\xa0", "").replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def extract_pkd_section(code: str):
    """Z kodu PKD (np. '01.11.Z') wyciąga 2-cyfrowy dział PKD."""
    if not isinstance(code, str):
        return None
    m = re.match(r"(\d{2})", code)
    return m.group(1) if m else None


def extract_indicator_code(full_name: str):
    """Wyciąga kod wskaźnika z tekstu w kolumnie WSKAZNIK."""
    if not isinstance(full_name, str):
        return None
    full_name = full_name.strip()

    if full_name.startswith("GS (I)"):
        return "GS_I"
    if full_name.startswith("GS "):
        return "GS"
    if full_name.startswith("Przych. fin."):
        return "PRZYCH_FIN"
    return full_name.split()[0]


def safe_cagr(end_value, start_value, years: int):
    """CAGR z zabezpieczeniem dla <=0 i braków."""
    if start_value is None or end_value is None:
        return np.nan
    if np.isnan(start_value) or np.isnan(end_value):
        return np.nan

    if start_value <= 0 or end_value <= 0:
        if start_value == 0:
            return np.nan
        return (end_value - start_value) / abs(start_value) / years

    return (end_value / start_value) ** (1.0 / years) - 1.0


def minmax_scale(series: pd.Series) -> pd.Series:
    """Skalowanie do 0–100; jeśli wszystkie wartości takie same -> 50."""
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx):
        return pd.Series(np.nan, index=s.index)
    if np.isclose(mn, mx):
        return pd.Series(50.0, index=s.index)
    return 100.0 * (s - mn) / (mx - mn)


def zscore_scale(series: pd.Series) -> pd.Series:
    """
    Z-score: (x - mean) / std.
    Jeśli std == 0 lub same NaN -> zwraca 0 (wszędzie średnio).
    """
    s = series.astype(float)
    m = s.mean(skipna=True)
    std = s.std(skipna=True)
    if std is None or np.isnan(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - m) / std
    z[s.isna()] = np.nan
    return z


def merge_weights(custom: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Łączy wagi domyślne z własnymi (custom)."""
    weights = deepcopy(DEFAULT_WEIGHTS)
    if custom is None:
        return weights
    for idx_name, mapping in custom.items():
        if idx_name not in weights:
            weights[idx_name] = dict(mapping)
        else:
            weights[idx_name].update(mapping)
    return weights


def compute_weighted_index(df: pd.DataFrame, score_cols: List[str], weights_map: Dict[str, float]) -> pd.Series:
    """
    Liczy indeks jako ważoną średnią z danych kolumn (score_...).
    weights_map ma klucze odpowiadające NAZWOM BEZ 'score_' (np. 'revenue').
    """
    # budujemy wektor wag dopasowany do przekazanych kolumn
    base_names = [c.replace("score_", "") for c in score_cols]
    w = np.array([weights_map.get(name, 0.0) for name in base_names], dtype=float)
    if np.allclose(w.sum(), 0.0):
        # jeżeli wagi nie istnieją lub sumują się do 0 -> równe wagi
        w = np.ones(len(score_cols), dtype=float) / len(score_cols)
    else:
        w = w / w.sum()

    arr = df[score_cols].to_numpy(dtype=float)
    index_vals = np.nansum(arr * w, axis=1)
    # jeśli wszystkie składowe NaN -> wynik NaN
    all_nan_mask = np.all(np.isnan(arr), axis=1)
    index_vals[all_nan_mask] = np.nan
    return pd.Series(index_vals, index=df.index)


# -------------------------------------------------
# WCZYTYWANIE DANYCH FINANSOWYCH PKD
# -------------------------------------------------

def load_financial_pkd(path: str, sep: str = ";", encoding: str = "utf-8", skiprows: int = 0) -> pd.DataFrame:
    """
    Wczytuje plik finansowy wg PKD (wsk_fin.csv lub krz_pkd.csv) i zwraca
    ramkę w formacie „wide” na poziomie:
        pkd_section, year, EN, PEN, GS_I, NP, IO, LTC, STC, LTL, STL, pkd_name
    """
    print(f"[INFO] Wczytywanie danych finansowych PKD z {path} ...")

    df = pd.read_csv(path, sep=sep, encoding=encoding, skiprows=skiprows)

    # kolumny roczne = nazwy 4-cyfrowe
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    id_vars = [c for c in df.columns if c not in year_cols]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name="value_raw",
    )
    df_long["year"] = df_long["year"].astype(int)
    df_long["value"] = df_long["value_raw"].apply(parse_polish_number)

    for col in ["WSKAZNIK", "PKD", "NAZWA_PKD"]:
        if col not in df_long.columns:
            raise ValueError(f"Brak kolumny '{col}' w pliku {path}")

    df_long["indicator_code"] = df_long["WSKAZNIK"].apply(extract_indicator_code)
    df_long["pkd_section"] = df_long["PKD"].apply(extract_pkd_section)
    df_long = df_long[df_long["pkd_section"].notna()].copy()

    df_long["NAZWA_PKD"] = df_long["NAZWA_PKD"].astype(str)
    pkd_names = (
        df_long.groupby("pkd_section")["NAZWA_PKD"]
        .first()
        .reset_index()
        .rename(columns={"NAZWA_PKD": "pkd_name"})
    )

    indicators_interest = ["EN", "PEN", "GS_I", "NP", "IO", "LTC", "STC", "LTL", "STL"]
    df_long = df_long[df_long["indicator_code"].isin(indicators_interest)].copy()

    agg = (
        df_long.groupby(["pkd_section", "year", "indicator_code"])["value"]
        .sum()
        .reset_index()
    )

    wide = agg.pivot_table(
        index=["pkd_section", "year"],
        columns="indicator_code",
        values="value",
    ).reset_index()

    wide = wide.merge(pkd_names, on="pkd_section", how="left")

    return wide


def compare_financial_sets(a: pd.DataFrame, b: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Porównuje dwa zestawy finansów (wsk_fin i krz_pkd) w celu sprawdzenia zgodności.
    Nie wpływa na indeks – tylko loguje zgodność.
    """
    a_sorted = a.sort_values(["pkd_section", "year"]).reset_index(drop=True)
    b_sorted = b.sort_values(["pkd_section", "year"]).reset_index(drop=True)

    if a_sorted.shape != b_sorted.shape:
        print(f"[WARN] Różne rozmiary zbiorów: wsk_fin={a_sorted.shape}, krz_pkd={b_sorted.shape}")
        return []

    num_cols = ["EN", "PEN", "GS_I", "NP", "IO", "LTC", "STC", "LTL", "STL"]
    results = []

    for col in num_cols:
        if col not in a_sorted.columns or col not in b_sorted.columns:
            continue
        equal_mask = np.isclose(
            a_sorted[col].astype(float),
            b_sorted[col].astype(float),
            equal_nan=True,
        )
        ratio = float(equal_mask.mean())
        print(f"[INFO] Zgodność kolumny {col}: {ratio * 100:.2f}%")
        results.append((col, ratio))

    return results


# -------------------------------------------------
# BUDOWA WSKAŹNIKÓW
# -------------------------------------------------

def build_metrics(fin: pd.DataFrame) -> pd.DataFrame:
    """
    Z danych finansowych buduje wskaźniki dla CURRENT_YEAR:
      - poziomy (przychody, zysk, inwestycje, liczba firm, dług),
      - wskaźniki względne,
      - CAGR 3-letni,
      - zmienność marży z 3 lat.
    """
    df = fin.copy()

    # poziomy
    df["revenue"] = df["GS_I"]
    df["net_profit"] = df["NP"]
    df["investments"] = df["IO"]
    df["firms"] = df["EN"]
    df["profitable_firms"] = df["PEN"]
    df["bank_debt"] = df["LTC"].fillna(0.0) + df["STC"].fillna(0.0)
    df["total_liab"] = df["LTL"].fillna(0.0) + df["STL"].fillna(0.0)

    # względne
    df["profit_margin"] = df["net_profit"] / df["revenue"]
    df["profitable_share"] = df["profitable_firms"] / df["firms"]
    df["leverage_bank_to_rev"] = df["bank_debt"] / df["revenue"]
    df["leverage_tot_to_rev"] = df["total_liab"] / df["revenue"]
    df["invest_intensity"] = df["investments"] / df["revenue"]

    # CAGR 3-letni
    base = df[df["year"] == BASE_YEAR_FOR_CAGR][
        ["pkd_section", "revenue", "net_profit", "investments", "firms"]
    ].rename(
        columns={
            "revenue": "revenue_base",
            "net_profit": "net_profit_base",
            "investments": "investments_base",
            "firms": "firms_base",
        }
    )

    cur = df[df["year"] == CURRENT_YEAR].copy()
    cur = cur.merge(base, on="pkd_section", how="left")

    years_span = CURRENT_YEAR - BASE_YEAR_FOR_CAGR

    for col in ["revenue", "net_profit", "investments", "firms"]:
        cur[f"cagr_{col}"] = cur.apply(
            lambda row: safe_cagr(
                row.get(col, np.nan),
                row.get(f"{col}_base", np.nan),
                years_span,
            ),
            axis=1,
        )

    # zmienność marży z 3 lat
    last3 = df[df["year"].between(CURRENT_YEAR - 2, CURRENT_YEAR)]
    vol = (
        last3.groupby("pkd_section")["profit_margin"]
        .std()
        .reset_index()
        .rename(columns={"profit_margin": "profit_margin_std_3y"})
    )
    cur = cur.merge(vol, on="pkd_section", how="left")

    return cur


# -------------------------------------------------
# BUDOWA INDEXÓW (z wyborem normalizacji i wag)
# -------------------------------------------------

def build_indices(
    metrics: pd.DataFrame,
    normalization: str = "minmax",
    weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Buduje indeksy na podstawie tabeli wskaźników:
      - wybór metody normalizacji: 'minmax' lub 'zscore',
      - używa wag domyślnych lub nadpisanych (weights).
    """
    df = metrics.copy()
    weights_cfg = merge_weights(weights)

    # ponownie upewniamy się, że mamy podstawowe wskaźniki
    df["revenue"] = df["GS_I"]
    df["net_profit"] = df["NP"]
    df["investments"] = df["IO"]
    df["firms"] = df["EN"]
    df["profitable_firms"] = df["PEN"]
    df["bank_debt"] = df["LTC"].fillna(0.0) + df["STC"].fillna(0.0)
    df["total_liab"] = df["LTL"].fillna(0.0) + df["STL"].fillna(0.0)
    df["profit_margin"] = df["net_profit"] / df["revenue"]
    df["profitable_share"] = df["profitable_firms"] / df["firms"]
    df["leverage_bank_to_rev"] = df["bank_debt"] / df["revenue"]
    df["leverage_tot_to_rev"] = df["total_liab"] / df["revenue"]
    df["invest_intensity"] = df["investments"] / df["revenue"]

    # lista zmiennych do normalizacji
    features_positive = [
        "revenue",
        "firms",
        "cagr_revenue",
        "cagr_net_profit",
        "profit_margin",
        "profitable_share",
        "invest_intensity",
        "cagr_investments",
    ]
    features_negative = [
        "leverage_bank_to_rev",
        "leverage_tot_to_rev",
        "profit_margin_std_3y",
    ]

    # normalizacja -> kolumny score_...
    print(f"[INFO] Używana metoda normalizacji: {normalization}")
    if normalization.lower() == "minmax":
        for col in features_positive:
            if col in df.columns:
                df[f"score_{col}"] = minmax_scale(df[col])
        for col in features_negative:
            if col in df.columns:
                df[f"score_{col}"] = 100.0 - minmax_scale(df[col])
    elif normalization.lower() == "zscore":
        for col in features_positive:
            if col in df.columns:
                df[f"score_{col}"] = zscore_scale(df[col])
        for col in features_negative:
            if col in df.columns:
                # im mniej, tym lepiej -> odwracamy znak
                df[f"score_{col}"] = -zscore_scale(df[col])
    else:
        raise ValueError("normalization musi być 'minmax' albo 'zscore'.")

    # indeksy filarowe (score_... używają nazw z DEFAULT_WEIGHTS)
    df["size_index"] = compute_weighted_index(
        df,
        ["score_revenue", "score_firms"],
        weights_cfg["size"],
    )

    df["growth_index"] = compute_weighted_index(
        df,
        ["score_cagr_revenue", "score_cagr_net_profit"],
        weights_cfg["growth"],
    )

    df["profit_index"] = compute_weighted_index(
        df,
        ["score_profit_margin", "score_profitable_share"],
        weights_cfg["profit"],
    )

    df["risk_index"] = compute_weighted_index(
        df,
        ["score_leverage_bank_to_rev", "score_leverage_tot_to_rev", "score_profit_margin_std_3y"],
        weights_cfg["risk"],
    )

    df["outlook_index"] = compute_weighted_index(
        df,
        ["score_invest_intensity", "score_cagr_investments"],
        weights_cfg["outlook"],
    )

    # indeks bieżącej kondycji
    df["current_index"] = compute_weighted_index(
        df,
        ["size_index", "growth_index", "profit_index", "risk_index"],
        weights_cfg["current"],
    )

    # indeks przyszłości
    df["future_index"] = compute_weighted_index(
        df,
        ["growth_index", "outlook_index", "risk_index"],
        weights_cfg["future"],
    )

    # główny indeks
    df["main_index"] = compute_weighted_index(
        df,
        ["current_index", "future_index"],
        weights_cfg["main"],
    )

    # klasyfikacja branż
    median_current = df["current_index"].median()
    median_future = df["future_index"].median()

    def classify(row):
        c = row["current_index"]
        f = row["future_index"]
        if pd.isna(c) or pd.isna(f):
            return "Brak danych"
        if c >= median_current and f >= median_future:
            return "Gwiazdy wzrostu"
        if c < median_current and f >= median_future:
            return "Trudna teraźniejszość, dobra przyszłość"
        if c >= median_current and f < median_future:
            return "Stabilne, ale bez dynamiki"
        return "Branże ryzykowne / schyłkowe"

    df["class"] = df.apply(classify, axis=1)

    return df


# -------------------------------------------------
# MAPOWANIE PKD 2007 -> PKD 2025 (opcjonalne)
# -------------------------------------------------

def attach_pkd2025(df: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    """Dołącza kolumnę pkd_2025 na podstawie pliku mapowania (jeśli jest i działa)."""
    if not os.path.exists(mapping_path):
        print(f"[WARN] Plik mapowania PKD nie znaleziony ({mapping_path}) – pomijam mapowanie.")
        return df

    try:
        mp = pd.read_excel(mapping_path)
    except Exception as e:
        print(f"[WARN] Nie udało się wczytać mapowania PKD 2007->2025: {e}")
        return df

    if "symbol_2007" not in mp.columns or "symbol_2025" not in mp.columns:
        print("[WARN] Nieprawidłowy format mapowania PKD – brak 'symbol_2007' / 'symbol_2025'.")
        return df

    mp = mp.copy()
    mp["symbol_2007"] = mp["symbol_2007"].astype(str)
    mp_2digit = mp[mp["symbol_2007"].str.len() == 2].drop_duplicates("symbol_2007")
    mp_2digit = mp_2digit.rename(columns={"symbol_2007": "pkd_section", "symbol_2025": "pkd_2025"})

    out = df.merge(mp_2digit, on="pkd_section", how="left")
    return out


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ensure_output_dir(OUTPUT_DIR)

    if not os.path.exists(FILE_WSK_FIN):
        raise FileNotFoundError(f"Nie znaleziono pliku {FILE_WSK_FIN} (wsk_fin.csv).")

    # 1) główne dane finansowe
    fin_all = load_financial_pkd(FILE_WSK_FIN, sep=";", encoding="utf-8", skiprows=0)

    # 2) krz_pkd jako kontrola jakości
    if os.path.exists(FILE_KRZ_PKD):
        try:
            fin_krz = load_financial_pkd(FILE_KRZ_PKD, sep=",", encoding="ISO-8859-1", skiprows=1)
            print("[INFO] Porównanie wsk_fin.csv i krz_pkd.csv (numerycznie):")
            compare_financial_sets(fin_all, fin_krz)
        except Exception as e:
            print(f"[WARN] Nie udało się poprawnie wczytać / porównać krz_pkd.csv: {e}")
    else:
        print(f"[WARN] Plik {FILE_KRZ_PKD} nie został znaleziony – pomijam porównanie.")

    # 3) wskaźniki i indeksy
    metrics = build_metrics(fin_all)
    index_df = build_indices(metrics, normalization=NORMALIZATION_METHOD, weights=CUSTOM_WEIGHTS)
    index_df = attach_pkd2025(index_df, FILE_MAP_PKD)

    # 4) zapis
    output_path = os.path.join(OUTPUT_DIR, "index_branż.csv")

    columns_out = [
        "pkd_section",
        "pkd_2025",
        "pkd_name",
        "year",
        "revenue",
        "net_profit",
        "investments",
        "firms",
        "profitable_firms",
        "profit_margin",
        "profitable_share",
        "bank_debt",
        "total_liab",
        "leverage_bank_to_rev",
        "leverage_tot_to_rev",
        "invest_intensity",
        "cagr_revenue",
        "cagr_net_profit",
        "cagr_investments",
        "cagr_firms",
        "profit_margin_std_3y",
        "size_index",
        "growth_index",
        "profit_index",
        "risk_index",
        "outlook_index",
        "current_index",
        "future_index",
        "main_index",
        "class",
    ]

    existing_cols = [c for c in columns_out if c in index_df.columns]

    index_df[existing_cols].sort_values("main_index", ascending=False).to_csv(
        output_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[INFO] Gotowe! Indeks branż zapisany do: {output_path}")


if __name__ == "__main__":
    main()
