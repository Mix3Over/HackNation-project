import numpy as np
import pandas as pd
from io import StringIO

# ============================================================
# 1. STANY (dokładnie jak w danych z indeksami)
# ============================================================

states = [
    "Rolnictwo",
    "Górnictwo i kopalnictwo",
    "Przetwórstwo przemysłowe",
    "Wytwarzanie i zaopatrywanie w energię el., gaz, wodę",
    "Budownictwo",
    "Handel",
    "Transport",
    "Hotele i restauracje",
    "Pośrednictwo finansowe",
    "Obsługa nieruchomości i firm",
    "Edukacja",
    "Ochrona zdrowia",
    "Pozostała działalność usługowa",
    "Zakończyły działalność",
]

# ============================================================
# 2. BAZOWA MACIERZ PRZEJŚCIA P (2007/2008) – W %
#    (wiersze i kolumny w tej samej kolejności jak 'states')
# ============================================================

P_pct = np.array([
    [84.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.4],
    [0.0, 86.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.4],
    [0.0, 0.0, 94.5, 0.0, 0.0, 0.3, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 4.5],
    [0.0, 0.0, 0.0, 85.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.3],
    [0.0, 0.0, 0.6, 0.0, 81.9, 2.3, 0.3, 0.0, 1.4, 0.0, 0.0, 0.0, 0.3, 13.2],
    [0.0, 0.0, 0.3, 0.0, 3.8, 77.3, 1.5, 0.0, 2.1, 0.0, 0.0, 0.0, 0.5, 14.5],
    [0.0, 0.0, 0.2, 0.0, 2.1, 2.5, 74.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.4, 17.7],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 86.1, 4.8, 0.0, 0.0, 0.0, 0.9, 8.1],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.2, 7.3, 82.4, 0.0, 0.1, 0.2, 1.3, 7.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86.4, 1.5, 1.5, 0.3, 10.3],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86.8, 0.8, 0.6, 11.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 87.4, 1.5, 11.1],
    [11.6, 0.3, 5.0, 1.2, 5.9, 15.9, 2.7, 2.3, 23.7, 8.2, 7.3, 15.9, 0.0, 0.0],
    [13.0, 11.8, 13.7, 15.9, 12.7, 13.5, 21.5, 23.7, 12.3, 12.7, 13.7, 10.4, 0.0, 0.0],
])

# Zamiana na prawdopodobieństwa i normalizacja wierszy
P = P_pct / 100.0
P = P / P.sum(axis=1, keepdims=True)

# ============================================================
# 3. FUNKCJE MARKOWA (stałe P)
# ============================================================

def simulate_markov(P, x0, years):
    """
    Homogeniczny łańcuch Markowa: x_{t+1} = x_t P
    Zwraca macierz (years+1 x n) z rozkładami po latach.
    """
    x = np.array(x0, dtype=float)
    x /= x.sum()
    history = [x]
    for _ in range(years):
        x = x @ P
        history.append(x)
    return np.vstack(history)

def stationary_distribution(P):
    """
    Rozkład stacjonarny π taki, że πP = π.
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    v = np.real(eigvecs[:, idx])
    v[v < 0] = 0
    pi = v / v.sum()
    return pi

# ============================================================
# 4. INDEKS ROZWOJU BRANŻ (2018–2027) – Z SUROWEGO TEKSTU
# ============================================================

csv_text = """Sector;2018;2019;2020;2021;2022;2023;2024;2025;2026;2027
Rolnictwo;32,4;47,33;41,19;47,8;39,36;48,62;57,49;55,74;58,46;61,17
Górnictwo i kopalnictwo;25,65;46,18;39,45;43,39;36,87;48,99;53,76;54,52;57,64;60,76
Przetwórstwo przemysłowe;35,83;50,01;43,79;48,74;40,99;49,91;55,89;54,62;56,66;58,7
Wytwarzanie i zaopatrywanie w energię el., gaz, wodę;31,8;44,28;42,74;49,12;58,26;47,25;55,56;60,25;63,56;66,87
Budownictwo;33,56;52,68;46,71;53,6;42,08;49,4;56,07;55,77;57,79;59,8
Handel;34,85;49,87;44,27;49,37;38,97;49,12;56,37;54,37;56,43;58,5
Transport;36,78;50,13;43,33;48,58;40,39;49,51;56,04;54,05;55,96;57,88
Hotele i restauracje;31,22;46,23;41,55;48,65;37,72;47,49;56,27;54,71;57,35;59,98
Pośrednictwo finansowe;35,22;52,85;45,98;51,83;41,51;51,3;55,84;55,54;57,48;59,42
Obsługa nieruchomości i firm;34,01;48,96;35,68;40,09;34,24;50,29;59,21;54,19;56,93;59,68
Edukacja;34,47;48,52;43,14;50,4;37,69;45,64;55,15;52,26;54,08;55,89
Ochrona zdrowia;31,66;41,23;34,24;38,3;28,12;37,59;48,21;42,23;43,53;44,82
Pozostała działalność usługowa;30,38;46,61;45,99;51,45;37,65;47,98;55,18;54,86;57,32;59,78
Zakończyły działalność;34,67;51,56;45,11;52,35;41,17;51,53;58,38;57,41;59,81;62,21
"""

idx_df = pd.read_csv(StringIO(csv_text), sep=";", decimal=",")
idx_df.set_index("Sector", inplace=True)

# dopasowanie kolejności do 'states'
idx_df = idx_df.loc[states]

# ============================================================
# 5. BUDOWA CZASOWO-ZMIENNEJ MACIERZY P_t NA PODSTAWIE INDEKSU
# ============================================================

def build_time_dependent_P(P_base, idx_df, alpha=1.0):
    """
    Tworzy słownik {rok: P_t}, gdzie P_t to macierz przejścia
    ważona indeksem rozwoju w danym roku:
    
       a_j(t) = ( I_j(t) / mean_I(t) )^alpha
       P_ij(t) ~ P_base_ij * a_j(t), z normalizacją wierszy.
    """
    years = idx_df.columns
    n = P_base.shape[0]
    P_t_dict = {}

    for year in years[:-1]:  # 2018->2019, ..., 2026->2027
        I = idx_df[year].values.astype(float)
        mean_I = I.mean()
        attractiveness = (I / mean_I) ** alpha  # a_j(t)

        P_weighted = P_base * attractiveness.reshape(1, n)
        P_year = P_weighted / P_weighted.sum(axis=1, keepdims=True)

        P_t_dict[int(year)] = P_year

    return P_t_dict

def simulate_time_inhomogeneous(P_t_dict, x0, start_year, end_year):
    """
    Niejednorodny w czasie łańcuch Markowa:
      x_{t+1} = x_t P_t
    P_t_dict – {rok: P_rok} dla przejścia rok -> rok+1
    """
    years = list(range(start_year, end_year + 1))
    x = np.array(x0, dtype=float)
    x /= x.sum()

    history = {start_year: x.copy()}

    for y in years[:-1]:
        P_y = P_t_dict[y]
        x = x @ P_y
        history[y + 1] = x.copy()

    df_hist = pd.DataFrame(history).T
    df_hist.columns = states
    df_hist.index.name = "Rok"
    return df_hist

# ============================================================
# 6. MAIN – przykładowe użycie
# ============================================================

if __name__ == "__main__":
    print("Kontrola sum wierszy P (powinno być 1):")
    print(np.round(P.sum(axis=1), 3))

    # --- 6.1. Homogeniczny łańcuch Markowa (stałe P) ---
    x0 = np.ones(len(states)) / len(states)  # start: równe udziały
    years = 10

    hist_const = simulate_markov(P, x0, years)
    df_const = pd.DataFrame(hist_const, columns=states)
    df_const.index.name = "Krok"
    print("\n=== Rozkład przy stałej macierzy P (10 kroków) ===")
    print(df_const.round(4))

    # --- 6.2. Rozkład stacjonarny ---
    pi = stationary_distribution(P)
    print("\n=== Rozkład stacjonarny (π) ===")
    for s, v in zip(states, pi):
        print(f"{s:50s}: {v:.4f}")

    # --- 6.3. Łańcuch z indeksem rozwoju (P_t) ---
    P_t = build_time_dependent_P(P, idx_df, alpha=1.0)
    df_dyn = simulate_time_inhomogeneous(P_t, x0, start_year=2018, end_year=2027)

    print("\n=== Rozkład sektorów z uwzględnieniem indeksu rozwoju (2018–2027) ===")
    print(df_dyn.round(4))
