import numpy as np

# --- 1. Dane wejściowe -------------------------------------------------------

states = [
    "Rolnictwo",
    "Gornictwo_i_kopalnictwo",
    "Przetworstwo_przemyslowe",
    "Energia_gaz_woda",
    "Budownictwo",
    "Handel",
    "Transport",
    "Hotele_i_restauracje",
    "Posrednictwo_finansowe",
    "Obsluga_nieruchomosci_i_firm",
    "Edukacja",
    "Ochrona_zdrowia",
    "Pozostala_dzialalnosc_uslugowa",
    "Zakonczone_dzialalnosc",
]

absorbing_index = 13  # indeks stanu "Zakonczone_dzialalnosc"

P = np.array([
    [0.865, 0.   , 0.005, 0.001, 0.004, 0.006, 0.001, 0.001, 0.   , 0.001, 0.   , 0.   , 0.   , 0.118],
    [0.   , 0.866, 0.009, 0.   , 0.009, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.116],
    [0.   , 0.   , 0.856, 0.   , 0.002, 0.006, 0.001, 0.   , 0.   , 0.002, 0.   , 0.   , 0.   , 0.133],
    [0.   , 0.   , 0.   , 0.945, 0.005, 0.   , 0.001, 0.   , 0.   , 0.002, 0.   , 0.   , 0.016, 0.030],
    [0.   , 0.   , 0.005, 0.   , 0.857, 0.003, 0.001, 0.   , 0.   , 0.007, 0.   , 0.   , 0.   , 0.127],
    [0.   , 0.   , 0.006, 0.   , 0.002, 0.841, 0.001, 0.   , 0.   , 0.003, 0.   , 0.   , 0.   , 0.147],
    [0.   , 0.   , 0.003, 0.   , 0.004, 0.008, 0.819, 0.   , 0.   , 0.004, 0.   , 0.   , 0.002, 0.159],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.007, 0.   , 0.773, 0.002, 0.002, 0.   , 0.001, 0.   , 0.215],
    [0.   , 0.   , 0.007, 0.   , 0.003, 0.007, 0.   , 0.   , 0.754, 0.007, 0.   , 0.   , 0.   , 0.223],
    [0.   , 0.   , 0.003, 0.   , 0.003, 0.002, 0.001, 0.001, 0.   , 0.862, 0.   , 0.   , 0.001, 0.127],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.005, 0.   , 0.   , 0.   , 0.015, 0.713, 0.   , 0.   , 0.267],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.001, 0.   , 0.001, 0.   , 0.861, 0.   , 0.137],
    [0.001, 0.001, 0.   , 0.013, 0.002, 0.001, 0.   , 0.001, 0.   , 0.010, 0.   , 0.   , 0.868, 0.104],
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 1.000],
])


# --- 2. Funkcje narzędziowe ---------------------------------------------------

def decompose_absorbing(P, absorbing_index):
    """
    Rozkłada macierz P na Q (stany przejściowe) i R (przejście do stanu pochłaniającego).
    Zakładamy, że stan pochłaniający jest jeden (absorbing_index).
    """
    n = P.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[absorbing_index] = False
    Q = P[mask][:, mask]          # 13x13
    R = P[mask][:, [absorbing_index]]  # 13x1
    return Q, R, mask


def fundamental_matrix(Q):
    """N = (I - Q)^(-1)"""
    I = np.eye(Q.shape[0])
    return np.linalg.inv(I - Q)


def expected_time_to_absorption(N):
    """t_i = oczekiwana liczba kroków do absorpcji startując ze stanu i."""
    ones = np.ones((N.shape[0], 1))
    return (N @ ones).flatten()


def survival_curve(P, start_index, absorbing_index, horizon):
    """
    Krzywa przeżycia: prawdopodobieństwo, że firma jeszcze NIE jest
    w stanie pochłaniającym po k krokach, dla k=1..horizon.
    """
    n = P.shape[0]
    dist = np.zeros(n)
    dist[start_index] = 1.0
    surv = []
    for _ in range(horizon):
        dist = dist @ P
        surv.append(1.0 - dist[absorbing_index])
    return np.array(surv)


def stationary_eigenvector_for_13x13(Q):
    """
    Liczymy wektor własny dla macierzy 13x13 po warunkowaniu na to,
    że firma nie upadła (czyli normalizujemy wiersze do sumy 1).
    """
    row_sums = Q.sum(axis=1, keepdims=True)
    Q_cond = Q / row_sums  # prawdziwy łańcuch Markowa na 'żywych' firmach

    # Szukamy wektora własnego dla wartości własnej 1
    eigvals, eigvecs = np.linalg.eig(Q_cond.T)
    idx = np.argmin(np.abs(eigvals - 1))
    v = np.real(eigvecs[:, idx])
    v = np.abs(v)
    v = v / v.sum()
    return v, Q_cond


# --- 3. Główna część analizy --------------------------------------------------

if __name__ == "__main__":
    Q, R, mask = decompose_absorbing(P, absorbing_index)

    # 3.1. Macierz fundamentalna i oczekiwany czas życia
    N = fundamental_matrix(Q)
    t = expected_time_to_absorption(N)

    print("Oczekiwany czas do zakonczenia dzialalnosci (w latach):")
    live_states = [s for i, s in enumerate(states) if i != absorbing_index]
    for s, val in zip(live_states, t):
        print(f"{s:35s}: {val:5.2f}")

    # 3.2. Krzywa przeżycia (przykład: 10 lat dla Handlu)
    horizon = 10
    start_state = states.index("Handel")
    surv = survival_curve(P, start_state, absorbing_index, horizon)
    print("\nPrzyklad - Handel, prawdopodobienstwo istnienia po k latach:")
    for k, p in enumerate(surv, start=1):
        print(f"k={k:2d}: {p:.3f}")

    # 3.3. Wektor własny dla macierzy 13x13 (warunkowej)
    v, Q_cond = stationary_eigenvector_for_13x13(Q)
    print("\nWektor wlasny (stacjonarny rozklad dla Q_cond, suma=1):")
    for s, val in zip(live_states, v):
        print(f"{s:35s}: {val:.4f}")
