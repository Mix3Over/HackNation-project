from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def _create_supervised_series(series_scaled: np.ndarray,
                              n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zamienia jednowymiarową serię na problem nadzorowany:
    X = [x(t-3), x(t-2), x(t-1)] -> y = x(t)
    """
    X, y = [], []
    for i in range(n_lags, len(series_scaled)):
        X.append(series_scaled[i - n_lags:i, 0])
        y.append(series_scaled[i, 0])
    return np.array(X), np.array(y)


def forecast_index_lstm(
    df: pd.DataFrame,
    pkd_section: int,
    index_col: str,
    n_ahead: int = 3,        # ile kolejnych lat prognozujemy
    n_lags: int = 3,         # ile poprzednich lat używamy jako wejście LSTM
    epochs: int = 200,
    batch_size: int = 8,
    verbose: int = 0
):
    """
    Robi forecast wybranego indeksu (kolumna index_col) dla danej sekcji PKD
    za pomocą LSTM, na bazie historycznych lat z index_branz.csv.

    Parametry
    ---------
    df : pd.DataFrame
        Dane z index_branz.csv (m.in. kolumny: pkd_section, year, index_col).
    pkd_section : int
        Numer sekcji PKD (kolumna 'pkd_section').
    index_col : str
        Nazwa kolumny z interesującym nas indeksem (np. 'main_index').
    n_ahead : int
        Na ile kolejnych lat robimy prognozę.
    n_lags : int
        Ile ostatnich obserwacji (lat) traktujemy jako input do LSTM.
    epochs : int
        Liczba epok uczenia.
    batch_size : int
        Rozmiar batcha.
    verbose : int
        Verbosity Keras (0 / 1 / 2).

    Zwraca
    ------
    model : keras.Model
        Wytrenowany model LSTM.
    result_df : pd.DataFrame
        Dane historyczne + prognoza (kolumny: 'year', index_col, 'is_forecast').
    history : keras.callbacks.History
        Obiekt historii uczenia (można użyć do wykresów błędu).
    """

    # 1. Filtrujemy dane dla konkretnej sekcji PKD
    df_pkd = df[df["pkd_section"] == pkd_section].copy()

    if df_pkd.empty:
        raise ValueError(f"Brak danych dla pkd_section={pkd_section}")

    # 2. Upewniamy się, że dane są posortowane po roku
    df_pkd = df_pkd.sort_values("year")

    if index_col not in df_pkd.columns:
        raise ValueError(f"Kolumna '{index_col}' nie istnieje w DataFrame.")

    # 3. Bierzemy serię indeksu jako float (niektóre CSV-y wczytują to jako object)
    series = df_pkd[index_col].astype(float).values.reshape(-1, 1)

    # Sprawdzamy, czy jest wystarczająco dużo punktów na LSTM
    if len(series) <= n_lags + 1:
        raise ValueError(
            f"Za mało danych historycznych dla pkd_section={pkd_section} "
            f"i index_col='{index_col}' (potrzeba > {n_lags + 1} obserwacji)."
        )

    # 4. Skalowanie do [0, 1] – LSTM dużo lepiej działa na zeskalowanych danych
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series)

    # 5. Tworzymy dane nadzorowane: sekwencje wejściowe + target
    X, y = _create_supervised_series(series_scaled, n_lags)

    # 6. Train/validation split (prosty podział czasowy 80/20)
    split_idx = max(1, int(len(X) * 0.8))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Reshape pod LSTM: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], n_lags, 1))
    X_val = X_val.reshape((X_val.shape[0], n_lags, 1))

    # 7. Budujemy model LSTM
    model = Sequential()
    model.add(LSTM(32, input_shape=(n_lags, 1)))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stop]
    )

    # 8. Prognozowanie długoterminowe (n_ahead lat)
    #    – korzystamy z ostatniego okna danych i dokładamy kolejne predykcje
    last_window = series_scaled[-n_lags:].reshape(1, n_lags, 1)

    forecasts_scaled = []
    for _ in range(n_ahead):
        next_scaled = model.predict(last_window, verbose=0)[0, 0]
        forecasts_scaled.append(next_scaled)

        # przesuwamy okno i dokładamy nową prognozę
        last_window = np.roll(last_window, -1, axis=1)
        last_window[0, -1, 0] = next_scaled

    # 9. Odskalowanie prognozy
    forecasts = scaler.inverse_transform(
        np.array(forecasts_scaled).reshape(-1, 1)
    ).ravel()

    last_year = int(df_pkd["year"].max())
    future_years = [last_year + i for i in range(1, n_ahead + 1)]

    # 10. Składamy wynik – historia + prognoza
    hist_df = df_pkd[["year", index_col]].copy()
    hist_df["is_forecast"] = False

    forecast_df = pd.DataFrame({
        "year": future_years,
        index_col: forecasts,
        "is_forecast": True
    })

    result_df = pd.concat([hist_df, forecast_df], ignore_index=True)

    return model, result_df, history


if __name__ == "__main__":
    import pandas as pd

    # 1. Wczytanie danych
    df = pd.read_csv("index_branz.csv", sep=",")  # albo sep=";" jeśli masz średniki

    # 2. Forecast np. dla hurtu (46) po 'main_index' na 5 lat do przodu
    model, result_df, history = forecast_index_lstm(
        df,
        pkd_section=46,
        index_col="main_index",
        n_ahead=5,
        n_lags=3,
        epochs=200,
        batch_size=8,
        verbose=1
    )

    print(result_df)

