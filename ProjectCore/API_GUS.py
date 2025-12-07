import json

import requests


def pobierz_dane_o_firmach_bezpiecznie():
    # ID zmiennej 60559 (Podmioty gospodarki narodowej)
    url = "https://bdl.stat.gov.pl/api/v1/data/by-variable/60559"

    # Twój klucz API
    api_key = "2c297299-79b0-4f8f-a402-08de348d05e1"

    # WERSJA MINIMALNA PARAMETRÓW
    # Usunąłem "year" (API samo dobierze dostępne lata)
    # Usunąłem "unit-level" (API samo dobierze poziom, zazwyczaj Polskę)
    params = {
        "format": "json",
        "page-size": 10,  # Ograniczamy liczbę wyników dla czytelności
    }

    headers = {"X-ClientId": api_key, "User-Agent": "HackathonPoC/1.0"}

    print(f"1. Wysyłam zapytanie do: {url}")
    print("   (Bez filtrów na rok i województwa, aby uniknąć błędu 400)...")

    try:
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            dane = response.json()
            print("\n--- SUKCES! Pobrane dane o firmach: ---")

            if "results" in dane:
                for unit in dane["results"]:
                    # unit['name'] to nazwa jednostki (np. "POLSKA")
                    print(f"Obszar: {unit['name']}")
                    for val in unit["values"]:
                        # val['val'] to liczba firm
                        print(f"  - Rok: {val['year']} -> Liczba firm: {val['val']}")
            else:
                print("Odpowiedź nie zawiera pola 'results'.")
                print(json.dumps(dane, indent=2))  # Wypisz co przyszło

        else:
            print(f"\nBłąd! Status: {response.status_code}")
            # Próba odczytania błędu JSON, jeśli istnieje
            try:
                print("Szczegóły błędu z serwera:", response.json())
            except:
                print("Treść odpowiedzi (nie JSON):", response.text)

    except Exception as e:
        print(f"Wyjątek krytyczny: {e}")


if __name__ == "__main__":
    pobierz_dane_o_firmach_bezpiecznie()
