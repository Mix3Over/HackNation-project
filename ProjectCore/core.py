from BaseCharts.CAGR import bigCagr
from BaseCharts.Derivative import derivative
from BaseCode.Data_per_sector import CreateDataPerSector
from MarkovChain.build_industry_index_years import mainIndustryIndex
from BaseCharts.Prediction import add_predictions

from typing import Dict

DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "size": {                 # filar: skala
        "revenue": 0.5,
        "firms": 0.5,
    },
    "growth": {               # filar: wzrost
        "cagr_revenue": 0.5,
        "cagr_net_profit": 0.5,
    },
    "profit": {               # filar: rentowność
        "profit_margin": 0.5,
        "profitable_share": 0.5,
    },
    "risk": {                 # filar: ryzyko
        "leverage_bank_to_rev": 1 / 3,
        "leverage_tot_to_rev": 1 / 3,
        "profit_margin_std_3y": 1 / 3,
    },
    "outlook": {              # filar: perspektywy
        "invest_intensity": 0.5,
        "cagr_investments": 0.5,
    },
    "current": {              # indeks bieżącej kondycji
        "size_index": 0.20,
        "growth_index": 0.25,
        "profit_index": 0.25,
        "risk_index": 0.30,
    },
    "future": {               # indeks perspektyw
        "growth_index": 0.30,
        "outlook_index": 0.40,
        "risk_index": 0.30,
    },
    "main": {                 # główny indeks
        "current_index": 0.60,
        "future_index": 0.40,
    },
}


def CreateAndUploadData():
    #CreateDataPerSector(["EN Liczba jednostek gospodarczych"])
    #bigCagr()
    #derivative()
    mainIndustryIndex([2018, 2019, 2020, 2021, 2022, 2023, 2024], 3)

    add_predictions("Data/index_branż.csv", "future_index")
    add_predictions("Data/index_branż.csv", "main_index")


CreateAndUploadData()