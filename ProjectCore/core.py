from BaseCharts.CAGR import bigCagr
from BaseCharts.Derivative import derivative
from BaseCode.Data_per_sector import CreateDataPerSector
from MarkovChain.build_industry_index_years import mainIndustryIndex
from BaseCharts.Prediction import add_predictions

from MarkovChain.MarkovChain import mainMarkovChain

from typing import Dict

import pandas as pd

DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "size": {
        "revenue": 0.5,
        "firms": 0.5,
    },
    "growth": {
        "cagr_revenue": 0.5,
        "cagr_net_profit": 0.5,
    },
    "profit": {
        "profit_margin": 0.5,
        "profitable_share": 0.5,
    },
    "risk": {
        "leverage_bank_to_rev": 1 / 3,
        "leverage_tot_to_rev": 1 / 3,
        "profit_margin_std_3y": 1 / 3,
    },
    "outlook": {
        "invest_intensity": 0.5,
        "cagr_investments": 0.5,
    },
    "current": {
        "size_index": 0.20,
        "growth_index": 0.25,
        "profit_index": 0.25,
        "risk_index": 0.30,
    },
    "future": {
        "growth_index": 0.30,
        "outlook_index": 0.40,
        "risk_index": 0.30,
    },
    "main": {
        "current_index": 0.60,
        "future_index": 0.40,
    },
}

def CreateMainIndexDataPredictions(main_index_map):
    if main_index_map is None:
        main_index_map = {}

    df_table = pd.DataFrame.from_dict(main_index_map, orient='index')
    df_table = df_table.sort_index()
    df_table = df_table.reindex(sorted(df_table.columns), axis=1)
    df_table.to_csv("Data/main_index_predictions.csv", sep=";")


def CreateAndUploadData():
    #CreateDataPerSector(["EN Liczba jednostek gospodarczych"])
    #bigCagr()
    #derivative()
    mainIndustryIndex([2018, 2019, 2020, 2021, 2022, 2023, 2024], 3)

    main_index_map = add_predictions("Data/index_branż.csv", "future_index")
    if main_index_map is None:
        main_index_map = {}

    CreateMainIndexDataPredictions(main_index_map)

    #agregacja.py

    #mainMarkovChain()


CreateAndUploadData()
