# -*- coding: utf-8 -*-
# @Time : 11/28/20 4:28 PM
# @Author : ZHANG XIAOLI

import numpy as np
import pandas as pd


def process_datasets(balance: pd.DataFrame, bank_shibor: pd.DataFrame, interest: pd.DataFrame) -> pd.DataFrame:
    # process balance
    balance = balance.fillna(0)
    columns = list(balance.columns)
    grouped = balance.groupby('report_date')[columns[2:]].sum().reset_index()

    # process bank shibor
    bank_shibor = bank_shibor.rename(columns={'mfd_date': 'report_date'})
    df = pd.merge(grouped, bank_shibor, on='report_date', how='left')

    # process interest
    interest = interest.rename(columns={'mfd_date': 'report_date'})
    df = pd.merge(df, interest, on='report_date', how='left')

    df = df.fillna(method='ffill')

    return df
