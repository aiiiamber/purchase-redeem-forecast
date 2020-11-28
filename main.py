# -*- coding: utf-8 -*-
# @Time : 11/28/20 2:55 PM
# @Author : ZHANG XIAOLI
import pandas as pd
from utils.build_datasets import process_datasets


def main():
    # process datasets
    balance = pd.read_csv("./data/user_balance_table.csv")
    bank_shibor = pd.read_csv("./data/mfd_bank_shibor.csv")
    interest = pd.read_csv("./data/mfd_day_share_interest.csv")

    df = process_datasets(balance, bank_shibor, interest)
    pass


if __name__ == "__main__":
    main()