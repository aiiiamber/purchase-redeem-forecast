# -*- coding: utf-8 -*-
# @Time : 11/28/20 2:55 PM
# @Author : ZHANG XIAOLI
import os
import pandas as pd

from utils.conf import parse_args
from utils.build_datasets import process_datasets
from utils.mestrics import score

from models.lightGBM import LGBModel


def save_result(purchase_pred, redeem_pred, file_name):
    date = [i for i in range(20140901, 20140931)]
    df = pd.DataFrame({
        'date': date,
        'purchase_pred': purchase_pred,
        'redeem_pred': redeem_pred
    })
    df.to_csv(os.path.join('./result', file_name), index=False, header=False)

def main():
    args = parse_args()

    print("process datasets ...")
    balance = pd.read_csv("./data/user_balance_table.csv")
    bank_shibor = pd.read_csv("./data/mfd_bank_shibor.csv")
    interest = pd.read_csv("./data/mfd_day_share_interest.csv")

    df = process_datasets(balance, bank_shibor, interest)

    print("split features and targets ...")
    targets = df.loc[1:, ['total_purchase_amt', 'total_redeem_amt']]
    columns = list(df.columns)[1:]
    # columns.remove('total_purchase_amt')
    # columns.remove('total_redeem_amt')
    features = df.loc[:df.shape[0]-2, columns]
    val_features = df.loc[df.shape[0]-1, columns]

    print("train the model ... ")
    model = LGBModel(args, targets, features, val_features)
    result_df = pd.DataFrame(columns=[
        'test_pur_true', 'test_redeem_true', 'test_pur_pred',
        'test_redeem_pred', 'val_pur_pred', 'val_redeem_pred'
                                      ])
    for i in range(-30, 0):
        result = model.train(i)
        result_df = pd.concat([result_df, result])

    score(result_df['test_pur_true'], result_df['test_redeem_true'],
          result_df['test_pur_pred'], result_df['test_redeem_pred'])

    save_result(result_df['val_pur_pred'], result_df['val_redeem_pred'], 'lgb_result_00002.csv')


if __name__ == "__main__":
    main()