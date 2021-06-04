# -*- coding: utf-8 -*-
# @Time : 11/28/20 9:20 PM
# @Author : ZHANG XIAOLI

import argparse


def parse_args():
    print("Parse the parameters from command line ...")

    parser = argparse.ArgumentParser('Purchase and redeem forecast')
    parser.add_argument('--num_leaves', type=int, default=32)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--min_data_in_leaf', type=int, default=2)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    parser.add_argument('--num_round', type=int, default=1000)

    args, unparsed = parser.parse_known_args()
    return args