# -*- coding: utf-8 -*-
# @Time : 11/29/20 12:03 PM
# @Author : ZHANG XIAOLI
import math
import numpy as np


def score(purchase_true, redeem_true, purchase_pred, redeem_pred):

    def _score(true, pred):
        k = (np.log(0.2) - np.log(10)) * 10 / 3
        b = np.log(10)
        weight = [4*i / 9 for i in range(10)]
        weight.extend([1*i / 3 for i in range(10)])
        weight.extend([2 * i / 9 for i in range(10)])
        weight = np.array(weight).reshape(30, 1)
        mark = np.array([math.exp(x * k + b) if x < 0.3 else 0 for x in abs(true - pred) / true ])
        return np.squeeze(mark.dot(weight))

    purchase_score, redeem_score = _score(purchase_true, purchase_pred), _score(redeem_true, redeem_pred)
    print("purchase_score is {}, redeem_score is {}". format(purchase_score, redeem_score))
    score = 0.45 * purchase_score + 0.55 * redeem_score
    print("final score is {}".format(score))