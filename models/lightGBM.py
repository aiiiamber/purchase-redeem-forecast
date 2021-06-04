# -*- coding: utf-8 -*-
# @Time : 11/28/20 9:10 PM
# @Author : ZHANG XIAOLI

import pandas as pd
import lightgbm as lgb


class LGBModel(object):

    def __init__(self, args, targets, features, val_features):
        self._args = args
        self.params = self._parse_argument()
        self.purchase_targets = self._load_targets(targets, 'total_purchase_amt')
        self.redeem_targets = self._load_targets(targets, 'total_redeem_amt')
        self.features = self._load_features(features)
        self.val_features = self._load_features(val_features).reshape((1, self.features.shape[1]))

    def train(self, test_position):
        if test_position + 30 == 0:
            feature_position = self.features.shape[0]
            target_position = 0
        else:
            target_position = test_position + 30
            feature_position = -target_position
        features = self.features[:feature_position, ]
        purchase_targets = self.purchase_targets[target_position:]
        redeem_targets = self.redeem_targets[target_position:]

        train_features = features[:-30, ]
        test_features = features[-30:, ]
        train_purchase_targets = purchase_targets[:-30]
        test_purchase_targets = purchase_targets[-30:]
        train_redeem_targets = redeem_targets[:-30]
        test_redeem_targets = redeem_targets[-30:]

        purchase_clf = self._train(
            train_features, train_purchase_targets,
            test_features, test_purchase_targets
        )
        redeem_clf = self._train(
            train_features, train_redeem_targets,
            test_features, test_redeem_targets
        )

        test_features = self.features[test_position].reshape((1, self.features.shape[1]))
        test_purchase_targets = self.purchase_targets[test_position + 1]
        test_redeem_targets = self.redeem_targets[test_position + 1]

        purchase_pred, redeem_pred = self.lgb_predict(purchase_clf, redeem_clf, test_features)
        val_purchase_pred, val_redeem_pred = self.lgb_predict(purchase_clf, redeem_clf, self.val_features)

        result = {
            'test_pur_true': test_purchase_targets,
            'test_redeem_true': test_redeem_targets,
            'test_pur_pred': purchase_pred,
            'test_redeem_pred': redeem_pred,
            'val_pur_pred': val_purchase_pred,
            'val_redeem_pred': val_redeem_pred,
        }
        return pd.DataFrame(result)

    def _train(self, train_features, train_labels, test_features, test_labels):
        train_data = lgb.Dataset(train_features, train_labels)
        val_data = lgb.Dataset(test_features, test_labels)

        clf = lgb.train(self.params,
                        train_data,
                        num_boost_round=self._args.num_round,
                        valid_sets=[train_data, val_data],
                        verbose_eval=50,
                        early_stopping_rounds=100)
        return clf

    def lgb_predict(self, purchase_clf, redeem_clf, features):
        return purchase_clf.predict(features), redeem_clf.predict(features)

    def _load_targets(self, targets, target_name):
        return targets[target_name].values

    def _load_features(self, df):
        return df.values

    def _parse_argument(self):
        params = {
            'num_leaves': self._args.num_leaves,
            'max_depth': self._args.max_depth,
            'min_data_in_leaf': self._args.min_data_in_leaf,
            'min_child_samples': 20,
            'objective': 'regression',
            'learning_rate': self._args.learning_rate,
            "boosting": "gbdt",
            "feature_fraction": self._args.feature_fraction,
            "bagging_freq": 0,
            "bagging_fraction": 0.6,
            "bagging_seed": 23,
            "metric": 'mse',
            "lambda_l1": 0.2,
            "nthread": 4,
            "verbose": -1
        }
        return params



