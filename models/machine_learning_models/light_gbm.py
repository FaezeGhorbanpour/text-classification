
import lightgbm
import optuna

from models.model import Model
import numpy as np
from sklearn import metrics


class LightGBM(Model):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = 'light_gbm'

        self.lgb_train = lightgbm.Dataset(self.train_x, self.train_y)
        self.lgb_eval = lightgbm.Dataset(self.validation_x, self.validation_y, reference=self.lgb_train)
        self.lgb_test = lightgbm.Dataset(self.test_x, self.test_y, reference=self.lgb_train)

    def objective(self, trial):
        params = {
            'feature_pre_filter': False,
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        model = lightgbm.train(
            params, self.lgb_train, valid_sets=[self.lgb_eval], verbose_eval=False, callbacks=[pruning_callback]
        )

        preds = model.predict(self.validation_x)
        pred_labels = np.rint(preds)
        accuracy = metrics.accuracy_score(self.validation_y, pred_labels)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = lightgbm.train(
            params, self.lgb_train, valid_sets=[self.lgb_eval], verbose_eval=False,
        )
        probs = model.predict(self.test_x)
        preds = np.rint(probs)
        return preds, probs


