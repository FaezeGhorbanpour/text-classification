from catboost import CatBoostClassifier

from models.model import Model
import numpy as np
from sklearn import metrics


class Catboost(Model):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = 'catboost'

    def objective(self, trial):
        params = {
            'loss_function': 'RMSE',
            # 'task_type': 'GPU',
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
            'max_bin': trial.suggest_int('max_bin', 200, 400),
            # 'rsm': trial.suggest_uniform('rsm', 0.3, 1.0),
            'subsample': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.006, 0.018),
            'n_estimators': 25000,
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 16),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }

        model = CatBoostClassifier(**params)
        model.fit(self.train_x, self.train_y, eval_set=[(self.validation_x, self.validation_y)],
                  early_stopping_rounds=200, verbose=False)
        probs = model.predict(self.validation_x)
        preds = np.rint(probs)
        accuracy = metrics.accuracy_score(self.validation_y, preds)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = CatBoostClassifier(**params)
        model.fit(self.train_x, self.train_y, eval_set=[(self.validation_x, self.validation_y)],
                  early_stopping_rounds=200, verbose=False)
        probs = model.predict(self.test_x)
        preds = np.rint(probs)
        return preds, probs
