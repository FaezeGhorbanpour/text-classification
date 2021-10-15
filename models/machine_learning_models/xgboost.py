import xgboost
import optuna

from models.model import Model
import numpy as np
from sklearn import metrics


class Xgboost(Model):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = 'xgboost'
        self.xgb_train = xgboost.DMatrix(self.train_x, self.train_y)
        self.xgb_evaluation = xgboost.DMatrix(self.validation_x, self.validation_y)
        self.xgb_test = xgboost.DMatrix(self.test_x, self.test_y)

    def objective(self, trial):

        params = {
            'tree_method': 'gpu_hist',
            # this parameter means using the GPU when training our model to speedup the training process
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate',
                                                       [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
            'n_estimators': 4000,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        }

        if params["booster"] == "gbtree" or params["booster"] == "dart":
            params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            params["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            params["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if params["booster"] == "dart":
            params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            params["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            params["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        model = xgboost.train(params, self.xgb_train, evals=[(self.xgb_evaluation, "validation")],
                              callbacks=[pruning_callback])
        probs = model.predict(self.xgb_evaluation)
        preds = np.rint(probs)
        accuracy = metrics.accuracy_score(self.validation_y, preds)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = xgboost.train(params, self.xgb_train, evals=[(self.xgb_evaluation, "validation")], )
        probs = model.predict(self.xgb_test)
        preds = np.rint(probs)
        return preds, probs
