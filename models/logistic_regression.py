from sklearn.linear_model import LogisticRegression

from models.model import Model
import numpy as np
from sklearn import metrics


class Logistic(Model):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.model_name = 'logistic'

    def objective(self, trial):
        params = {
            "class_weight": trial.suggest_categorical('class_weight', [None, 'balanced']),
            "solver": trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            "C": trial.suggest_loguniform("C", 2 ** -10, 2 ** 15),
        }
        model = LogisticRegression(**params)
        model.fit(self.train_x, self.train_y)
        probs = model.predict(self.validation_x)
        preds = np.rint(probs)
        accuracy = metrics.accuracy_score(self.validation_y, preds)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = LogisticRegression(**params)
        model.fit(self.train_x, self.train_y)
        probs = model.predict(self.test_x)
        preds = np.rint(probs)
        return preds, probs
