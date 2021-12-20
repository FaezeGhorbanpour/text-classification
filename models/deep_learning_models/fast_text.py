from embeddings.embedding import Embedding
from models.model import Model
import numpy as np
import pandas as pd
from sklearn import metrics
import fasttext, os



class Fasttext(Model):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'fasttext'
        self.train_path = os.path.join(self.embedding.dataset.output_path, 'fasttext/train.csv')
        self.test_path = os.path.join(self.embedding.dataset.output_path, 'fasttext/test.csv')
        self.validation_path = os.path.join(self.embedding.dataset.output_path, 'fasttext/validation.csv')
        self.save_train_test()

    def save_train_test(self,):

        train_y = ['__label__' + str(i) for i in self.train_y]
        pd.DataFrame({'text': self.embedding.dataset.train_x,
                      'label': train_y}).to_csv(self.train_path, sep='\t',header=False, index=False)

        validation_y = ['__label__' + str(i) for i in self.validation_y]
        pd.DataFrame({'text': self.embedding.dataset.validation_x,
                      'label': validation_y}).to_csv(self.validation_path, sep='\t',header=False, index=False)

        test_y = ['__label__' + str(i) for i in self.test_y]
        pd.DataFrame({'text': self.embedding.dataset.test_x,
                      'label': test_y}).to_csv(self.test_path, sep='\t',header=False, index=False)


    def objective(self, trial):
        params = {
            "dim": trial.suggest_categorical('dim', [100, 200, 300]),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
            "wordNgrams": trial.suggest_categorical("wordNgrams", [1, 2, 4]),
        }
        model = fasttext.train_supervised(self.train_path, **params)

        def score(text: str) -> int:
            labels, prob = model.predict(text, 1)
            pred = int(labels[0][-1])
            return pred

        preds = [score(i) for i in self.embedding.dataset.test_x]
        accuracy = metrics.accuracy_score(self.validation_y, preds)
        return accuracy

    def train(self):
        params = self.load_params()
        model = fasttext.train_supervised(self.train_path, **params)
        return model

    def train_test(self):
        params = self.load_params()
        model = fasttext.train_supervised(self.train_path, **params)

        def score(text: str) -> (int, float):
            labels, prob = model.predict(text, 1)
            pred = int(labels[0][-1])
            return pred, prob
        preds = np.array([score(i)[0] for i in self.embedding.dataset.test_x])
        probs = np.array([score(i)[1] for i in self.embedding.dataset.test_x])
        return preds, probs
