import os

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn import metrics

from models.deep_learning_models.deep_model import DeepModel


class Lstm(DeepModel):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'lstm'

    def model(self, params):
        if params['bidirectional']:
            second_layer = Bidirectional(
                LSTM(params['hidden_layer'] // 2, dropout=params['dropout'], recurrent_dropout=params['dropout'],
                     activation=params['activation'],))
        else:
            second_layer = LSTM(params['hidden_layer'], dropout=params['dropout'], recurrent_dropout=params['dropout'],
                                activation=params['activation'])

        model = Sequential([Embedding(self.embedding.dataset.max_words, params['hidden_layer'],
                                      input_length=self.embedding.dataset.max_length),
                            second_layer,
                            Dense(params['hidden_layer']//2, activation=params['activation'], name='dense1'),
                            BatchNormalization(name='normalization'),
                            Dense(self.embedding.dataset.get_labels_count(), activation='softmax', name='dense2'),])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(params['lr'], amsgrad=True), metrics=['accuracy'])

        return model

    def objective(self, trial):
        params = {
            "bidirectional": trial.suggest_categorical('bidirectional', [True, False]),
            # "optimizer": trial.suggest_categorical('optimizer', [Adam, SGD, RMSprop]),
            "dropout": trial.suggest_categorical("dropout", [0.2, 0.4, 0.6, 0.8]),
            "hidden_layer": trial.suggest_categorical("hidden_layer", [16, 32, 64, 128, 256]),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', ]),
        }
        model = self.model(params)
        preds, probs = self.train(model)
        reals = np.argmax(self.validation_y, axis=1)
        accuracy = metrics.accuracy_score(reals, preds)
        return accuracy


    def train_test(self):
        params = self.load_params()
        model = self.model(params)
        preds, probs = self.train(model)
        return preds, probs
