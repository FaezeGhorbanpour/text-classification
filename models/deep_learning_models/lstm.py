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

        model.compile(loss=params['loss'], optimizer=Adam(params['lr'], amsgrad=True), metrics=['accuracy'])

        return model

    def objective(self, trial):
        params = {
            "loss": 'categorical_crossentropy',
            "bidirectional": trial.suggest_categorical('bidirectional', [True, False]),
            # "optimizer": trial.suggest_categorical('optimizer', [Adam, SGD, RMSprop]),
            "dropout": trial.suggest_categorical("dropout", [0.2, 0.4, 0.6, 0.8]),
            "hidden_layer": trial.suggest_categorical("hidden_layer", [16, 32, 64, 128, 256]),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', ]),
        }
        model = self.model(params)
        model.fit(self.train_x, self.train_y, validation_data=(self.validation_x, self.validation_y),
                  epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2,
                  callbacks=self.callbacks_list)
        probs = model.predict(self.validation_x)
        preds = np.argmax(probs, axis=1)
        reals = np.argmax(self.validation_y, axis=1)
        accuracy = metrics.accuracy_score(reals, preds)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = self.model(params)
        model.fit(self.train_x, self.train_y, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                  validation_data=(self.test_x, self.test_y), callbacks=self.callbacks_list, )
        probs = model.predict(self.test_x)
        preds = np.argmax(probs, axis=1)
        return preds, probs
