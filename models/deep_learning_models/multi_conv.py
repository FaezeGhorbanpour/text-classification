from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Embedding, Dense, Convolution1D, concatenate, GlobalMaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn import metrics

from models.deep_learning_models.deep_model import DeepModel


class MultiConv(DeepModel):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'multi_conv'

    def model(self, params):
        graph_in = Input(shape=(self.embedding.dataset.max_length, params['hidden_layer']), name='input')

        convs = []
        for filter_size in params['filter_size']:
            x = Convolution1D(params['conv_layer'], filter_size, padding='same', activation=params['activation'], name='conv'+str(filter_size))(graph_in)
            convs.append(x)

        graph_out = concatenate(convs, axis=1)
        graph_out = GlobalMaxPooling1D()(graph_out)
        graph = Model(graph_in, graph_out)

        model = Sequential([Embedding(input_dim=self.embedding.dataset.max_words, output_dim=params['hidden_layer'],
                                      input_length=self.embedding.dataset.max_length),
                            graph,
                            Dropout(params['dropout'], name='dropout'),
                            Dense(params['hidden_layer']//2, activation=params['activation'], name='dense1'),
                            BatchNormalization(name='normalization'),
                            Dense(self.embedding.dataset.get_labels_count(), activation='softmax', name='dense2')])

        model.compile(loss=params['loss'], optimizer=Adam(params['lr'], amsgrad=True), metrics=['accuracy'])

        return model

    def objective(self, trial):
        params = {
            "loss": 'categorical_crossentropy',
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', ]),
            "filter_size": trial.suggest_categorical('filter_size', [[2,4,6], [4,6,8], ]),
            # "optimizer": trial.suggest_categorical('optimizer', [Adam, SGD, RMSprop]),
            "dropout": trial.suggest_categorical("dropout", [0.2, 0.4, 0.6, 0.8]),
            "hidden_layer": trial.suggest_categorical("hidden_layer", [16, 64, 256]),
            "conv_layer": trial.suggest_categorical("conv_layer", [16, 64, 256]),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
        }
        model = self.model(params)
        model.fit(self.train_x, self.train_y, validation_data=(self.validation_x, self.validation_y),
                  epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=1,
                  callbacks=self.callbacks_list)
        probs = model.predict(self.validation_x)
        preds = np.argmax(probs, axis=1)
        reals = np.argmax(self.validation_y, axis=1)
        accuracy = metrics.accuracy_score(reals, preds)
        return accuracy

    def train_test(self):
        params = self.load_params()
        model = self.model(params)
        model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                  epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                  callbacks=self.callbacks_list)
        probs = model.predict(self.test_x)
        preds = np.argmax(probs, axis=1)
        return preds, probs
