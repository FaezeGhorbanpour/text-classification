import os

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from callbacks.print_results import PrintResults
from callbacks.save_best_model import SaveBestModel
from models.model import Model


class DeepModel(Model):
    def __init__(self, embedding, epochs=30, batch_size=256):
        super().__init__(embedding)
        self.embedding = embedding
        self.train_y, self.test_y, self.validation_y = embedding.categorical_labels()

        self.epochs = epochs
        self.batch_size = batch_size



    def get_callbacks(self):
        # filepath = os.path.join(self.embedding.local_datasets.output_path,
        #                         self.get_name() + '_model_{epoch:02d}_{val_accuracy:02f}.h5')
        # checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_loss', save_best_only=True, mode='max')
        print_results = PrintResults(self.validation_x, self.validation_y)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.epochs // 4, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.epochs // 2, verbose=1)
        save_best_model = SaveBestModel(mode='max', metric='val_accuracy', wanted_metric_value=0.6,
                                        path=self.embedding.dataset.output_path+'models', model_name=self.get_name())
        return [save_best_model, print_results, reduce_lr, early_stopping]



    def train(self, model):
        model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                  epochs=self.epochs, batch_size=self.batch_size, verbose=2,
                  callbacks=self.get_callbacks())
        try:
            model = keras.models.load_model(
                os.path.join(self.embedding.dataset.output_path, 'models/temporary_model.h5'))
        except:
            pass
        probs = model.predict(self.test_x)
        preds = np.argmax(probs, axis=1)
        return preds, probs