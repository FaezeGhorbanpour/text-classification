from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from models.model import Model


class DeepModel(Model):
    def __init__(self, embedding, epochs=30, batch_size=256):
        super().__init__(embedding)
        self.embedding = embedding
        self.train_y, self.test_y, self.validation_y = embedding.categorical_labels()

        self.epochs = epochs
        self.batch_size = batch_size

        self.train_x, self.test_x, self.validation_x = embedding.dataset.tokenizer()


        # filepath = os.path.join(self.embedding.dataset.output_path,
        #                         self.get_name() + '_model_{epoch:02d}_{val_accuracy:02f}.h5')
        # checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_loss', save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.epochs // 4, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.epochs // 2, verbose=1)
        self.callbacks_list = [reduce_lr, early_stopping]




