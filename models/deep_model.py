from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from models.model import Model


class DeepModel(Model):
    def __init__(self, embedding, epochs=30, batch_size=256):
        super().__init__(embedding)
        self.embedding = embedding
        embedding.categorical_labels()

        self.epochs = epochs
        self.batch_size = batch_size

        self.tokenizer(embedding.dataset)


        # filepath = os.path.join(self.embedding.dataset.output_path,
        #                         self.get_name() + '_model_{epoch:02d}_{val_accuracy:02f}.h5')
        # checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_loss', save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.epochs // 4, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.epochs // 2, verbose=1)
        self.callbacks_list = [reduce_lr, early_stopping]


    def tokenizer(self, dataset):

        tokenizer_obj = Tokenizer()

        tokenizer_obj.fit_on_texts(dataset.train_x)

        train_x_tokens = tokenizer_obj.texts_to_sequences(dataset.train_x)
        test_x_tokens = tokenizer_obj.texts_to_sequences(dataset.test_x)
        validation_x_tokens = tokenizer_obj.texts_to_sequences(dataset.validation_x)

        self.train_x = pad_sequences(train_x_tokens, maxlen=dataset.max_length, padding='post')
        self.test_x = pad_sequences(test_x_tokens, maxlen=dataset.max_length, padding='post')
        self.validation_x = pad_sequences(validation_x_tokens, maxlen=dataset.max_length, padding='post')



