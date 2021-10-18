

from models.model import Model


class TransformerModel(Model):
    def __init__(self, embedding, epochs=30, batch_size=256):
        super().__init__(embedding)
        self.embedding = embedding
        self.train_y, self.test_y, self.validation_y = embedding.categorical_labels()

        self.epochs = epochs
        self.batch_size = batch_size

        self.params = {
            'evaluate_during_training': True,
            'logging_steps': 100,
            'num_train_epochs': self.epochs,
            'evaluate_during_training_steps': self.epochs * 10,
            'save_eval_checkpoints': False,
            'train_batch_size': self.batch_size,
            'eval_batch_size': self.batch_size // 2,
            'fp16': True,
        }



