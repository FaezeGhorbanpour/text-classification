
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score

from models.deep_learning_models.deep_model import DeepModel


class Xlnet(DeepModel):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'xlnet'


    def objective(self, trial):
        raise Exception('optuna objective method is not implemented! try without optuna')

    def train_test(self):
        params = {
            'evaluate_during_training': True,
            'logging_steps': 100,
            'num_train_epochs': self.epochs,
            'evaluate_during_training_steps': self.epochs * 10,
            'save_eval_checkpoints': False,
            'train_batch_size': self.batch_size,
            'eval_batch_size': self.batch_size // 2,
            'fp16': True,
        }
        model = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=self.embedding.dataset.get_labels_count()
                                    , use_cuda=True, args=params)
        model.train_model(train_df=(self.train_x, self.train_y), eval_df=(self.validation_x, self.validation_y))
        preds, probs, wrong_predictions = model.eval_model(eval_df=(self.validation_x, self.validation_y),
                                                           acc=accuracy_score)
        preds = np.argmax(probs, axis=1)
        return preds, probs
