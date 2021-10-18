
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score

from models.transformer_models.transformer_model import TransformerModel


class Albert(TransformerModel):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'albert'


    def objective(self, trial):
        raise Exception('optuna objective method is not implemented! try without optuna')

    def train_test(self):

        model = ClassificationModel('albert', 'albert-base-v2', num_labels=self.embedding.dataset.get_labels_count()
                                    , use_cuda=True, args=self.params)
        model.train_model(train_df=(self.train_x, self.train_y), eval_df=(self.validation_x, self.validation_y))
        preds, probs, wrong_predictions = model.eval_model(eval_df=(self.validation_x, self.validation_y),
                                                           acc=accuracy_score)
        preds = np.argmax(probs, axis=1)
        return preds, probs
