import json
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.structs import TrialState
from sklearn.metrics import cohen_kappa_score


class Model:
    def __init__(self, embedding):
        self.model_name = ''
        self.embedding = embedding
        self.train_y, self.test_y, self.validation_y = self.embedding.labels_to_id()
        self.train_x, self.train_y = embedding.get_train()
        self.test_x, self.test_y = embedding.get_test()
        self.validation_x, self.validation_y = embedding.get_validation()

    def save_params(self, params):
        with open(os.path.join(self.embedding.dataset.output_path, 'params/' + self.get_name() + '.json'), 'w') as f:
            json.dump(params, f)

    def load_params(self):
        if not os.path.isfile(os.path.join(self.embedding.dataset.output_path, 'params/' + self.get_name() + '.json')):
            raise Exception('First run optuna main to find the best parameters')
        with open(os.path.join(self.embedding.dataset.output_path, 'params/' + self.get_name() + '.json'), 'r') as f:
            params = json.load(f)
        return params

    def get_name(self):
        return self.embedding.dataset.data_name + '_' + self.embedding.embedding_name + '_' + self.model_name

    def objective(self, trial):
        print('Objective function does not implemented.')
        return 0

    def optuna_main(self, n_trials=100):
        study = optuna.create_study(study_name=self.get_name(),
                                    sampler=TPESampler(),
                                    load_if_exists=True,
                                    direction="maximize",
                                    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10)
                                    )
        study.optimize(self.objective, n_trials=n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print(' Number: ', trial.number)
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        self.save_params(trial.params)

    def train_test(self):
        print('train_test function does not implemented.')
        return np.array([]), np.array([])

    def main(self):
        preds, probs = self.train_test()

        self.train_y, self.test_y, self.validation_y = self.embedding.labels_to_id()  # todo
        if len(probs.shape) > 1:
            probs = np.max(probs, axis=1)

        f_score_micro = f1_score(self.test_y, preds, average='micro', zero_division=0)
        f_score_macro = f1_score(self.test_y, preds, average='macro', zero_division=0)
        f_score_weighted = f1_score(self.test_y, preds, average='weighted', zero_division=0)
        accuarcy = accuracy_score(self.test_y, preds)

        s = ''
        print('accuracy', accuarcy)
        s += '\naccuracy\t' + str(accuarcy)
        print('f_score_micro', f_score_micro)
        s += '\nf_score_micro\t' + str(f_score_micro)
        print('f_score_macro', f_score_macro)
        s += '\nf_score_macro\t' + str(f_score_macro)
        print('f_score_weighted', f_score_weighted)
        s += '\nf_score_weighted\t' + str(f_score_weighted)

        fpr, tpr, thresholds = roc_curve(self.test_y, probs)
        AUC = auc(fpr, tpr)
        print('AUC', AUC)
        s += '\nAUC\t' + str(AUC)

        cohen_score = cohen_kappa_score(self.test_y, preds)
        print('cohen_score', cohen_score)
        s += '\ncohen_score\t' + str(cohen_score)

        report = classification_report(self.test_y, preds, target_names=self.embedding.dataset.labels_name,
                                       zero_division=0)
        print('classification report\n')
        print(report)
        s += '\nclassification report\t' + str(report)

        cm = confusion_matrix(self.test_y, preds)
        print('confusion matrix\n')
        print(cm)
        s += '\nconfusion matrix\t' + str(cm)

        with open(os.path.join(self.embedding.dataset.output_path, 'results/' + self.get_name() + '.txt'), 'w') as f:
            f.write(s)
