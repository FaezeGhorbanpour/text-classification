import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score
from torch import nn, optim
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch

from tqdm import tqdm

from models.transformer_models.transformer_model import TransformerModel


class ParsBert(nn.Module):
    def __init__(self, embedding, ):
        super(ParsBert, self).__init__()
        self.bert = AutoModel.from_pretrained("/home/faeze/huggingface_models/bert-base-parsbert-uncased")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, embedding.dataset.get_labels_count())

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state[:, 0, :]
        output = self.dropout(last_hidden_state)
        return self.out(output)


class ParsBertClassifier(TransformerModel):
    def __init__(self, embedding):
        super().__init__(embedding)
        self.model_name = 'parsbert'
        self.embedding = embedding
        self.tokenizer = AutoTokenizer.from_pretrained("/home/faeze/huggingface_models/bert-base-parsbert-uncased")
        train = pd.DataFrame({'text': self.train_x, 'label': self.train_y})
        self.train_data_loader = self.create_data_loader(train, self.tokenizer, embedding.dataset.max_length,
                                                         self.batch_size)
        validation = pd.DataFrame({'text': self.validation_x, 'label': self.validation_y})
        self.val_data_loader = self.create_data_loader(validation, self.tokenizer, embedding.dataset.max_length,
                                                       self.batch_size)
        self.device = self.embedding.dataset.device

    def train(self, param):
        model = ParsBert(self.embedding)
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=param['lr'])
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        val_acc = 0

        for epoch in tqdm(range(self.batch_size)):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print("-" * 10)

            train_acc, train_loss = self.train_epoch(
                model, self.train_data_loader, loss_fn, optimizer, self.device, len(self.train_x)
            )

            print(f"Epoch: {epoch}, Train loss: {train_loss}, accuracy: {train_acc}")

            val_acc, val_loss = self.eval_model(
                model, self.val_data_loader, loss_fn, self.device, len(self.validation_x)
            )

            print(f"Epoch: {epoch}, Val loss: {val_loss}, accuracy: {val_acc}")

            if val_acc > 0.80:
                torch.save(model, self.get_name() + ".bin")
                best_accuracy = val_acc
        return val_acc

    def objective(self, trial):
        param = {
            'lr': trial.suggest_loguniform('head_lr', 1e-5, 1e-1),
        }
        return self.train(param)

    def train_test(self):
        model = torch.load(self.get_name() + '.bin')
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        test = pd.DataFrame({'text': self.test_x, 'label': self.test_y})
        self.test_data_loader = self.create_data_loader(test, self.tokenizer, self.embedding.dataset.max_length,
                                                        self.batch_size)
        preds, probs = self.get_predictions(model, self.val_data_loader, loss_fn, self.device, len(self.test_x))
        # preds = np.argmax(probs, axis=1)
        return preds, probs
