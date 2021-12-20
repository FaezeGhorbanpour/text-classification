
from torch.utils.data import Dataset, DataLoader
from models.model import Model
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn


class TransformerDataLoader(Dataset):
    def __init__(self, text, sent_label, tokenizer, max_len):
        self.text = text
        self.label = sent_label
        self.max_len = max_len
        self.encoded_text = tokenizer(
            list(text), padding=True, truncation=True, max_length=max_len, return_tensors='pt'
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item_idx):
        item = {
            key: values[item_idx].clone().detach()
            for key, values in self.encoded_text.items()
        }

        item['text'] = self.text[item_idx]
        item['label'] = self.label[item_idx]
        item['id'] = item_idx

        return item


class TransformerModel(Model):
    def __init__(self, embedding, epochs=30, batch_size=64):
        super().__init__(embedding)
        self.embedding = embedding
        self.train_y, self.test_y, self.validation_y = embedding.labels_to_id()
        self.train_x = embedding.dataset.train_x
        self.test_x = embedding.dataset.test_x
        self.validation_x = embedding.dataset.validation_x

        self.epochs = epochs
        self.batch_size = batch_size

    def create_data_loader(self, df, tokenizer, max_len, bs):
        ds = TransformerDataLoader(
            df["text"].to_numpy(), df["label"].to_numpy(), tokenizer, max_len
        )

        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=1)

    def train_epoch(self, model, data_loader, loss_fn, optimizer, device, n_examples):
        model = model.train()
        print_every = 10  # prints the loss every 10 minibatches
        losses = []
        correct_predictions = 0
        # num_batches = self.embedding.train_x.shape[0] // self.batch_size
        for idx, d in enumerate(data_loader):
            input_ids = d["id"].to(device)
            attention_mask = d["mask"].to(device)
            targets = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions = correct_predictions + torch.sum(
                preds == targets
            )
            # print(type(correct_predictions.double()))
            loss_batch = loss.item()
            # if (idx % print_every) == 0:
            #     print(f"The loss in {idx}th / {num_batches} batch is {loss_batch}")
            losses.append(loss_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)

    def eval_model(self, model, data_loader, loss_fn, device, n_examples):
        model = model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["id"].to(device)
                attention_mask = d["mask"].to(device)
                targets = d["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    def get_predictions(self, model, data_loader, loss_fn, device, n_examples):
        model = model.eval()

        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["id"].to(device)
                attention_mask = d["mask"].to(device)
                targets = d["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)

                loss = loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return preds.numpy(), probs.detach().numpy()





