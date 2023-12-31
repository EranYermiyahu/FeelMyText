import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
import optuna


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, batch_size, learning_rate, epochs, optuna=False, trial=None):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.train_losses_list = []
        self.val_acc_list = []
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=len(train_loader) * self.epochs)
        self.criterion = nn.CrossEntropyLoss()
        self.trial = trial
        self.is_optuna = optuna

    def forward_pass(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask)
        # outputs = outputs.logits
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            for i, (input_ids, attention_mask, labels) in enumerate(progress_bar):
                # Early kill in optuna
                if self.is_optuna:
                    if i % 5:
                        continue
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = self.forward_pass(input_ids, attention_mask, labels)
                _, preds = torch.max(outputs, dim=1)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_description(f"Training Epoch {epoch+1}, Loss: {epoch_loss / (i+1)}")
            self.scheduler.step()
            self.train_losses_list.append(epoch_loss)
            epoch_val_acc = self.validate()
            self.val_acc_list.append(epoch_val_acc)

            if self.is_optuna:
                if self.trial is not None:
                    self.trial.report(epoch_val_acc, epoch)
                    # Handle pruning based on the intermediate value.
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()

        return self.train_losses_list, self.val_acc_list

    def validate(self):
        self.model.eval()
        progress_bar = tqdm(self.val_loader, desc="Validating")
        with torch.no_grad():
            val_loss = 0.0
            correct_predictions = 0
            for i, (input_ids, attention_mask, labels) in enumerate(progress_bar):
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs, loss = self.forward_pass(input_ids, attention_mask, labels)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                val_loss += loss.item()
                progress_bar.set_description(f"Validating, Loss: {val_loss / (i+1)}, Accuracy: {correct_predictions.double() / ((i+1)*self.batch_size)}")
            acc = correct_predictions.double() / len(self.val_loader.dataset)
            print(f'Validation Loss: {val_loss / len(self.val_loader)}, Accuracy: {acc}')
            return acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def calculate_accuracy(self, testloader, early_stop=None):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(testloader, desc="Calculating Accuracy")
            for i, (input_ids, attention_mask, labels) in enumerate(progress_bar):
                if early_stop is not None:
                    if i == early_stop:
                        break
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs, loss = self.forward_pass(input_ids, attention_mask, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
