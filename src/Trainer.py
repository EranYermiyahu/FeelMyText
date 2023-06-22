import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, batch_size, learning_rate):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        # self.attention_mask = attention_mask
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        decay_step_size = 10
        decay_factor = 0.5
        self.scheduler = StepLR(self.optimizer, step_size=decay_step_size, gamma=decay_factor)
        self.criterion = nn.CrossEntropyLoss()

    def forward_pass(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (input_ids, attention_mask, labels) in enumerate(self.train_loader):
                print(f'batch {i+1}')
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = self.forward_pass(input_ids, attention_mask, labels)
                _, preds = torch.max(outputs, dim=1)
                # print(f" preds is {preds} and label is {labels}")
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(self.train_loader)}')
            self.scheduler.step()
            self.validate()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct_predictions = 0
            for i, (input_ids, attention_mask, labels) in enumerate(self.val_loader):
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs, loss = self.forward_pass(input_ids, attention_mask, labels)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                val_loss += loss.item()
            acc = correct_predictions.double() / len(self.val_loader.dataset)
            print(f'Validation Loss: {val_loss / len(self.val_loader)}, Accuracy: {acc}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
