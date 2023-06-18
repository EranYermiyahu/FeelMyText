import torch
from torch import nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, device, batch_size=32):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward_pass(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def train(self, epochs):
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                _, loss = self.forward_pass(inputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}')
            self.validate()

    def validate(self):
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            val_loss = 0.0
            correct_predictions = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, loss = self.forward_pass(inputs, labels)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                val_loss += loss.item()
            acc = correct_predictions.double() / len(self.val_dataset)
            print(f'Validation Loss: {val_loss / len(val_loader)}, Accuracy: {acc}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
