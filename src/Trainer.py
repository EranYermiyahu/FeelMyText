import torch
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, batch_size, learning_rate, epochs):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.epochs = epochs
        # self.attention_mask = attention_mask
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer = AdamW(model.parameters(), lr = learning_rate, eps = 1e-8)
        decay_step_size = 10
        decay_factor = 0.5
        #self.scheduler = StepLR(self.optimizer, step_size=decay_step_size, gamma=decay_factor)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps=0,num_training_steps = len(train_loader)*self.epochs)
        self.criterion = nn.CrossEntropyLoss()


    def forward_pass(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            for i, (input_ids, attention_mask, labels) in enumerate(progress_bar):
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = self.forward_pass(input_ids, attention_mask, labels)
                _, preds = torch.max(outputs, dim=1)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_description(f"Training Epoch {epoch+1}, Loss: {epoch_loss / (i+1)}")
            self.scheduler.step()
            self.validate()

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

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def calculate_accuracy(self, testloader):
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(testloader):
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs, loss = self.forward_pass(input_ids, attention_mask, labels)
                _, predicted = torch.max(outputs.data, 1)  # Get the predicted labels

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
