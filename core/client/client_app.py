import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Thêm path để import model
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.dirname(current_dir)
if os.path.join(core_dir, "model") not in sys.path:
    sys.path.append(os.path.join(core_dir, "model"))

from model import IDS_MLP
import copy

class LocalClient:
    def __init__(self, client_id, device, data_path=None, batch_size=64):
        self.client_id = client_id
        self.device = device
        self.data_path = data_path if data_path else f"client_{client_id}_data.pt"
        self.model = IDS_MLP().to(device)
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Load local data
        X_client, y_client = torch.load(self.data_path)
        
        # Split local data into train/val (80/20)
        n_samples = len(X_client)
        train_size = int(0.8 * n_samples)
        
        self.train_data = TensorDataset(X_client[:train_size], y_client[:train_size])
        self.val_data = TensorDataset(X_client[train_size:], y_client[train_size:])
        
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size)
        
        print(f"Client {client_id} loaded: {train_size} train, {n_samples - train_size} val samples.")

    def set_parameters(self, parameters):
        """Update local model with global parameters."""
        self.model.load_state_dict(copy.deepcopy(parameters))

    def get_parameters(self):
        """Return local model parameters."""
        return self.model.state_dict()

    def train(self, global_params=None, epochs=1, lr=0.001, proximal_mu=0.0):
        """
        Train locally and return parameters and number of samples.
        Supports FedProx if proximal_mu > 0 and global_params is provided.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Standard Loss
                loss = self.criterion(outputs, labels)
                
                # FedProx Proximal Term
                if proximal_mu > 0 and global_params is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        # L2 norm between local and global parameters
                        proximal_term += torch.norm(param - global_params[name])**2
                    loss += (proximal_mu / 2) * proximal_term
                #tach train de ung dung nhieu thuat toan khac nhau
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(), len(self.train_data)

    def evaluate(self):
        """Evaluate local model on local validation set."""
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        return avg_loss, accuracy