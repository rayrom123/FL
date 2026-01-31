import torch
import torch.nn as nn
import torch.nn.functional as F

class IDS_MLP(nn.Module):
    def __init__(self, input_dim=31, num_classes=34): 
        """
        input_dim: 31 (Sau feature engineering từ 46 features ban đầu)
        num_classes: 34 (Số lượng nhãn tấn công max ID là 33 + 1)
        """
        super(IDS_MLP, self).__init__()
        
        # Layer 1: Input -> Hidden 1
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128) # Giúp hội tụ nhanh hơn trong FL
        self.dropout1 = nn.Dropout(0.3) # Tránh học vẹt
        
        # Layer 2: Hidden 1 -> Hidden 2
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        # Layer 3: Hidden 2 -> Hidden 3
        self.fc3 = nn.Linear(64, 32)
        
        # Output Layer: Hidden 3 -> Classes
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 45)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        # Không cần Softmax ở đây nếu dùng CrossEntropyLoss
        x = self.output(x)
        return x