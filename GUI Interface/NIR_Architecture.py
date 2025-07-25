import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class SpectralDataset(Dataset):
    def __init__(self, spectra, labels):
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx]

class SpectralCNN1D(nn.Module):
    def __init__(self, input_length=4149, num_outputs=7):
        super(SpectralCNN1D, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(0.3)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.4)
        
        # Calculate the size after convolutions and pooling
        self.feature_size = self._get_conv_output_size(input_length)
        
        # Fully connected layers for multi-output regression
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output layers for fabric types and compositions
        self.fabric_output = nn.Linear(128, 7) 
        
        self.dropout_fc = nn.Dropout(0.5)
        
    def _get_conv_output_size(self, input_length):
        # Calculate output size after all conv and pooling layers
        size = input_length
        # After conv1 + pool1
        size = size // 4
        # After conv2 + pool2  
        size = size // 4
        # After conv3 + pool3
        size = size // 2
        # After conv4 + pool4
        size = size // 2
        return size * 256  # 256 is the number of channels after conv4
    
    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, sequence_length)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc3(x))
        x = self.dropout_fc(x)
        
        # Output layer
        fabric_out = self.fabric_output(x)

        
        return fabric_out
    

NIR_model = SpectralCNN1D(input_length=100, num_outputs=7)
NIR_model.load_state_dict(torch.load("nir_model_pca_weight.pth", 
                                             map_location=torch.device('cuda'),
                                             weights_only=True))

NIR_model.eval()