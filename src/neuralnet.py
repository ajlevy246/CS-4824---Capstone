import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 82

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Non-Linear Layers
        self.relu = nn.ReLU()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3, 
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )

        # Dropout Layers
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.25)
        
        # Max Pooling Layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten Layers
        self.flatten = nn.Flatten()
        
        # Dense Layers
        self.dense1 = nn.Linear(
            in_features=64 * 22 * 22, 
            out_features=128,
        )
        self.dense2 = nn.Linear(
            in_features=128, 
            out_features=num_classes,
        )


    def forward(self, x):
        # Convolutional Layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.drop1(x)
        
        # Pooling
        x = self.pool(x)

        # Flatten and Dense Layers
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.drop2(x)
        x = self.dense2(x)

        
        return x
