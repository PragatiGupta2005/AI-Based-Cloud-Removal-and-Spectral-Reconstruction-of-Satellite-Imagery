import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQAN(nn.Module):
    """
    Learned Quality Assessment Network (QAN) for CloudClear AI.
    Replaces the metric-based evaluator with a true deep-learning regression model 
    as described in the research requirements.
    Evaluates reconstructed images against references using deep feature extraction.
    """
    def __init__(self, in_channels=8): # 4 bands reconstructed + 4 bands reference
        super(DeepQAN, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        
        # Quality score regression (predicting a composite score in [0, 1])
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Output a quality score [0, 1]
        return x
