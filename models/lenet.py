import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=75000, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes)
        )
    
    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x,1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

if __name__ == "__main__":  
    model = LeNet5(5)
    print(model)


