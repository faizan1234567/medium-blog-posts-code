import torch
import torch.nn as nn
import torchvision
import numpy as np

class SimpleNN(nn.Module):
  def __init__(self, in_channels: int = 1, 
               kernel_size: int= 3, stride: 
               int= 1, n_classes: int = 10,
               dropout: float = 0.3):
    """Simple Neural network architecture for FashionMNIST classification
    
    Args:
      in_channels: int
      kernel_size: int
      stride: int
      n_classes: int
      dropout: float
      """
    super(SimpleNN, self).__init__()
    self.in_channels = in_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.n_classes = n_classes
    self.dropout = dropout
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    self.model =  nn.Sequential(nn.Conv2d(self.in_channels, 16, kernel_size = self.kernel_size, stride = self.stride),
                          nn.BatchNorm2d(16),
                          nn.ReLU(),
                          nn.MaxPool2d(kernel_size = 2),
                          nn.Flatten(),
                          nn.Linear(13 * 13 * 16, 256),
                          nn.BatchNorm1d(256),
                          nn.ReLU(),
                          nn.Dropout(self.dropout),
                          nn.Linear(256, self.n_classes)).to(self.device)
    
  def forward(self, x):
    output = self.model(x)
    return output