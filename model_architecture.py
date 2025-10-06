import torch
from torch import nn

class model(nn.Module):
  """
  Defines the neural network architecture.
  """
  def __init__(self, input_features=78, hidden_units=32, output_features=1):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_features)
    )

  def forward(self, x):
    return self.layer_stack(x)