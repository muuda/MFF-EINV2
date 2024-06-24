import torch
import torch.nn as nn
import torch.nn.functional as F

class mff_subnetwork_0(nn.Module):
    """
        When subnetwork is 0, the mff module is empty
    """
    def __init__(self):
        super().__init__()

    def forward(self, initial_features: torch.Tensor):
        
        return initial_features