import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.mff_module_utilities import (FD,TFCM)


class mff_subnetwork_1(nn.Module):
    """
        Implementation of MFF Module(subnetwork=1)
        
        args: 
            tfcm_layer: number of convolutional blocks in TFCM
    """
    def __init__(self, tfcm_layer=6):
        super().__init__()
        
        # Stage 1 - Feature Extraction
        self.stage1_extraction_1x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            TFCM(cin=64,tfcm_layer = tfcm_layer),
        )
        
        # Activation
        self.activation = nn.ReLU(inplace=True)

    def forward(self, initial_features: torch.Tensor):
        """
            initial_features: spectrogram, (B, C, T, F)
        """
             
        # Stage 1
        features_1x = self.stage1_extraction_1x(initial_features)
        output_mff = self.activation(features_1x) + 0.5*initial_features  # out: [B, C, T, F]
        
        return output_mff