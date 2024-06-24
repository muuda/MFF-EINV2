import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.mff_module_utilities import (FD,TFCM)
from methods.utils.mff_module_subnetworks.subnetwork_0 import mff_subnetwork_0
from methods.utils.mff_module_subnetworks.subnetwork_1 import mff_subnetwork_1
from methods.utils.mff_module_subnetworks.subnetwork_2 import mff_subnetwork_2
from methods.utils.mff_module_subnetworks.subnetwork_3 import mff_subnetwork_3
from methods.utils.mff_module_subnetworks.subnetwork_4 import mff_subnetwork_4

class mff_module(nn.Module):
    """
        Implementation of MFF Module
        
        args: 
            subnetwork: number of parallel subnetworks, select from [0, 1, 2, 3, 4]
            tfcm_layer: number of convolutional blocks in TFCM
    """
    def __init__(self, subnetwork=3, tfcm_layer=6):
        super().__init__()
        
        if(subnetwork==0):
            self.mff = mff_subnetwork_0()
        elif(subnetwork==1):
            self.mff = mff_subnetwork_1(tfcm_layer=tfcm_layer)
        elif(subnetwork==2):
            self.mff = mff_subnetwork_2(tfcm_layer=tfcm_layer)
        elif(subnetwork==3):
            self.mff = mff_subnetwork_3(tfcm_layer=tfcm_layer)
        elif(subnetwork==4):
            self.mff = mff_subnetwork_4(tfcm_layer=tfcm_layer)
        else:
            raise ValueError(f"Unsupported subnetwork: {subnetwork}")

    def forward(self, initial_features: torch.Tensor):
        """
            initial_features: spectrogram, (B, C, T, F)
        """
             
        output_mff = self.mff(initial_features) # out: [B, C, T, F]
        
        return output_mff