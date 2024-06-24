import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.mff_module_utilities import (FD,TFCM)


class mff_subnetwork_2(nn.Module):
    """
        Implementation of MFF Module(subnetwork=2)
        
        args: 
            tfcm_layer: number of convolutional blocks in TFCM
    """
    def __init__(self, tfcm_layer=6):
        super().__init__()
        
        # Stage 1 - Feature Extraction
        self.stage1_extraction_1x = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            TFCM(cin=64,tfcm_layer = tfcm_layer),
        )
        self.stage1_extraction_4x = nn.Sequential(
            FD(cin=64, cout=128),
            nn.Dropout(p=0.05),
            TFCM(cin=128,tfcm_layer = tfcm_layer),
        )
        # Stage 1 - Feature Exchange
        self.stage1_exchange_upsample_4to1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
        )
        self.stage1_exchange_downsample_1to4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,7), stride=(1,4), padding=(0,2), dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )

        # Stage 2 - Feature Extraction
        self.stage2_extraction_1x = nn.Sequential(
            TFCM(cin=64,tfcm_layer = tfcm_layer),
        )
        self.stage2_extraction_4x = nn.Sequential(
            TFCM(cin=128,tfcm_layer = tfcm_layer),
        )
        # Stage 2 - Feature Exchange
        self.stage2_exchange_upsample_4to1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
        )

        # Activation
        self.activation = nn.ReLU(inplace=True)

    def forward(self, initial_features: torch.Tensor):
        """
            initial_features: spectrogram, (B, C, T, F)
        """
             
        # Stage 1
        features_1x = self.stage1_extraction_1x(initial_features)
        features_4x = self.stage1_extraction_4x(initial_features)
        
        up4t1 = self.stage1_exchange_upsample_4to1(features_4x)
        updated_features_1x = F.interpolate(up4t1,size=(up4t1.shape[2],up4t1.shape[3]*4),mode="nearest")\
                                + features_1x
        updated_features_4x = self.stage1_exchange_downsample_1to4(features_1x) + features_4x
        features_1x = self.activation(updated_features_1x)
        features_4x = self.activation(updated_features_4x)
        
        # Stage 2
        features_1x = self.stage2_extraction_1x(features_1x)  
        features_4x = self.stage2_extraction_4x(features_4x)        
        
        up4t1 = self.stage2_exchange_upsample_4to1(features_4x)
        updated_features_1x = F.interpolate(up4t1,size=(up4t1.shape[2],up4t1.shape[3]*4),mode="nearest") \
                                + features_1x
        output_mff = self.activation(updated_features_1x) + 0.5*initial_features  # out: [B, C, T, F]
        
        return output_mff