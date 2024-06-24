import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.mff_module_utilities import (FD,TFCM)


class mff_subnetwork_3(nn.Module):
    """
        Implementation of MFF Module(subnetwork=3)
        
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
        self.stage2_extraction_16x = nn.Sequential(
            FD(cin=128, cout=256),
            nn.Dropout(p=0.05),
            TFCM(cin=256,tfcm_layer = tfcm_layer),
        )
        # Stage 2 - Feature Exchange
        self.stage2_exchange_upsample_4to1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
        )
        self.stage2_exchange_upsample_16to1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
        )
        self.stage2_exchange_downsample_1to4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,7), stride=(1,4), padding=(0,2), dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.stage2_exchange_upsample_16to4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
        )
        self.stage2_exchange_downsample_1to16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,7), stride=(1,4), padding=(0,2), dilation=1, bias=False),
            nn.Conv2d(64, 256, kernel_size=(1,7), stride=(1,4), padding=(0,2), dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.stage2_exchange_downsample_4to16 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1,7), stride=(1,4), padding=(0,2), dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )

        # Stage 3 - Feature Extraction
        self.stage3_extraction_1x = nn.Sequential(
            TFCM(cin=64,tfcm_layer = tfcm_layer),
        )
        self.stage3_extraction_4x = nn.Sequential(
            TFCM(cin=128,tfcm_layer = tfcm_layer),
        )
        self.stage3_extraction_16x = nn.Sequential(
            TFCM(cin=256,tfcm_layer = tfcm_layer),
        )
        # Stage 3 - Feature Exchange
        self.stage3_exchange_upsample_4to1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
        )
        self.stage3_exchange_upsample_16to1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1)),
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
        features_16x = self.stage2_extraction_16x(features_4x)
        features_4x = self.stage2_extraction_4x(features_4x)        
        
        up4t1 = self.stage2_exchange_upsample_4to1(features_4x)
        up16t1 = self.stage2_exchange_upsample_16to1(features_16x)
        updated_features_1x = F.interpolate(up4t1,size=(up4t1.shape[2],up4t1.shape[3]*4),mode="nearest") \
                                + F.interpolate(up16t1,size=(up16t1.shape[2],up16t1.shape[3]*16),mode="nearest") \
                                + features_1x
        up16t4 = self.stage2_exchange_upsample_16to4(features_16x)
        updated_features_4x = F.interpolate(up16t4,size=(up16t4.shape[2],up16t4.shape[3]*4),mode="nearest")\
                                + self.stage2_exchange_downsample_1to4(features_1x)\
                                + features_4x 
        updated_features_16x = self.stage2_exchange_downsample_4to16(features_4x)\
                                + self.stage2_exchange_downsample_1to16(features_1x)\
                                + features_16x  
                                
        features_1x = self.activation(updated_features_1x)
        features_4x = self.activation(updated_features_4x)
        features_16x = self.activation(updated_features_16x)
        
        # Stage 3
        features_1x = self.stage3_extraction_1x(features_1x)
        features_4x = self.stage3_extraction_4x(features_4x)
        features_16x = self.stage3_extraction_16x(features_16x)
        
        up4t1 = self.stage3_exchange_upsample_4to1(features_4x) 
        up16t1 = self.stage3_exchange_upsample_16to1(features_16x)
        updated_features_1x = F.interpolate(up4t1,size=(up4t1.shape[2],up4t1.shape[3]*4),mode="nearest")\
                                + F.interpolate(up16t1,size=(up16t1.shape[2],up16t1.shape[3]*16),mode="nearest")\
                                + features_1x
        
        output_mff = self.activation(updated_features_1x) + 0.5*initial_features  # out: [B, C, T, F]
        
        return output_mff