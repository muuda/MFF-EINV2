import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.mff_module import mff_module
from methods.utils.model_utilities import (DoubleConv, init_layer)
from methods.utils.conformer.encoder import ConformerBlocks


class MFF_EINV2(nn.Module):
    """
        Implementation of MFF-EINV2
    """
    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.num_classes = dataset.num_classes
        self.subnetwork = self.cfg['training']['subnetwork']
        self.tfcm_layer = self.cfg['training']['tfcm_layer']
        
        # Initial Stage (First Dual Conv)
        self.initial_stage = nn.Sequential(
            DoubleConv(in_channels=7, out_channels=64),
            nn.Dropout(p=0.05),
        )
        
        # MFF Module
        self.mff_module = mff_module(self.subnetwork, self.tfcm_layer)

        # SED and DOA Dual Conv Blocks
        self.sed_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.Dropout(p=0.05),
            nn.AvgPool2d(kernel_size=(2, 2))
        )
        self.doa_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.Dropout(p=0.05),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        
        self.sed_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.Dropout(p=0.05),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.Dropout(p=0.05),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        
        self.sed_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=256),
            nn.Dropout(p=0.05),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=256),
            nn.Dropout(p=0.05),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        
        # Stitch Block
        self.stitch = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9))
        ])
        
        # Conformer Blocks
        self.sed_conformer_track1 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.sed_conformer_track2 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.sed_conformer_track3 = ConformerBlocks(encoder_dim=256, num_layers=2)

        self.doa_conformer_track1 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.doa_conformer_track2 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.doa_conformer_track3 = ConformerBlocks(encoder_dim=256, num_layers=2)


        # FC Layers
        self.fc_sed_track1 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_sed_track2 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_sed_track3 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_doa_track1 = nn.Linear(256, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(256, 3, bias=True)
        self.fc_doa_track3 = nn.Linear(256, 3, bias=True)
        self.final_act_sed = nn.Sequential()
        self.final_act_doa = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.fc_sed_track1)
        init_layer(self.fc_sed_track2)
        init_layer(self.fc_sed_track3)
        init_layer(self.fc_doa_track1)
        init_layer(self.fc_doa_track2)
        init_layer(self.fc_doa_track3)

    def forward(self, x: torch.Tensor):
        """
            x: spectrogram, (B, C, T, F)
        """
        # Initial Stage
        x = self.initial_stage(x) # [B, 7, T, F]  --> [b, 64, T, F]

        # MFF Module
        x = self.mff_module(x)
        
        # SED and DOA Dual Conv Blocks
        x_sed = self.sed_conv_block1(x)
        x_doa = self.doa_conv_block1(x)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 1], x_doa)
        
        x_sed = self.sed_conv_block2(x_sed)
        x_doa = self.doa_conv_block2(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 1], x_doa)
        x_sed = self.sed_conv_block3(x_sed)
        x_doa = self.doa_conv_block3(x_doa)

        x_sed = x_sed.mean(dim=3) # (B, D, T)
        x_doa = x_doa.mean(dim=3) # (B, D, T)

        # Conformer Blocks
        x_sed = x_sed.permute(0, 2, 1) # (B, T, D)
        x_doa = x_doa.permute(0, 2, 1) # (B, T, D)
  
        x_sed_1 = self.sed_conformer_track1(x_sed)
        x_doa_1 = self.doa_conformer_track1(x_doa)
        x_sed_1 = torch.einsum('c, ntc -> ntc', self.stitch[2][:, 0, 0], x_sed_1) + \
            torch.einsum('c, ntc -> ntc', self.stitch[2][:, 0, 1], x_doa_1)
        x_doa_1 = torch.einsum('c, ntc -> ntc', self.stitch[2][:, 1, 0], x_sed_1) + \
            torch.einsum('c, ntc -> ntc', self.stitch[2][:, 1, 1], x_doa_1)
        
        x_sed_2 = self.sed_conformer_track2(x_sed) 
        x_doa_2 = self.doa_conformer_track2(x_doa)
        x_sed_2 = torch.einsum('c, ntc -> ntc', self.stitch[3][:, 0, 0], x_sed_2) + \
            torch.einsum('c, ntc -> ntc', self.stitch[3][:, 0, 1], x_doa_2)
        x_doa_2 = torch.einsum('c, ntc -> ntc', self.stitch[3][:, 1, 0], x_sed_2) + \
            torch.einsum('c, ntc -> ntc', self.stitch[3][:, 1, 1], x_doa_2)

        x_sed_3 = self.sed_conformer_track3(x_sed)
        x_doa_3 = self.doa_conformer_track3(x_doa)
        x_sed_3 = torch.einsum('c, ntc -> ntc', self.stitch[4][:, 0, 0], x_sed_3) + \
            torch.einsum('c, ntc -> ntc', self.stitch[4][:, 0, 1], x_doa_3)
        x_doa_3 = torch.einsum('c, ntc -> ntc', self.stitch[4][:, 1, 0], x_sed_3) + \
            torch.einsum('c, ntc -> ntc', self.stitch[4][:, 1, 1], x_doa_3)

        # FC Layers
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_sed_2))
        x_sed_3 = self.final_act_sed(self.fc_sed_track3(x_sed_3))
        x_sed = torch.stack((x_sed_1, x_sed_2, x_sed_3), 2)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_doa_2))
        x_doa_3 = self.final_act_doa(self.fc_doa_track3(x_doa_3))
        x_doa = torch.stack((x_doa_1, x_doa_2, x_doa_3), 2)
        output = {
            'sed': x_sed,
            'doa': x_doa,
        }

        return output