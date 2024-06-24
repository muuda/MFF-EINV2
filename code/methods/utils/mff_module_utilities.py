import torch.nn as nn

# MIT License

# Copyright (c) 2022 Shimin Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
    
class FD(nn.Module):
    def __init__(self, cin, cout, K=(1, 7), S=(1, 4), P=(0, 2)):
        super(FD, self).__init__()
        self.fd = nn.Sequential(
            nn.Conv2d(cin, cout, K, S, P, groups=2),
            nn.BatchNorm2d(cout),
            nn.ReLU(cout)
        )

    def forward(self, x):
        """
            inp: B x C x T x F
        """
        return self.fd(x)
    
class TFCM_Block(nn.Module):
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 dila=1,
                 causal=True,
                 ):
        super(TFCM_Block, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=(1, 1)),
            nn.BatchNorm2d(cin),
            nn.ReLU(cin),
        )
        dila_pad = dila * (K[1] - 1)
        if causal:
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, dila_pad, 0), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(dila, 1), groups=cin),
                nn.BatchNorm2d(cin),
                nn.ReLU(cin)
            )
        else:
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad//2, dila_pad//2, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                nn.BatchNorm2d(cin),
                nn.PReLU(cin)
            )

        self.pconv2 = nn.Conv2d(cin, cin, kernel_size=(1, 1))
        self.causal = causal
        self.dila_pad = dila_pad

    def forward(self, inps):
        """
            inp: B x C x T x F
        """
        outs = self.pconv1(inps)
        outs = self.dila_conv(outs)
        outs = self.pconv2(outs)
        return outs + inps


class TFCM(nn.Module):
    """
        Unofficial PyTorch implementation of Baidu's MTFAA-Net, which is available at:
        https://github.com/echocatzh/MTFAA-Net/blob/main/tfcm.py
    """
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 tfcm_layer=6,
                 causal=True,
                 ):
        super(TFCM, self).__init__()
        self.tfcm = nn.ModuleList()
        for idx in range(tfcm_layer):
            self.tfcm.append(
                TFCM_Block(cin, K, 2**idx, causal=causal)
            )

    def forward(self, inp):
        out = inp
        for idx in range(len(self.tfcm)):
            out = self.tfcm[idx](out)
        return out