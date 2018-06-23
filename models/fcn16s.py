
import torch
import torch.nn as nn
from torch.autograd import Variable


class FCN16s(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()

        #conv1
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/2

        #conv2
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)            
        ) # 1/4

        #conv3
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/8
        
        #conv4
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/16

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        ) # 1/32

        #fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        #fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_pool5 = nn.Sequential(
            nn.Conv2d(4096, n_class, 1)
        )
        self.score_pool4 = nn.Sequential(
            nn.Conv2d(512, n_class, 1)
        )

        
        self.upscore2x = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        )
        self.upscore16x = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 32, stride=16, bias=False)
        )


    def foward(self, x):
        output = self.conv_1(x)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)

        # """We add a 1 × 1 convolution layer on top of 
        # pool4 to produce additional class predictions.
        # """ 
        pool4_output = self.score_pool4(output)

        output = self.conv_5(output)
        output = self.fc6(output)
        output = self.fc7(output)

        # """We fuse this output (pool4_output) with the predictions 
        # computed on top of conv7 (convolutionalized fc7) at stride 
        # 32 by adding a 2× upsampling layer and summing both predic-
        # tions."""

        pool5_output = self.score_pool5(output)
        pool5_output = self.upscore2x(pool5_output)
        fused_output = pool5_output + pool4_output[:, :, 5 : 5 + pool5_output.size()[2] + 5 : 5 + pool5_output.size()[3]]

        # """Finally, the stride 16 predictions
        # are upsampled back to the image."""
        output = self.self.upscore16x(fused_output)
        output = output[:, :, 27 : 27 + x.size()[2], 27 : 27 + x.size()[3]] #crop the output to be the same size as input
        return output










