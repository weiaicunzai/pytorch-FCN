
import torch
import torch.nn as nn
from torch.autograd import Variable

class FCN32s(nn.Module):

    def __init__(self, n_class=21):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2) # 1/2
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2) # 1/4
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2) # 1/8
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1)
            nn.MaxPool2d(2, stride=2) # 1/16
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2, stride=2) #  1/ 32
        )

        #convolutionailzed fc6
        self.conv_6 = nn.Sequential(
            
        )



#deconvolution = nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0, bias=False)
#for p in deconvolution.parameters():
#    p = torch.ones(2, 2)
#    print(p)
#
##convolution = nn.Conv2d(1, 1, 2, stride)
#image = torch.Tensor(4).view(2, 2)
#image[0, 0] = 1
#image[0, 1] = 2
#image[1, 0] = 3
#image[1, 1] = 4
#print(image)
#
#print(deconvolution(Variable(image.view(1, 1, 2, 2))))
#