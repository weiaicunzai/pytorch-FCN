
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
            nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/2
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/4
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/8
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/16
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True) #  1/ 32
        )

        #convolutionailzed fc6
        self.conv_6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),            
            nn.Dropout2d()
        )

        #convolutionized fc7
        self.conv_7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score = nn.Sequential(
            nn.Conv2d(4096, n_class, 1)
        )

        #transpose convolution
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)


    def forward(self, x):
        output = self.conv_1(x)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)
        output = self.conv_7(output)
        output = self.score(output)
        output = self.upscore(output)

        output = output[:, :, 19 : 19 + x.size()[2], 19:19 + x.size()[3]]
        return output



#image = torch.Tensor(3, 300, 500)
#print(image.size())
#
#fcn32s= FCN32s()
#output = fcn32s(Variable(image.view(1, 3, 300, 500)))
#print(output.size())