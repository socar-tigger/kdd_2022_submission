import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# lightening module에서 사용 위해 PMG 모듈에서 변경함 
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class PMG(nn.Module):
    def __init__(self, model, feature_size, num_classes):
        super(PMG, self).__init__()

        self.features = model  
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)
        
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )

    def _forward(self, x):

        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)

        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)
        
        return xc1, xc2, xc3, x_concat

    def forward(self, x1, x2, x3, org):
        out1, _, _, _ = self._forward(x1)
        _, out2, _, _ = self._forward(x2)
        _, _, out3, _ = self._forward(x3)
        _, _, _, out_org = self._forward(org)
        
        return [out1, out2, out3, out_org]
    
class PMG_Multi(nn.Module):
    def __init__(self, model, feature_size, device):
        super(PMG_Multi, self).__init__()

        self.features = model  
        self.max1 = nn.MaxPool2d(kernel_size=56, stride=56)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)
        
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.heads1 = [nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 1).to(device), 
                            nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 7).to(device)]

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.heads2 = [nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 1).to(device), 
                            nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 7).to(device)]

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2,
                      kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.heads3 = [nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 1).to(device), 
                            nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 7).to(device)]

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.heads_concat = [nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 1).to(device), 
                            nn.Linear(feature_size, 1).to(device), nn.Linear(feature_size, 7).to(device)]
    def _forward(self, x):

        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl1 = self.conv_block1(xf3)
        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xl1 = self.max1(xl1)
        xl1 = xl1.view(xl1.size(0), -1)
        xcl1 = self.classifier1(xl1)
        xc1s = [h(xcl1) for h in self.heads1]

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xcl2 = self.classifier2(xl2)
        xc2s = [h(xcl2) for h in self.heads2]

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xcl3 = self.classifier3(xl3)
        xc3s = [h(xcl3) for h in self.heads3]

        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)
        x_concats = [h(x_concat) for h in self.heads_concat]
        return xc1s, xc2s, xc3s, x_concats

    def forward(self, x1, x2, x3, org):
        out1, _, _, _ = self._forward(x1)
        _, out2, _, _ = self._forward(x2)
        _, _, out3, _ = self._forward(x3)
        _, _, _, out_org = self._forward(org)
        
        return [out1, out2, out3, out_org]