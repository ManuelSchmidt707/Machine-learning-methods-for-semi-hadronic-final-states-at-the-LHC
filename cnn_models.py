import torch
import torch.nn as nn
import warnings

class Basic_Block(torch.nn.Module):
    """
    Basic Convolutional Block of ResNet.
    Implements a residual block with two convolutional layers, batch normalization, dropout, and ReLU activation.
    Dropout is not used in the thesis.
    """

    def __init__(self, inchannels, outchannels, stride=1, downsample=None, bias=False, dropout=0):
        super(Basic_Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = torch.nn.BatchNorm2d(outchannels)
        self.drop1 = torch.nn.Dropout2d(p=dropout, inplace=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = torch.nn.BatchNorm2d(outchannels)
        self.drop2 = torch.nn.Dropout2d(p=dropout, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        id = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)
        if self.downsample is not None:
            id = self.downsample(x)
        out += id
        out = self.relu(out)
        return out
    
class AdaptedResNet(torch.nn.Module):
    """
    Adapted ResNet Model.
    Customizable ResNet implementation with variable layers in each section, utilizing the Basic_Block class for building blocks.

    Parameters:
    - Block_type: The block type to use (e.g., Basic_Block).
    - layers: List of integers specifying the number of layers in each section.
    - num_classes: Number of output classes (default: 2).
    - inchannels: Number of input channels (Charged, Neutral, Leptonic: 3; CNL+ bottom- and light jet-centers: 5).
    - bias: Whether to use bias in convolutional layers (default: False).
    - k: Scaling factor for the number of channels (default: 2).
    - dropout: Dropout rate (default: 0).
    """

    def __init__(self, Block_type, layers, num_classes=2, inchannels=3, bias=False, k=2, dropout=0):
        super(AdaptedResNet, self).__init__()
        self.k = k
        self.hight = 16 * self.k

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(inchannels, self.hight, kernel_size=7, stride=2, padding=10, bias=bias),
            torch.nn.BatchNorm2d(self.hight),
            torch.nn.ReLU(inplace=True)
        )

        self.layer1 = self.make_layer(Block_type, self.hight, layers[0], bias=bias, dropout=dropout)
        self.layer2 = self.make_layer(Block_type, self.hight * 2, layers[1], stride=2, bias=bias, dropout=dropout)
        self.layer3 = self.make_layer(Block_type, self.hight * 2, layers[2], stride=2, bias=bias, dropout=dropout)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(self.hight, num_classes)

    def make_layer(self, Block_type, channels, num_blocks, stride=1, bias=None, dropout=0):
        downsample = None
        if stride != 1 or self.hight != channels:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.hight, channels, kernel_size=1, stride=stride, bias=bias),
                torch.nn.BatchNorm2d(channels)
            )
        layers = [Block_type(self.hight, channels, stride, downsample, bias=bias, dropout=dropout)]
        self.hight = channels
        for _ in range(1, num_blocks):
            layers.append(Block_type(self.hight, channels, bias=bias, dropout=dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AdaptedResNet_expects_kin(torch.nn.Module):
    """
    Adapted ResNet Model that expects an additional input 'kin'. Used in the case of 5 input channels that calculate input from kin.
    Similar to AdaptedResNet but expects an additional input argument in the forward method.
    """

    def __init__(self, Block_type, layers, num_classes=2, inchannels=3, bias=False, k=2, dropout=0):
        super(AdaptedResNet_expects_kin, self).__init__()
        self.k = k
        self.hight = 16 * self.k

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(inchannels, self.hight, kernel_size=7, stride=2, padding=10, bias=bias),
            torch.nn.BatchNorm2d(self.hight),
            torch.nn.ReLU(inplace=True)
        )

        self.layer1 = self.make_layer(Block_type, self.hight, layers[0], bias=bias, dropout=dropout)
        self.layer2 = self.make_layer(Block_type, self.hight * 2, layers[1], stride=2, bias=bias, dropout=dropout)
        self.layer3 = self.make_layer(Block_type, self.hight * 2, layers[2], stride=2, bias=bias, dropout=dropout)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(self.hight, num_classes)

    def make_layer(self, Block_type, channels, num_blocks, stride=1, bias=None, dropout=0):
        downsample = None
        if stride != 1 or self.hight != channels:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.hight, channels, kernel_size=1, stride=stride, bias=bias),
                torch.nn.BatchNorm2d(channels)
            )
        layers = [Block_type(self.hight, channels, stride, downsample, bias=bias, dropout=dropout)]
        self.hight = channels
        for _ in range(1, num_blocks):
            layers.append(Block_type(self.hight, channels, bias=bias, dropout=dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class CNN_CNL(nn.Module):
    """
    CNL Network from "Uncovering doubly charged scalars with dominant three-body decays using machine learning"
    """

    def __init__(self, inchannels=3):
        super(CNN_CNL, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(inchannels, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(4),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(5),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(5),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.AvgPool2d(5),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(5),
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * 32 * 2, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 2),
        )

    def forward(self, x):
        x = self.layer1(x.unsqueeze(1)).squeeze()

class ResNeXt_Block(torch.nn.Module):
    '''
    The Convolution Block used by the ResNeXt model.
    '''
    def __init__(self, inchannels, outchannels, stride = 1, downsample = None, bias = False, dropout = 0, cardinality = 16):

        super(ResNeXt_Block, self).__init__()
        
        self.bn1 = torch.nn.BatchNorm2d(inchannels)
        self.bn2 = torch.nn.BatchNorm2d(inchannels)
        self.bn3 = torch.nn.BatchNorm2d(outchannels)    
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv1x1_down = torch.nn.Conv2d(inchannels, inchannels, kernel_size=1, stride=1, bias=bias)
        self.conv3x3 = torch.nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=bias)
        self.conv1x1_up = torch.nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, bias=bias)

        self.downsample = downsample
        self.stride = stride
        self.cardinality = cardinality

    def forward(self, x):
        id = x 

        out = self.conv1x1_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3x3(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1x1_up(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            id = self.downsample(x)

        out += id
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):
    '''
    The Implementation of the ResNeXt model. 
    '''
    def __init__(self, Block_type, layers, num_classes = 2, inchannels = 3, bias = False, k = 1, dropout = 0, cardinality = 16):
        super(ResNeXt, self).__init__()
    
        if k * 64 < cardinality:
            warnings.warn("The number of groups in the ResNeXt model (cardinality) is bigger than the number of channels. This is not allowed. The number of groups will be reduced.")
            cardinality = int(k * 64/2)
            warnings.warn("The number of groups (cardinality) is now set to: " + str(cardinality))

        self.k = k
        self.hight = 64 * self.k
        self.cardinality = cardinality

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(inchannels, self.hight, kernel_size=7,stride=2,padding=10,bias=bias),
            torch.nn.BatchNorm2d(self.hight),
            torch.nn.ReLU(inplace=True)
        )

        self.layer1 = self.make_layer(Block_type, self.hight, layers[0], bias = bias, dropout=dropout, cardinality = self.cardinality)
        self.layer2 = self.make_layer(Block_type, self.hight * 2, layers[1], stride = 2, bias = bias, dropout=dropout, cardinality = self.cardinality)
        self.layer3 = self.make_layer(Block_type, self.hight * 2, layers[2], stride = 2, bias = bias, dropout=dropout, cardinality = self.cardinality)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.classifier = torch.nn.Linear(self.hight, num_classes)

    def make_layer(self, Block_type, channels, num_blocks, stride = 1, bias = None, dropout = 0, cardinality = 32):

        downsample = None

        if stride != 1 or self.hight != channels:
            downsample = torch.nn.Sequential( 
                torch.nn.Conv2d(self.hight, channels, kernel_size=1, stride=stride, bias=bias),
                torch.nn.BatchNorm2d(channels)
            )

        layer = []

        for i in range(1, num_blocks):
            layer.append(Block_type(self.hight, self.hight, bias = bias, dropout = dropout, cardinality = cardinality))

        layer.append(Block_type(self.hight, channels, stride, downsample, bias = bias, dropout = dropout, cardinality = cardinality))

        self.hight = channels

        return torch.nn.Sequential(*layer)

    def forward(self, x):
        if self.input == '3D':
            x = self.input_layer(x.unsqueeze(1)).squeeze()
        else:
            x = self.input_layer(x)

        x = self.layer1(x)          
        x = self.layer2(x)          
        x = self.layer3(x)                  

        x = self.avgpool(x)         
        x = torch.flatten(x, 1)     
        x = self.classifier(x)

        return x
