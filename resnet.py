import torch.nn as nn
#add imports as necessary

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #ResNet layers
        self.layer1 = self.new_block(64, 64, stride=1)
        self.layer2 = self.new_block(64, 128, stride=2)
        self.layer3 = self.new_block(128, 256, stride=2)
        self.layer4 = self.new_block(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        #TODO: implement the forward function for resnet,
        #use all the functions you've made
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)

        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def new_block(self, in_channels, out_channels, stride):
        #TODO: make a convolution with the above params
        #layers = []


        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=0),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU()]


        #layers.append(block(in_channels, out_channels, stride))

        return nn.Sequential(*layers)

"""
class block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x.clone()
        if self.in_channels != self.out_channels:
            identity = self.identity(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x += identity #check dimension
        x = self.relu(x)

        return x
"""
