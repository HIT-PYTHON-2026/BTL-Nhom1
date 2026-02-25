import torch
import torch.nn as nn
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size= 3,stride=stride,padding= 1, bias= False)
class Block(nn.Module):
    expansion = 1
    def __init__(self, inchannels,outchannels,stride=1,downsample = None):
        super(Block,self).__init__()
        self.conv1 = conv3x3(inchannels,outchannels,stride)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outchannels,outchannels)
        self.bn2 = nn.BatchNorm2d(outchannels)
         
        self.downsample = downsample
        self.stride =stride

    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self,block,layer,num_classes,parameter_dropout):
        super(ResNet,self).__init__()
        self.inchannels = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=(3,3),bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        
        self.layer1 = self.layer(block, 64,layer[0])
        self.layer2 = self.layer(block, 128,layer[1],stride = 2)
        self.layer3 = self.layer(block, 256,layer[2],stride = 2)
        self.layer4 = self.layer(block, 512,layer[3],stride = 2)
        
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(p=parameter_dropout), 
            nn.Linear(512 * block.expansion, num_classes)
        )
    def layer(self,block,channels,blocks,stride = 1):
        downsample = None
        if stride != 1 or self.inchannels != channels*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inchannels,channels*block.expansion, kernel_size= 1,stride = stride, bias = False)
                                       ,nn.BatchNorm2d(channels*block.expansion))
        layers = []
        layers.append(block(self.inchannels,channels,stride,downsample))
        self.inchannels = channels*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inchannels,channels))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpooling(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
def Resnet18(num_classes,parameter_dropout):
    model = ResNet(Block,[2,2,2,2],num_classes,parameter_dropout)
    return model
