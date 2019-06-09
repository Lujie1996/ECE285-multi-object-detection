import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as td 
import torchvision as tv
import nntools as nt
import loss

class NNClassifier(nt.NeuralNetwork):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.loss = loss.yoloLoss(7, 2, 5, 0.5)
    
    def criterion(self, y, d):
        return self.loss(y, d)

class YOLO(NNClassifier):
    def __init__(self, num_classes, fine_tuning=False):
        super(YOLO, self).__init__()
        C = 20  
        vgg = tv.models.vgg16_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
        self.features = vgg.features
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.Dropout(p = 0.5),
            nn.LeakyReLU(0.1)
        )
        self.fc2 = nn.Sequential(nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + C)))
        
    def forward(self, x):
        # print('{}: forward() started.'.format(datetime.datetime.now()))
        f = self.features(x)
        
        y = self.conv1(f)
        y = self.conv2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = y.view(-1,7,7,30)
        # print('{}: forward() finished.'.format(datetime.datetime.now()))
        return y


class ResYOLO(NNClassifier):
    def __init__(self, num_classes, fine_tuning=False):
        super(ResYOLO, self).__init__()
        C = num_classes  
        resnet = tv.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = fine_tuning
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.outconv1 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.outconv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.Dropout(p = 0.5),
            nn.LeakyReLU(0.1)
        )
        self.fc2 = nn.Sequential(nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + C)))
        
    def forward(self, x):
        # print('{}: forward() started.'.format(datetime.datetime.now()))
        f = self.conv1(x)
        y = self.bn1(f)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.outconv1(y)
        y = self.outconv2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        # print('{}: forward() finished.'.format(datetime.datetime.now()))
        y = y.view(-1,7,7,30)
        return y