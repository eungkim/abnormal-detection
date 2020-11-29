import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicClf(nn.Module):
    def __init__(self, stride=1, padding=1, num_class=10):
        super(BasicClf, self).__init__()
        conv1 = nn.Conv2d(1, 32, 3, stride, padding)
        conv2 = nn.COnv2d(32, 32, 3, stride, padding)
        pool1 = nn.MaxPool2d(3)
        dropout1 = nn.Dropout(p=0.25)

        conv3 = nn.Conv2d(32, 64, 3, stride, padding)
        conv4 = nn.Conv2d(64, 64, 3, stride, padding)
        pool2 = nn.MaxPool2d(3)
        dropout2 = nn.Dropout(p=0.25)

        conv5 = nn.Conv2d(64, 128, 3, stride, padding)
        conv6 = nn.Conv2d(128, 128, 3, stride, padding)
        pool3 = nn.MaxPool2d(3)
        dropout3 = nn.Dropout(p=0.25)

        conv7 = nn.Conv2d(128, 256, 3, stride, padding)
        conv8 = nn.Conv2d(256, 256, 3, stride, padding)

        self.conv = nn.Sequential(
            conv1, 
            nn.ReLU(),
            conv2, 
            nn.ReLU(),
            pool1, 
            dropout1,

            conv3, 
            nn.ReLU(),
            conv4, 
            nn.ReLU(),
            pool2, 
            dropout2,

            conv5, 
            nn.ReLU(),
            conv6, 
            nn.ReLU(),
            pool3, 
            dropout3,
            conv6,
            nn.ReLU(),
            conv8,
            nn.ReLU())

        fc1 = nn.Linear(256, 1024)
        dropout4 = nn.Dropout(p=0.5)
        fc2 = nn.Linear(1024, num_class)

        self.fc = nn.Sequential(
            fc1,
            dropout4,
            fc2)
    
    def forward(self, x):
        out = self.conv(x)
        out, _ = torch.max(torch.max(out, 1)[0], 1)  #Global Max Pooling
        out = self.fc(out)
        return F.softmax(out, dim=1)

