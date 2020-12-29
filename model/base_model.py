import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class BasicClf(nn.Module):
    def __init__(self, stride=1, padding=2, num_class=10, num_layer=4):
        super(BasicClf, self).__init__()
        self.conv = self._make_layers(num_layer, stride, padding)
        self.fc = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_class)
        )

    def _make_layers(self, num_layer, stride, padding):
        layers = []
        size = 32
        for i in range(num_layer-1):
            if i==0:
                block = [
                    nn.Conv2d(1, size, 3, stride, padding), nn.ReLU(),
                    nn.Conv2d(size, size, 3, stride, padding), nn.ReLU(),
                    nn.MaxPool2d(3), nn.Dropout(p=0.25)
                ]
                layers.extend(block)
            else:
                block = [
                    nn.Conv2d(size, size*2, 3, stride, padding), nn.ReLU(),
                    nn.Conv2d(size*2, size*2, 3, stride, padding), nn.ReLU(),
                    nn.MaxPool2d(3), nn.Dropout(p=0.25)
                ]
                size *=2
                layers.extend(block)
        block = [
            nn.Conv2d(size, size*2, 3, stride, padding), nn.ReLU(),
            nn.Conv2d(size*2, size*2, stride, padding), nn.ReLU()
        ]
        layers.extend(block)
        return nn.Sequential(*layers)

    def global_max_pool_2d(self, x):
        return utils.GlobalMaxPool2d(x)
        
        #x = [batch, h, w]
    def forward(self, x):
        #x = [batch, 1, h, w]
        x = x.unsqueeze(1)
        #out = [batch, 256, h, w]
        out = self.conv(x)
        #out = [batch, 256]
        out = self.global_max_pool_2d(out)
        #out = [batch, num_class]
        out = self.fc(out)
        return F.softmax(out, dim=1)

