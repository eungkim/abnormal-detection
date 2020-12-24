import torch
import numpy as np
import pandas as pd


def GlobalMaxPool2d(x):
    output, _ = torch.max(torch.max(x, -1)[0], -1)
    return output
