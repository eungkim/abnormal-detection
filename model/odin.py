import torch
import torch.nn as nn
import torch.nn.functional as F


class TempScaling(nn.Module):
    def __init__(self, temp):
        super(TempScaling, self).__init__()
        self.temp = temp

        #(batch, num_classes)
    def forward(self, x):
        #(batch, num_classe)
        numer = torch.exp(x/self.temp)
        #(batch, 1)
        denom = torch.sum(numer, dim=1).view(-1, 1)
        #(batch, num_classes)
        out = torch.div(numer, denom)
        return out


def input_preprocess(model, x, eps): #input_pre(model=BaseModel(), x=data)
    x.requires_grad = True
    #(batch, num_class)
    out = model(x)
    #(batch)
    out, _ = torch.max(out, dim=1)
    out = torch.sum(out)
    out.backward(retain_graph=True)
    grad = -x.grad

    perturbation = torch.multiply(torch.sign(grad), eps)
    x_tilde = x - perturbation
    return x_tilde


class OutDetector(nn.Module):
    def __init__(self, eps):
        super(OutDetector, self).__init__()
        self.eps = eps

        #(batch, num_class)
    def forward(self, x):
        #(batch)
        max_softmax = torch.max(x, dim=1)
        #(batch)
        out = max_softmax > self.eps 
        return out #(batch), Boolean Tensor, True in-dist, False out-dist 
