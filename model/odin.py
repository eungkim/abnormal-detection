import torch
import torch.nn as nn
import torch.nn.functional as F


#Temperature Scaling part
class TempScaling(nn.Module):
    def __init__(self, model, temp):
        super(TempScaling, self).__init__()
        self.model = model
        self.temp = temp

    def forward(self, data):
        #x = [batch, num_classes]
        x = self.model(data)
        #numer = [batch, num_classe]
        numer = torch.exp(x/self.temp)
        #denom = [batch, 1]
        denom = torch.sum(numer, dim=1).view(-1, 1)
        #out = [batch, num_classes]
        out = torch.div(numer, denom)
        return out


#Input Preprocessing part
class InputPreprocess(nn.Module):
    def __init__(self, model, eps):
        super(InputPreprocess, self).__init__()
        self.eps = eps
        self.model = model

    def forward(self, x):
        x.requires_grad = True
        #out = [batch, num_class]
        out = self.model(x)
        #out = [batch]
        out, _ = torch.max(out, dim=1)
        out = torch.sum(out)
        out.backward(retain_graph=True)
        grad = -x.grad

        perturbation = torch.multiply(torch.sign(grad), eps)
        x_tilde = x - perturbation
        return x_tilde 


#Out=of-distribution Detector part
class OutDetector(nn.Module):
    def __init__(self, sigma):
        super(OutDetector, self).__init__()
        self.sigma = sigma

        #x = [batch, num_class]
    def forward(self, x):
        #max_softmax = [batch]
        max_softmax = torch.max(x, dim=1)
        #out = [batch]
        out = max_softmax > self.sigma 
        return out #Boolean Tensor, True in-dist, False out-dist 


class ODIN(nn.Module):
    """
    temp_scaler = TempScaling(base_model, temp)
    input_preprocss = InputPreprocess(TempScaling(base_model, temp), eps)
    ood_detector = OutDetector(sigma)
    """
    def __init__(self, base_model, temp, eps, sigma):
        super(ODIN, self).__init__()

        self.temp_scaler = TempScaling(base_model, temp)
        self.input_preprocess = InputPreprocess(TempScaling(base_model, temp), eps)
        self.ood_detector = OutDetector(sigma)

    def forward(self, x):
        x_tilda = self.input_preprocess(x) 
        out = self.temp_scaler(x_tilda)
        out = self.ood_detector(out)
        return out

