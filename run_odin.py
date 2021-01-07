import torch
from model.base_model import BasicClf
import model.odin as odin 
import argparse


parser = argparse.ArgumentParser(description="type the arguments")
parser.add_argument('--temp', type=float ,help="temp")
parser.add_argument('--eps', type=float, help="eps")
parser.add_argument('--sigma', type=float, help="sigma")
args = parser.parse_args()

temp = args.temp 
eps = args.eps
sigma = args.sigma

# Suppose that we have a well trained classifier named "base_model"
base_model = BasicClf()

# Suppose that we have the test_dataset named "test_data"
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=8,
    shuffle=False
)

model = odin.ODIN(base_model, temp, eps, sigma)

y_hat_total = torch.tensor([])
y_total = torch.tensor([])
for data in test_loader:
    x, y = data
    #out is boolen tensor, which has True value when it is ood sample
    out = model(x)
    y_hat_total = torch.cat((y_hat_total, out), 0) 
    y_total = torch.cat((y_total, y), 0)

#evaluate using AUROC, AUPR, Detection Error 


