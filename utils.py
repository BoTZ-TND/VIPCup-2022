import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist 

class dist_average:
    def __init__(self,device):
        self.device=device
        self.acc=torch.zeros(1).to(device)
        self.count=0
    def step(self,input_):
        self.count+=1
        if type(input_)!=torch.Tensor:
            input_=torch.tensor(input_).to(self.device,dtype=torch.float)
        else:
            input_=input_.detach()
        self.acc+=input_
            
    def get(self):
        return self.acc.item()/self.count

def ACC(x,y):
    with torch.no_grad():
        a=torch.max(x,dim=1)[1]
        acc= torch.sum(a==y).float()/x.shape[0]
    #print(y,a,acc)        
    return acc

def cont_grad(x,rate=1):
    return rate*x+(1-rate)*x.detach()