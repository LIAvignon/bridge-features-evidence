import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

__paper__           = "A bridge between features and evidence for attribute-driven perfect privacy"
__code-author__     = "Paul-Gauthier Noé"
__paper-authors__   = ["Paul-Gauthier Noé", "Andreas Nautsch", "Driss Matrouf", "Pierre-Michel Bousquet", "Jean-François Bonastre"]
__license__         = "MIT license"


class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, data_file):
        self.labels     = labels                                        # Labels
        self.list_IDs   = list_IDs                                      # Sample's ids
        self.data       = torch.Tensor(np.loadtxt(data_file))           # Data
        print(self.data.size())
        print("Data Loaded !")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self,index):
        ID  = self.list_IDs[index]                                      # Sample's ids
        x   = self.data[index]                                          # Data
        y   = self.labels[ID]                                           # Labels
        return x, y, ID


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super(CouplingLayer, self).__init__()
        
        self.input_dim      = input_dim
        self.hidden_dim     = hidden_dim
        self.s              = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, input_dim), nn.Tanh())
        self.t              = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, input_dim))
        self.mask           = nn.Parameter(mask, requires_grad=False)

    def f(self, x):             # from x to y
        x1          = x*self.mask
        s1          = self.s(x1)
        t1          = self.t(x1)
        log_detJ    = torch.sum(s1*(1-self.mask),dim=1)
        return x1+(1-self.mask)*(x*torch.exp(s1)+t1), log_detJ

    def g(self, y):              # from y to x
        y1 = y*self.mask
        s1 = self.s(y1)
        t1 = self.t(y1)
        return y1+(1-self.mask)*(y-t1)*torch.exp(-s1)



class RealNVP(nn.Module):

    def __init__(self,input_dim, hidden_dim, masks):
        super(RealNVP, self).__init__()

        self.input_dim  = input_dim
        self.masks      = masks
        self.cpl        = nn.ModuleList([CouplingLayer(input_dim, hidden_dim, masks[i]) for i in range(len(masks))])
        print(len(self.masks))

    def log_prior(self, z, l, priors):
        i0  = l==0
        i1  = l==1
        i0  = i0.reshape((len(i0),))
        i1  = i1.reshape((len(i1),))
        lp0 = priors[0].log_prob(z[i0])
        lp1 = priors[1].log_prob(z[i1])
        return torch.cat((lp0,lp1))


    def f(self, x):      # from x to z
        log_detJ = 0
        z = x
        for i in (range(len(self.masks))):
            z, log_detJi = self.cpl[i].f(z)
            log_detJ += log_detJi
        return z, log_detJ

    def g(self, z):      # from z to x
        for i in reversed(range(len(self.masks))):
            z = self.cpl[i].g(z)
        x = z
        return x 
