from time import time
import torch
from torch import nn


class NODEfunc(nn.Module):

    def __init__(self, dim, nhidden, augment_dim=0, time_dependent=True):
        super(NODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.time_dependent = time_dependent
        dim = dim + augment_dim
        if self.time_dependent:
            self.fc1 = nn.Linear(dim + 1, nhidden)
        else:
            self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(x.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class SONODEfunc(nn.Module):
    def __init__(self, dim, nhidden, time_dependent=True, modelname=None, actv=nn.Tanh()):
        super(SONODEfunc, self).__init__()
        self.modelname = modelname
        indim = 2 * dim if self.modelname == 'SONODE' else dim
        self.time_dependent = time_dependent
        if self.time_dependent:
            indim += 1
        # only have residual for generalized model
        self.res = 2.0 if self.modelname == "GHBNODE" else 0.0
        self.elu = nn.ELU(inplace=False)
        self.fc1 = nn.Linear(indim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.gamma = nn.Parameter(torch.Tensor([-3.0]))
        self.actv = actv
        self.nfe = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, t, x):
        cutoff = int(len(x)/2)
        z = x[:cutoff]
        v = x[cutoff:]
        if self.modelname == 'SONODE':
            z = torch.cat((z, v), dim=1)
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(z.shape[0], 1).to(x.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_z = torch.cat([t_vec, z], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_z)
        else:
            out = self.fc1(z)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        if self.modelname == 'SONODE':
            out = self.fc3(out)
            return torch.cat((v, out))
        else:
            out = self.fc3(out) - self.sigmoid(self.gamma) * v - self.res * z
            if self.modelname == "GHBNODE":
                actv_v = self.actv
            else:
                actv_v = nn.Identity()
            return torch.cat((actv_v(v), out))


class NesterovNODEfunc(nn.Module):
    def __init__(self, dim, nhidden, time_dependent=True, modelname=None, xi=None, actv=nn.Tanh()):
        super(NesterovNODEfunc, self).__init__()
        self.modelname = modelname
        self.time_dependent = time_dependent
        # only have residual for generalized model
        self.res = 0.0 if xi is None else xi
        self.elu = nn.ELU(inplace=False)
        if self.time_dependent:
            self.fc1 = nn.Linear(dim + 1, nhidden)
        else:
            self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.actv = actv
        self.nfe = 0
        self.verbose = False

    def forward(self, t, z):
        if self.verbose:
            print("Inside ODE function")
            print("z:", z)
            print("t:", t)
        cutoff = int(len(z)/2)
        h = z[:cutoff]
        dh = z[cutoff:]
        k_reciprocal = torch.pow(t, 3/2) * torch.exp(-t/2)
        m = (3/2 * torch.pow(t, 1/2) * torch.exp(-t/2) - 1/2 * k_reciprocal) * h \
            + k_reciprocal * dh
        x = h * k_reciprocal
        if z.is_cuda:
            k_reciprocal = k_reciprocal.to(z.get_device())
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(x.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_z = torch.cat([t_vec, h], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_z)
        else:
            out = self.fc1(h)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        actv_df = self.actv if self.modelname == "GNesterovNODE" else nn.Identity()
        out = actv_df(self.fc3(out))
        dm = - m - out - self.res * h
        if self.verbose:
            print("out:", out)
            print("m:", m)
            print("x:", x)
            print("dm:", dm)
        if self.modelname in ("GNesterovNODE"):
            # actv_v = nn.Tanh()
            actv_v = self.actv
        else:
            actv_v = nn.Identity()
        return torch.cat((actv_v(m), dm))
