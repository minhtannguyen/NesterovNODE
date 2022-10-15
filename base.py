from crypt import methods
import torch
from einops import rearrange
from torch import nn
from torchdiffeq import odeint_adjoint

from basehelper import *


def bmul(vec, mat, axis=0):
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


class Tinvariant_NLayerNN(NLayerNN):
    def forward(self, t, x):
        return super(Tinvariant_NLayerNN, self).forward(x)


class dfwrapper(nn.Module):
    def __init__(self, df, shape, recf=None):
        super(dfwrapper, self).__init__()
        self.df = df
        self.shape = shape
        self.recf = recf

    def forward(self, t, x):
        bsize = x.shape[0]
        if self.recf:
            x = x[:, :-self.recf.osize].reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dr = self.recf(t, x, dx).reshape(bsize, -1)
            dx = dx.reshape(bsize, -1)
            dx = torch.cat([dx, dr], dim=1)
        else:
            x = x.reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dx = dx.reshape(bsize, -1)
        return dx


class NODEintegrate(nn.Module):

    def __init__(self, df, shape=None, tol=1e-5, adjoint=True, evaluation_times=None, recf=None, nesterov_algebraic=None, activation_h=None, activation_output=None, time_requires_grad=True, verbose=True):
        """
        Create an OdeRnnBase model
            x' = df(x)
            x(t0) = x0
        :param df: a function that computes derivative. input & output shape [batch, channel, feature]
        :param x0: initial condition.
            - if x0 is set to be nn.parameter then it can be trained.
            - if x0 is set to be nn.Module then it can be computed through some network.
        """
        super().__init__()
        self.df = dfwrapper(df, shape, recf) if shape else df
        self.tol = tol
        if verbose:
            print(f"Tolerance: {self.tol}")
            print(f"Adjoint:", adjoint)
        self.odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        self.evaluation_times = evaluation_times if evaluation_times is not None else torch.Tensor([0.0, 1.0])
        if time_requires_grad == False:
            self.evaluation_times.requires_grad = False
        if verbose:
            print("Evaluation times:", self.evaluation_times)
            print("Learnable eval times:", self.evaluation_times.requires_grad)
        self.shape = shape
        self.recf = recf
        self.nesterov_algebraic = nesterov_algebraic
        self.activation_h = nn.Identity() if activation_h is None else activation_h
        self.activation_output = nn.Identity() if activation_output is None else activation_output
        if verbose:
            print("self.nesterov_algebraic:", self.nesterov_algebraic)
            print("self.activation_h for NODEintegrate:", self.activation_h)
        if recf:
            assert shape is not None

    def forward(self, x0):
        """
        Evaluate odefunc at given evaluation time
        :param x0: shape [batch, channel, feature]. Set to None while training.
        :param evaluation_times: time stamps where method evaluates, shape [time]
        :param x0stats: statistics to compute x0 when self.x0 is a nn.Module, shape required by self.x0
        :return: prediction by ode at evaluation_times, shape [time, batch, channel, feature]
        """
        bsize = x0.shape[0]
        if self.shape:
            assert x0.shape[1:] == torch.Size(self.shape), \
                'Input shape {} does not match with model shape {}'.format(x0.shape[1:], self.shape)
            x0 = x0.reshape(bsize, -1)
            if self.recf:
                reczeros = torch.zeros_like(x0[:, :1])
                reczeros = repeat(reczeros, 'b 1 -> b c', c=self.recf.osize)
                x0 = torch.cat([x0, reczeros], dim=1)
            out = odeint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol)
            if self.nesterov_algebraic:
                out = self.calc_algebraic_factor(out)
            if self.recf:
                rec = out[-1, :, -self.recf.osize:]
                out = out[:, :, :-self.recf.osize]
                out = out.reshape(-1, bsize, *self.shape)
                return out, rec
            else:
                return out
        else:
            out = odeint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol)
            if self.nesterov_algebraic:
                out = self.calc_algebraic_factor(out)
            return out

    @property
    def nfe(self):
        return self.df.nfe

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.evaluation_times.to(device)

    def calc_algebraic_factor(self, z):
        # split the input into the starting time step and the other time steps
        z_0 = z[:1]
        z_T = z[1:] 
        # get the corresponding value of t for the other time steps
        if len(self.evaluation_times.shape) == 2:
            T = self.evaluation_times[:, 1:]
        else:
            T = self.evaluation_times[1:]
        x, m = torch.split(z_T, 1, dim=2)
        # T^(-3/2) * e^(T/2)
        k = torch.pow(T, -3/2) * torch.exp(T / 2)
        if z.is_cuda:
            k = k.to(z.get_device())
            T = T.to(z.get_device())
        # h(T) = [x(T) m(T)] * Transpose([T^(-3/2)*e^(T/2) I])
        # h = x * k
        # dh = k * (m - (3/2 * torch.pow(T, 1/2) * torch.exp(-T/2) - 1/2 * 1/k) * h)
        k = self.activation_h(k)
        h = self.activation_output(bmul(k, x))
        dh = self.activation_output(bmul(k, m - bmul(3/2 * torch.pow(T, 1/2) * torch.exp(-T/2) - 1/2 * 1/k, h)))
        z_t = torch.cat((h, dh), dim=2)
        out = torch.cat((z_0, z_t), dim=0)
        return out
    
    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)


class NODElayer(NODEintegrate):
    def forward(self, x0):
        out = super(NODElayer, self).forward(x0)
        if isinstance(out, tuple):
            out, rec = out
            return out[-1], rec
        else:
            return out[-1]


'''
class ODERNN(nn.Module):
    def __init__(self, node, rnn, evaluation_times, nhidden):
        super(ODERNN, self).__init__()
        self.t = torch.as_tensor(evaluation_times).float()
        self.n_t = len(self.t)
        self.node = node
        self.rnn = rnn
        self.nhidden = (nhidden,) if isinstance(nhidden, int) else nhidden

    def forward(self, x):
        assert len(x) == self.n_t
        batchsize = x.shape[1]
        out = torch.zeros([self.n_t, batchsize, *self.nhidden]).to(x.device)
        for i in range(1, self.n_t):
            odesol = odeint(self.node, out[i - 1], self.t[i - 1:i + 1])
            h_ode = odesol[1]
            out[i] = self.rnn(h_ode, x[i])
        return out
'''


class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0
        self.elem_t = None

    def forward(self, t, x):
        self.nfe += 1
        if self.elem_t is None:
            return self.df(t, x)
        else:
            return self.elem_t * self.df(self.elem_t, x)

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1)


class SONODE(NODE):
    def forward(self, t, x):
        """
        Compute [y y']' = [y' y''] = [y' df(t, y, y')]
        :param t: time, shape [1]
        :param x: [y y'], shape [batch, 2, vec]
        :return: [y y']', shape [batch, 2, vec]
        """
        self.nfe += 1
        v = x[:, 1:, :]
        out = self.df(t, x)
        return torch.cat((v, out), dim=1)


class HeavyBallNODE(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True, sign=1):
        super().__init__(df)
        # Momentum parameter gamma
        self.gamma = Parameter([gamma_guess], frozen=False)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = Parameter([corr], frozen=corrf)
        self.sp = nn.Softplus()
        self.sign = sign # Sign of df
        self.actv_h = nn.Identity() if actv_h is None else actv_h # Activation for dh, GHBNODE only

    def forward(self, t, x):
        """
        Compute [theta' m' v'] with heavy ball parametrization in
        $$ h' = -m $$
        $$ m' = sign * df - gamma * m $$
        based on paper https://www.jmlr.org/papers/volume21/18-808/18-808.pdf
        :param t: time, shape [1]
        :param x: [theta m], shape [batch, 2, dim]
        :return: [theta' m'], shape [batch, 2, dim]
        """
        self.nfe += 1
        h, m = torch.split(x, 1, dim=1)
        dh = self.actv_h(- m)
        dm = self.df(t, h) * self.sign - self.gammaact(self.gamma()) * m
        dm = dm + self.sp(self.corr()) * h
        out = torch.cat((dh, dm), dim=1)
        if self.elem_t is None:
            return out
        else:
            return self.elem_t * out

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1, 1)


HBNODE = HeavyBallNODE # Alias


class NesterovNODE(NODE):
    def __init__(self, df, actv_h=None, corr=-100, corrf=True, use_h=False, full_details=False, nesterov_algebraic=True, actv_m=None, actv_dm=None, actv_df=None, sign=1):
        super().__init__(df)
        self.corr = Parameter([corr], frozen=corrf)
        self.sp = nn.Softplus()
        self.sign = sign # Sign of df
        self.actv_h = nn.Identity() if actv_h is None else actv_h # Activation for dh, GNNODE only
        self.actv_m = nn.Identity() if actv_m is None else actv_m # Activation for dh, GNNODE only
        self.actv_dm = nn.Identity() if actv_dm is None else actv_dm # Activation for dh, GNNODE only
        self.actv_df = nn.Identity() if actv_df is None else actv_df # Activation for df, GNNODE only
        self.use_h = use_h
        self.full_details = full_details
        self.nesterov_algebraic = nesterov_algebraic

    def forward(self, t, z):
        """
        Compute [x' m'] with diff-alg nesterov parametrization in
        $$ h' = -m $$
        $$ m' = sign * df(t, h) - m - xi * h $$
        :param t: time, shape [1]
        :param z: [h dh], shape [batch, 2, dim]
        :return: [x' m'], shape [batch, 2, dim]
        """
        self.nfe += 1
        h, dh = torch.split(z, 1, dim=1)
        k_reciprocal = torch.pow(t, 3/2) * torch.exp(-t/2)
        if z.is_cuda:
            k_reciprocal = k_reciprocal.to(z.get_device())
        m = (3/2 * torch.pow(t, 1/2) * torch.exp(-t/2) - 1/2 * k_reciprocal) * h \
                + k_reciprocal * dh
        # x = h * k_reciprocal
        dx = self.actv_h(m)
        dm = self.actv_df(self.df(t, h)) * self.sign - m
        dm = self.actv_dm(self.actv_m(dm) - self.sp(self.corr()) * h)
        out = torch.cat((dx, dm), dim=1)
        if self.elem_t is None:
            return out
        else:
            return self.elem_t * out

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1, 1)

NNODE = NesterovNODE # Alias

class ODE_RNN(nn.Module):
    def __init__(self, ode, rnn, nhid, ic, rnn_out=False, both=False, tol=1e-7):
        super().__init__()
        self.ode = ode
        self.t = torch.Tensor([0, 1])
        self.nhid = [nhid] if isinstance(nhid, int) else nhid
        self.rnn = rnn
        self.tol = tol
        self.rnn_out = rnn_out
        self.ic = ic
        self.both = both

    def forward(self, t, x, multiforecast=None):
        """
        --
        :param t: [time, batch]
        :param x: [time, batch, ...]
        :return: [time, batch, *nhid]
        """
        n_t, n_b = t.shape
        h_ode = torch.zeros(n_t + 1, n_b, *self.nhid, device=x.device)
        h_rnn = torch.zeros(n_t + 1, n_b, *self.nhid, device=x.device)
        if self.ic:
            h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 't b c -> b (t c)')).view(h_ode[0].shape)
        if self.rnn_out:
            for i in range(n_t):
                self.ode.update(t[i])
                h_ode[i] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
                h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
            out = (h_rnn,)
        else:
            for i in range(n_t):
                self.ode.update(t[i])
                h_rnn[i] = self.rnn(h_ode[i], x[i])
                h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
            out = (h_ode,)

        if self.both:
            out = (h_rnn, h_ode)

        if multiforecast is not None:
            self.ode.update(torch.ones_like((t[0])))
            forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol)
            out = (*out, forecast)

        return out


class ODE_RNN_with_Grad_Listener(nn.Module):
    def __init__(self, ode, rnn, nhid, ic, rnn_out=False, both=False, tol=1e-7, method="dopri5", evaluation_times=None, nesterov_algebraic=None, activation_h=None, time_requires_grad=True):
        super().__init__()
        self.ode = ode
        self.evaluation_times = evaluation_times if evaluation_times is not None else torch.Tensor([0.0, 1.0])
        if time_requires_grad == False:
            self.evaluation_times.requires_grad = False
        self.nesterov_algebraic = nesterov_algebraic
        self.activation_h = nn.Identity() if activation_h is None else activation_h
        self.nhid = [nhid] if isinstance(nhid, int) else nhid
        self.rnn = rnn
        self.tol = tol
        self.rnn_out = rnn_out
        self.ic = ic
        self.both = both
        self.method = method

    def forward(self, t, x, multiforecast=None, retain_grad=False):
        """
        --
        :param t: [time, batch]
        :param x: [time, batch, ...]
        :return: [time, batch, *nhid]
        """
        n_t, n_b = t.shape
        h_ode = [None] * (n_t + 1)
        h_rnn = [None] * (n_t + 1)
        h_ode[-1] = h_rnn[-1] = torch.zeros(n_b, *self.nhid, device=x.device)

        if self.ic:
            h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 't b c -> b (t c)')).view((n_b, *self.nhid))
        else:
            h_ode[0] = h_rnn[0] = torch.zeros(n_b, *self.nhid, device=x.device)
        if self.rnn_out:
            for i in range(n_t):
                self.ode.update(t[i])
                h_ode[i] = odeint(self.ode, h_rnn[i], self.evaluation_times, atol=self.tol, rtol=self.tol, method=self.method)[-1]
                if self.nesterov_algebraic:
                    h_ode[i] = self.calc_algebraic_factor(h_ode[i])
                h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
            out = (h_rnn,)
        else:
            for i in range(n_t):
                self.ode.update(t[i])
                h_rnn[i] = self.rnn(h_ode[i], x[i])
                h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.evaluation_times, atol=self.tol, rtol=self.tol, method=self.method)[-1]
                if self.nesterov_algebraic:
                    h_ode[i + 1] = self.calc_algebraic_factor(h_ode[i + 1])
            out = (h_ode,)

        if self.both:
            out = (h_rnn, h_ode)

        out = [torch.stack([k.to(x.device) for k in h], dim=0) for h in out]

        if multiforecast is not None:
            self.ode.update(torch.ones_like((t[0]), device=x.device))
            forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol, method=self.method)
            if self.nesterov_algebraic:
                forecast = self.calc_algebraic_factor(forecast)
            out = (*out, forecast)

        if retain_grad:
            self.h_ode = h_ode
            self.h_rnn = h_rnn
            for i in range(n_t + 1):
                if self.h_ode[i].requires_grad:
                    self.h_ode[i].retain_grad()
                if self.h_rnn[i].requires_grad:
                    self.h_rnn[i].retain_grad()

        return out

    def calc_algebraic_factor(self, z):
        # split the input into the starting time step and the other time steps
        z_0 = z[:1]
        z_T = z[1:] 
        # get the corresponding value of t for the other time steps
        if len(self.evaluation_times.shape) == 2:
            T = self.evaluation_times[:, 1:]
        else:
            T = self.evaluation_times[1:]
        x, m = torch.split(z_T, 1, dim=-2)
        # T^(-3/2) * e^(T/2)
        k = torch.pow(T, -3/2) * torch.exp(T / 2)
        if z.is_cuda:
            k = k.to(z.get_device())
            T = T.to(z.get_device())
        # h(T) = [x(T) m(T)] * Transpose([T^(-3/2)*e^(T/2) I])
        # h = x * k
        # dh = k * (m - (3/2 * torch.pow(T, 1/2) * torch.exp(-T/2) - 1/2 * 1/k) * h)
        k = self.activation_h(k)
        h = bmul(k, x)
        dh = bmul(k, m - bmul(3/2 * torch.pow(T, 1/2) * torch.exp(-T/2) - 1/2 * 1/k, h))
        z_t = torch.cat((h, dh), dim=-2)
        out = torch.cat((z_0, z_t), dim=0)
        return out
