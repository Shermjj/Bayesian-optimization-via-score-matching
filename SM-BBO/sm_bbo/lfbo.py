# Code is based on https://github.com/lfbo-ml/lfbo

from typing import Any
import torch
import torch.nn as nn

from torch import Tensor

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors import DeterministicPosterior
torch.set_default_dtype(torch.float32)

def run_lfbo(X_obs, 
             f,
             param_dim,
             n_iter = 250):
    for i in range(n_iter):
        model = Network(param_dim, 
                        1, 
                        2, 
                        32)

        X, Y, W = prepare_data(X_obs, f)
        model = train_model(model, X, Y, W)

        acqf = LFAcquisitionFunction(model)

        a = optimize_acqf(acqf, 
                        bounds=torch.tensor(f._bounds).T, 
                        q=1, 
                        num_restarts=5,
                        raw_samples=100)[0]

        # print(f'par:{a}, val:{f(a)}')
        X_obs = torch.cat([X_obs, a], dim=0)
    return X_obs
        
class Network(Model):
    def __init__(self, input_dim, output_dim, num_layers, num_units):
        super(Network, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if not i:
                self.layers.append(nn.Linear(input_dim, num_units))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(num_units, num_units))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(num_units, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def posterior(self, X: Tensor, **kwargs: Any) -> DeterministicPosterior:
        y = self.forward(X).view(-1)
        return DeterministicPosterior(y)


class LFAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model: Model) -> None:
        super().__init__(model)

    def forward(self, X):
        return self.model.posterior(X).mean


def train_model(model, 
                X, 
                Y, 
                W,
                lr=1e-3,
                n_iter=300):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)
    for _ in range(n_iter):
        optimizer.zero_grad()
        y_ = model(X)
        loss = nn.BCEWithLogitsLoss(weight=W)(y_, Y)
        loss.backward()
        optimizer.step()

    return model


def prepare_data(X, f, eta=1.0):
    fx = f(X).view(-1)
    tau = torch.quantile(fx, 0.33)

    y = torch.less(fx, tau)
    x1, y1 = X[y], y[y]
    x0, y0 = X, torch.zeros_like(y)
    w1 = (tau - fx)[y]
    w1 = w1 ** eta / torch.mean(w1)
    w0 = 1 - y0.float()
    s1 = x1.size(0)
    s0 = x0.size(0)

    X = torch.cat([x1, x0], dim=0)
    Y = torch.cat([y1, y0], dim=0).float().view(-1, 1)
    W = torch.cat([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], dim=0).view(-1, 1)
    W = W / W.mean()
    return X.float(), Y.float(), W.float()