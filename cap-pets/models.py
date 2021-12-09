import numpy as np
import gym
import itertools

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    val = torch.fmod(torch.randn(size),2) * std
    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b

class EnsembleLayer(nn.Module):
    def __init__(self, network_size, in_size, out_size):
        super().__init__()
        self.w, self.b = get_affine_params(network_size, in_size, out_size)
    
    def forward(self, inputs):
        inputs = inputs.matmul(self.w) + self.b
        inputs = swish(inputs)
        return inputs

    def decays(self):
        return (self.w ** 2).sum() / 2.0

class GaussianEnsembleLayer(nn.Module):
    def __init__(self, network_size, in_size, out_size):
        super().__init__()
        self.w, self.b = get_affine_params(network_size, in_size, out_size * 2)
        self.out_size = out_size
        self.max_logvar = nn.Parameter(torch.ones(1, self.out_size, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.out_size, dtype=torch.float32) * 10.0)
    
    def forward(self, inputs, sample=True):
        outputs = inputs.matmul(self.w) + self.b
        mean = outputs[:, :, :self.out_size]
        logvar = outputs[:, :, self.out_size:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        var = torch.exp(logvar)
        if sample:
            noise = torch.randn_like(mean, device=inputs.device) * var.sqrt()
            return mean + noise * var.sqrt(), var
        else:
            return mean, var

    def decays(self):
        return (self.w ** 2).sum() / 2.0

    def compute_loss(self, output, target):
        train_loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        mean, var = output
        logvar = torch.log(var)

        inv_var = torch.pow(var, -1)
        train_losses = ((mean - target) ** 2) * inv_var + logvar
        train_losses = train_losses.mean(-1).mean(-1).sum()
        train_loss += train_losses
        return train_loss

class LogisticEnsembleLayer(nn.Module):
    def __init__(self, network_size, in_size, out_size):
        super().__init__()
        self.w, self.b = get_affine_params(network_size, in_size, out_size)
    
    def forward(self, inputs, sample=True):
        logits = inputs.matmul(self.w) + self.b
        return logits, None

    def decays(self):
        return (self.w ** 2).sum() / 2.0

    def compute_loss(self, output, target):
        logits, _ = output
        mean = torch.sigmoid(logits)

        train_loss = F.binary_cross_entropy(mean, target, reduce=False)
        train_loss = train_loss.mean(-1).mean(-1).sum()
        return train_loss

class ProbEnsemble(nn.Module):

    def __init__(self, state_size, action_size,
                 network_size=7, elite_size=5, cuda=True,
                 cost=False, binary_cost=False, hidden_size=200, lr=0.001):
        super().__init__()
        self.network_size = network_size
        self.num_nets = network_size
        self.state_size = state_size
        self.action_size = action_size
        self.binary_cost = binary_cost
        self.elite_size = elite_size
        self.elite_model_idxes = []

        self.in_features = state_size + action_size

        self.layer0 = EnsembleLayer(network_size, self.in_features, hidden_size)
        self.layer1 = EnsembleLayer(network_size, hidden_size, hidden_size)
        self.layer2 = EnsembleLayer(network_size, hidden_size, hidden_size)
        self.layer3 = EnsembleLayer(network_size, hidden_size, hidden_size)

        self.state_model = GaussianEnsembleLayer(network_size, hidden_size, state_size)
        self.reward_model = GaussianEnsembleLayer(network_size, hidden_size, 1)
        if binary_cost:
            self.cost_model = LogisticEnsembleLayer(network_size, hidden_size, 1)
        else:
            self.cost_model = GaussianEnsembleLayer(network_size, hidden_size, 1)
        
        self.inputs_mu = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=False)

        self.fit_input = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.grad_update = 0

        self.device = 'cuda' if cuda else 'cpu'

    def compute_decays(self):

        lin0_decays = 0.000025 * self.layer0.decays()
        lin1_decays = 0.00005 * self.layer1.decays()
        lin2_decays = 0.000075 * self.layer2.decays()
        lin3_decays = 0.000075 * self.layer3.decays()
        lin4_decays = 0.0001 * (self.state_model.decays() + self.reward_model.decays() + self.cost_model.decays())

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays + lin4_decays

    def fit_input_stats(self, data):
        self.fit_input = True
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(self.device).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(self.device).float()

    def forward(self, inputs, sample=False, return_state_variance=False):
        # Transform inputs
        if self.fit_input:
            inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        hidden = self.layer0(inputs)
        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)

        output = {
            "state": self.state_model(hidden, sample=sample),
            "reward": self.reward_model(hidden, sample=sample),
            "cost": self.cost_model(hidden, sample=sample),
        }
        return output

    def compute_loss(self, input, target):
        output = self(input)
        train_loss = self.state_model.compute_loss(output["state"], target["state"])
        train_loss += self.reward_model.compute_loss(output["reward"], target["reward"])
        train_loss += self.cost_model.compute_loss(output["cost"], target["cost"])
        train_loss += self.compute_decays()
        return train_loss

    def _save_best(self, epoch, holdout_losses):
        updated = False
        updated_count = 0
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / abs(best)
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True
                updated_count += 1
                improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def train(self, inputs, targets, batch_size=256, max_epochs_since_update=5, max_epochs=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._snapshots = {i: (None, 1e10) for i in range(self.num_nets)}
        self._epochs_since_update = 0

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        self.fit_input_stats(inputs)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])

        if max_epochs is not None:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        for epoch in epoch_iter:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]

                input = torch.from_numpy(inputs[batch_idxs]).float().to(self.device)
                target = { key: torch.from_numpy(value[batch_idxs]).float().to(self.device)
                           for key, value in targets.items() }
                train_loss = self.compute_loss(input, target)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                self.grad_update += 1

            idxs = shuffle_rows(idxs)

    def predict(self, state, action, variance=False):
        input = torch.cat([state, action], dim=-1)
        with torch.no_grad():
            output = self(input, sample=True)
            output["state"] = (output["state"][0] + state, output["state"][1])
        return output
