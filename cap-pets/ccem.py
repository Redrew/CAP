import math
import numpy as np
import torch
from torch.optim import Adam

import scipy.stats as stats

STATE_MAX = 100

class ConstrainedCEM:
    def __init__(self,
                 env,
                 epoch_length=1000,
                 plan_hor=30,
                 gamma=0.99,
                 c_gamma=0.99,
                 kappa=0,
                 binary_cost=False,
                 cost_limit=0,
                 learn_kappa=False,
                 cost_constrained=True,
                 penalize_uncertainty=True,
                 cuda=True,
                 ):
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.gamma = gamma
        self.c_gamma = c_gamma

        self.learn_kappa = learn_kappa
        if not self.learn_kappa:
            self.kappa = torch.tensor([kappa], requires_grad=False)
        else:
            self.kappa = torch.tensor([float(kappa)], requires_grad=True)
            self.kappa_optim = Adam([self.kappa], lr=0.1)
        self.binary_cost = binary_cost

        self.cost_limit = cost_limit
        self.epoch_length = epoch_length
        self.cost_constrained = cost_constrained
        self.penalize_uncertainty = penalize_uncertainty
        self.device = 'cuda' if cuda else 'cpu'

        # CEM parameters
        self.per = 1
        self.npart = 20
        self.plan_hor = plan_hor
        self.popsize = 500
        self.num_elites = 50
        self.max_iters = 5
        self.alpha = 0.1
        self.epsilon = 0.001
        self.lb = np.tile(self.ac_lb, [self.plan_hor])
        self.ub = np.tile(self.ac_ub, [self.plan_hor])
        self.decay = 1.25
        self.elite_fraction = 0.3
        self.elites = None

        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        self.model = None
        self.step = 0

    def set_model(self, model):
        self.model = model

    def select_action(self, obs, eval_t=False):
        if self.model is None:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        soln = self.obtain_solution(obs, self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

        return self.select_action(obs)

    def obtain_solution(self, obs, init_mean, init_var):
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            noise = X.rvs(size=[self.popsize, self.plan_hor * self.dU])

            samples = noise * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)
            
            rewards, costs, eps_lens = self.rollout(obs, samples)
            epoch_ratio = np.ones_like(eps_lens) * self.epoch_length / self.plan_hor
            terminated = eps_lens != self.plan_hor
            if self.c_gamma == 1:
                c_gamma_discount = epoch_ratio
            else:
                c_gamma_discount = (1 - self.c_gamma ** (epoch_ratio * self.plan_hor)) / (1 - self.c_gamma) / self.plan_hor
            rewards = rewards * epoch_ratio
            costs = costs * c_gamma_discount

            feasible_ids = ((costs <= self.cost_limit) & (~terminated)).nonzero()[0]
            if self.cost_constrained:
                if feasible_ids.shape[0] >= self.num_elites:
                    elite_ids = feasible_ids[np.argsort(-rewards[feasible_ids])][:self.num_elites]
                else:
                    elite_ids = np.argsort(costs)[:self.num_elites]
            else:
                elite_ids = np.argsort(-rewards)[:self.num_elites]
            self.elites = samples[elite_ids]
            new_mean = np.mean(self.elites, axis=0)
            new_var = np.var(self.elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            average_reward = rewards.mean().item()
            average_cost = costs.mean().item()
            average_len = eps_lens.mean().item()
            average_elite_reward = rewards[elite_ids].mean().item()
            average_elite_cost = costs[elite_ids].mean().item()
            average_elite_len = eps_lens[elite_ids].mean().item()
            if t == 0:
                start_reward = average_reward
                start_cost = average_cost
            t += 1
        
        self.step += 1
        return mean

    @torch.no_grad()
    def rollout(self, obs, ac_seqs):
        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(self.device)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # Before, ac seqs has dimension (400, 25) which are pop size and sol dim coming from CEM
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        #  After, ac seqs has dimension (400, 25, 1)

        transposed = ac_seqs.transpose(0, 1)
        # Then, (25, 400, 1)

        expanded = transposed[:, :, None]
        # Then, (25, 400, 1, 1)

        tiled = expanded.expand(-1, -1, self.npart, -1)
        # Then, (25, 400, 20, 1)

        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)
        # Then, (25, 8000, 1)

        # Expand current observation
        cur_obs = torch.from_numpy(obs).float().to(self.device)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        rewards = torch.zeros(nopt, self.npart, device=self.device)
        costs = torch.zeros(nopt, self.npart, device=self.device)
        length = torch.zeros(nopt, self.npart, device=self.device)

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            cur_obs, reward, cost = self._predict_next(cur_obs, cur_acs)
            # Clip state value
            cur_obs = torch.clamp(cur_obs, -STATE_MAX, STATE_MAX)
            reward = reward.view(-1, self.npart)
            cost = cost.view(-1, self.npart)

            rewards += reward
            costs += cost
            length += 1

            if t == 0:
                start_reward = reward
                start_cost = cost

        # Replace nan with high cost
        rewards[rewards != rewards] = -1e6
        costs[costs != costs] = 1e6

        return rewards.mean(dim=1).detach().cpu().numpy(), costs.mean(dim=1).detach().cpu().numpy(), length.mean(dim=1).detach().cpu().numpy()

    def optimize_kappa(self, episode_cost, permissible_cost=None):
        if permissible_cost is None:
            permissible_cost = self.cost_limit
        kappa_loss = -(self.kappa * (episode_cost - permissible_cost))

        self.kappa_optim.zero_grad()
        kappa_loss.backward()
        self.kappa_optim.step()

    def _predict_next(self, obs, acs):
        proc_obs = self._expand_to_ts_format(obs)
        proc_acs = self._expand_to_ts_format(acs)

        output = self.model.predict(proc_obs, proc_acs)
        next_obs, var = output["state"]
        reward, _ = output["reward"]
        cost, _ = output["cost"]

        next_obs = self._flatten_to_matrix(next_obs)
        reward = self._flatten_to_matrix(reward)
        cost = self._flatten_to_matrix(cost)

        obs = obs.detach().cpu().numpy()
        acs = acs.detach().cpu().numpy()

        if self.cost_constrained and self.penalize_uncertainty:
            cost_penalty = var.sqrt().norm(dim=2).max(0)[0]
            cost_penalty = cost_penalty.repeat_interleave(self.model.num_nets).view(cost.shape)
            cost += self.kappa.to(cost_penalty.device) * cost_penalty
        if self.binary_cost:
            cost = (torch.sigmoid(cost) > 0.5).float()

        return next_obs, reward, cost

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.model.num_nets, self.npart // self.model.num_nets, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)
        # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.model.num_nets, -1, self.npart // self.model.num_nets, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped

