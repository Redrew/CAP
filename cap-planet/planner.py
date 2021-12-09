from math import inf
import torch
from torch import jit
import numpy as np
import math

DEFAULT_UNCERTAINTY_MULTIPLIER = 1000
DEFAULT_UNCERTAINTY_MULTIPLIER_BINARY = 10000

# Model-predictive control planner with cross-entropy method and learned transition model
# class MPCPlanner(jit.ScriptModule):
class MPCPlanner(torch.nn.Module):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action', 'cost_constrained', 'cost_limit_per_step']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, cost_model, one_step_ensemble,
                 min_action=-inf, max_action=inf, cost_constrained=False, penalize_uncertainty=True,
                 cost_limit=0, action_repeat=2, max_length=1000, binary_cost=False, cost_discount=0.99, penalty_kappa=0, lr=0.01):
        super().__init__()
        self.transition_model, self.reward_model, self.cost_model, self.one_step_ensemble = transition_model, reward_model, cost_model, one_step_ensemble
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.cost_constrained = cost_constrained
        self.penalize_uncertainty = penalize_uncertainty
        self.binary_cost = binary_cost
        self.set_cost_limit(cost_limit, cost_discount, action_repeat, max_length)
        self.penalty_kappa = torch.tensor([float(penalty_kappa)], requires_grad=True)
        self.kappa_optim = torch.optim.Adam([self.penalty_kappa], lr=lr)
        self._fixed_cost_penalty = 0
        if self.binary_cost:
            self.uncertainty_multiplier = DEFAULT_UNCERTAINTY_MULTIPLIER_BINARY
        else:
            self.uncertainty_multiplier = DEFAULT_UNCERTAINTY_MULTIPLIER

    def set_cost_limit(self, cost_limit, cost_discount, action_repeat, max_length):
        self.cost_limit = cost_limit
        steps = max_length / action_repeat
        if cost_discount == 1:
            self.cost_limit_per_step = cost_limit / steps
        else:
            self.cost_limit_per_step = cost_limit * (1 - cost_discount ** action_repeat) / (1 - cost_discount ** max_length)

    def optimize_penalty_kappa(self, episode_cost, cost_limit=None):
        if cost_limit is None:
            cost_limit = self.cost_limit
        kappa_loss = -(self.penalty_kappa * (episode_cost - cost_limit))

        self.kappa_optim.zero_grad()
        kappa_loss.backward()
        self.kappa_optim.step()

    # @jit.script_method
    def forward(self, belief, state):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
            actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
            # Sample next states
            beliefs, states, _, states_std = self.transition_model(state, actions, belief)
            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).mean(dim=0)
            objective = returns
            if self.cost_constrained:
                costs = self.cost_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1)
                costs += self._fixed_cost_penalty
                uncertainty = self.one_step_ensemble.compute_uncertainty(beliefs, actions)
                if self.penalize_uncertainty:
                    penalty_kappa = self.penalty_kappa.detach().to(costs.device)
                    costs += penalty_kappa * self.uncertainty_multiplier * uncertainty
                if self.binary_cost:
                    logits = costs
                    costs = (torch.sigmoid(costs) > 0.5).float()
                avg_costs = costs.mean(dim=0)
                feasible_samples = (avg_costs <= self.cost_limit_per_step)
                objective[~feasible_samples] = - np.inf
                if feasible_samples.sum() < self.top_candidates:
                    objective = - avg_costs
            # Re-fit belief to the K best action sequences
            _, topk = objective.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
            # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
            # Return first action mean µ_t
        if self.penalize_uncertainty:
            self.uncertainty_last_step = uncertainty[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates).mean(2).cpu()
        return action_mean[0].squeeze(dim=1)
