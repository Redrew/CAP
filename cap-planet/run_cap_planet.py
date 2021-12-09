import argparse
from math import inf
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, CostModel, LogisticCostModel, TransitionModel, OneStepEnsemble
from planner import MPCPlanner
from utils import lineplot, violinplot, write_video


# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--cost-constrained', action='store_true', help='Follow cost constraint')
parser.add_argument('--penalize-uncertainty', action='store_true', help='Penalize model uncertainty as an additional cost')
parser.add_argument('--penalty-kappa', type=float, default=0, metavar='λ', help='The penalty kappa')
parser.add_argument('--learn-kappa', action='store_true', help='Learn the penalty kappa')
parser.add_argument('--binary-cost', action='store_true')
parser.add_argument('--cost-limit', type=float, default=0, metavar='CL', help='Discounted cost limit')
parser.add_argument('--permissible-cost', type=float, default=None, metavar='CL_', help='Cost allowed for exploration')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Reward discount factor')
parser.add_argument('--cost-discount', type=float, default=0.99, metavar='γ', help='Cost discount factor')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.0, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=12, metavar='H', help='Planning horizon distance')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
if args.permissible_cost is None:
    args.permissible_cost = args.cost_limit
if args.binary_cost:
    print("Using cost discount of 1 for binary cost")
    args.cost_discount = 1
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', args.id)
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'config.txt'), 'w') as config_file:
    json.dump(args.__dict__, config_file, indent=2)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'train_costs': [], 'test_episodes': [], 'test_rewards': [],
           'test_costs': [], 'observation_loss': [], 'reward_loss': [], 'cost_loss': [], 'kl_loss': [], 'one_step_loss': [],
           'penalty_kappa': [], 'test_eps_costs': [], 'test_eps_uncertainty': [], 'cost_violations': [], 'test_cost_violations': []}


# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, cost_reduce="max" if args.binary_cost else "sum")
test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, None, args.max_episode_length, args.action_repeat, args.bit_depth, "max" if args.binary_cost else "sum"), {}, args.test_episodes)
if args.experience_replay is not '' and os.path.exists(args.experience_replay):
  D = torch.load(args.experience_replay)
  metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
  D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
  # Initialise dataset D with S random seed episodes
  for s in range(1, args.seed_episodes + 1):
    observation, done, t = env.reset(), False, 0
    total_cost = 0
    while not done:
      action = env.sample_random_action()
      next_observation, reward, cost, done = env.step(action)
      total_cost += cost * args.cost_discount ** (args.action_repeat * (t + 0.5))
      D.append(observation, action, reward, cost, done)
      observation = next_observation
      t += 1
    metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
    metrics['episodes'].append(s)
    metrics['cost_violations'].append(total_cost > args.cost_limit)


# Initialise model parameters randomly
transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.activation_function).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.activation_function).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.activation_function).to(device=args.device)
if args.binary_cost:
    cost_model = LogisticCostModel(args.belief_size, args.state_size, args.hidden_size, args.activation_function).to(device=args.device)
else:
    cost_model = CostModel(args.belief_size, args.state_size, args.hidden_size, args.activation_function).to(device=args.device)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.activation_function).to(device=args.device)
one_step_ensemble = OneStepEnsemble(args.belief_size, env.action_size, args.state_size)
param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(cost_model.parameters()) + list(encoder.parameters()) + list(one_step_ensemble.parameters())
optimiser = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)
planner = MPCPlanner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model, cost_model, one_step_ensemble,
                     env.action_range[0], env.action_range[1],
                     cost_constrained=args.cost_constrained, penalize_uncertainty=args.penalize_uncertainty,
                     cost_limit=args.cost_limit, cost_discount=args.cost_discount, action_repeat=args.action_repeat, max_length=args.max_episode_length,
                     penalty_kappa=args.penalty_kappa, binary_cost=args.binary_cost).to(device=args.device)
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  transition_model.load_state_dict(model_dicts['transition_model'])
  observation_model.load_state_dict(model_dicts['observation_model'])
  reward_model.load_state_dict(model_dicts['reward_model'])
  cost_model.load_state_dict(model_dicts['cost_model'])
  encoder.load_state_dict(model_dicts['encoder'])
  one_step_ensemble.load_state_dict(model_dicts['one_step_ensemble'])
  optimiser.load_state_dict(model_dicts['optimiser'])
  planner_state = planner.state_dict()
  planner_state.update(model_dicts['planner'])
  planner.load_state_dict(planner_state)
  metrics = torch.load(os.path.join(os.path.dirname(args.models), 'metrics.pth'))
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, dtype=torch.float32, device=args.device)  # Allowed deviation in KL divergence


def update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, min_action=-inf, max_action=inf, explore=False):
  # Infer belief over current state q(s_t|o≤t,a<t) from the history
  belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
  belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
  action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
  if explore:
    action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
  action.clamp_(min=min_action, max=max_action)  # Clip action range
  next_observation, reward, cost, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
  return belief, posterior_state, action, next_observation, reward, cost, done


# Testing only
if args.test:
  # Set models to eval mode
  transition_model.eval()
  reward_model.eval()
  encoder.eval()
  one_step_ensemble.eval()
  with torch.no_grad():
    for eps in tqdm(range(args.test_episodes)):
      total_reward, total_cost = 0, 0
      video_frames= []
      observation = env.reset()
      belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        belief, posterior_state, action, observation, reward, cost, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), env.action_range[0], env.action_range[1])
        video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
        total_reward += reward * args.discount ** (args.action_repeat * (t + 0.5))
        total_cost += cost * args.cost_discount ** (args.action_repeat * (t + 0.5))
        if args.render:
          env.render()
        if done:
          pbar.close()
          break
      print("Reward:", total_reward, "Cost:", total_cost)
      write_video(video_frames, f'test_episode{eps}', 'test_results')  # Lossy compression
  env.close()
  quit()


# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
  # Model fitting
  losses = []
  for s in tqdm(range(args.collect_interval)):
    # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
    observations, actions, rewards, costs, nonterminals = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
    embeddings = bottle(encoder, (observations[1:], ))
    # Create initial belief and state for time t = 0
    init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(args.batch_size, args.state_size, device=args.device)
    # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(init_state, actions[:-1], init_belief, embeddings, nonterminals[:-1])
    # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
    observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
    reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))
    cost_loss = cost_model.loss(bottle(cost_model, (beliefs, posterior_states)), costs[:-1], reduction='none').mean(dim=(0, 1))
    one_step_loss = one_step_ensemble.loss(one_step_ensemble(beliefs, actions[:-1]), posterior_states)
    kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
    if args.global_kl_beta != 0:
      kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0, 1))
    # Calculate latent overshooting objective for t > 0
    if args.overshooting_kl_beta != 0:
      overshooting_vars = []  # Collect variables for overshooting to process in batch
      for t in range(1, args.chunk_size - 1):
        d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
        t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
        seq_pad = (0, 0, 0, 0, 0, t - d + args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
        # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
        overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
      overshooting_vars = tuple(zip(*overshooting_vars))
      # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
      beliefs, prior_states, prior_means, prior_std_devs = transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
      seq_mask = torch.cat(overshooting_vars[7], dim=1)
      # Calculate overshooting KL loss with sequence mask
      kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 
      # Calculate overshooting reward prediction loss with sequence mask
      if args.overshooting_reward_scale != 0:
        reward_loss += (1 / args.overshooting_distance) * args.overshooting_reward_scale * F.mse_loss(bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 

    # Apply linearly ramping learning rate schedule
    if args.learning_rate_schedule != 0:
      for group in optimiser.param_groups:
        group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
    # Update model parameters
    optimiser.zero_grad()
    (observation_loss + reward_loss + cost_loss + one_step_loss + kl_loss).backward()
    nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
    optimiser.step()
    # Store (0) observation loss (1) reward loss (2) cost loss (3) one step ensemble loss (4) KL loss
    losses.append([observation_loss.item(), reward_loss.item(), cost_loss.item(), one_step_loss.item(), kl_loss.item()])

  # Update and plot loss metrics
  losses = tuple(zip(*losses))
  metrics['observation_loss'].append(losses[0])
  metrics['reward_loss'].append(losses[1])
  metrics['cost_loss'].append(losses[2])
  metrics['one_step_loss'].append(losses[3])
  metrics['kl_loss'].append(losses[4])
  lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['cost_loss']):], metrics['cost_loss'], 'cost_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['one_step_loss']):], metrics['one_step_loss'], 'one_step_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)


  # Data collection
  with torch.no_grad():
    observation, total_reward, total_cost = env.reset(), 0, 0
    belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
    pbar = tqdm(range(args.max_episode_length // args.action_repeat))
    for t in pbar:
      belief, posterior_state, action, next_observation, reward, cost, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), env.action_range[0], env.action_range[1], explore=True)
      D.append(observation, action.cpu(), reward, cost, done)
      total_reward += reward * args.discount ** (args.action_repeat * (t + 0.5))
      total_cost += cost * args.cost_discount ** (args.action_repeat * (t + 0.5))
      observation = next_observation
      if args.render:
        env.render()
      if done:
        pbar.close()
        break
    

  if args.learn_kappa:
    planner.optimize_penalty_kappa(total_cost, args.permissible_cost)
  # Update and plot train reward metrics
  metrics['steps'].append(t * args.action_repeat + metrics['steps'][-1])
  metrics['episodes'].append(episode)
  metrics['train_rewards'].append(total_reward)
  metrics['train_costs'].append(total_cost)
  metrics['penalty_kappa'].append(planner.penalty_kappa.item())
  metrics['cost_violations'].append(total_cost > args.cost_limit)

  lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)
  lineplot(metrics['episodes'][-len(metrics['train_costs']):], metrics['train_costs'], 'train_costs', results_dir)
  lineplot(metrics['episodes'][-len(metrics['penalty_kappa']):], metrics['penalty_kappa'], 'penalty_kappa', results_dir)
  

  # Test model
  if episode % args.test_interval == 0:
    # Set models to eval mode
    transition_model.eval()
    observation_model.eval()
    reward_model.eval()
    cost_model.eval()
    encoder.eval()
    one_step_ensemble.eval()
    planner.eval()
    # Initialise parallelised test environments
    
    with torch.no_grad():
      observation, total_rewards, total_costs, video_frames = test_envs.reset(), np.zeros((args.test_episodes, )), np.zeros((args.test_episodes, )), []
      belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        belief, posterior_state, action, next_observation, reward, cost, done = update_belief_and_act(args, test_envs, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), env.action_range[0], env.action_range[1])
        total_rewards += reward.numpy() * args.discount ** (args.action_repeat * (t + 0.5))
        total_costs += cost.numpy() * args.cost_discount ** (args.action_repeat * (t + 0.5))
        if not args.symbolic_env:  # Collect real vs. predicted frames for video
          video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
        metrics['test_eps_costs'].append(cost.numpy())
        if args.penalize_uncertainty:
            uncertainty = planner.uncertainty_last_step[0]
            metrics['test_eps_uncertainty'].append(uncertainty.numpy())
        observation = next_observation
        if done.sum().item() == args.test_episodes:
          test_envs.reset()
          t += 1
          pbar.close()
          break
    
    # Update and plot reward metrics (and write video if applicable) and save metrics
    metrics['test_episodes'].append(episode)
    metrics['test_rewards'].append(total_rewards.tolist())
    metrics['test_costs'].append(total_costs.tolist())
    metrics['test_cost_violations'].append((total_costs > args.cost_limit).tolist())
    lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
    lineplot(metrics['test_episodes'], metrics['test_costs'], 'test_costs', results_dir)
    lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
    lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_costs'], 'test_costs_steps', results_dir, xaxis='step')
    episode_str = str(episode).zfill(len(str(args.episodes)))
    if args.penalize_uncertainty:
        violinplot(metrics['test_eps_costs'][-t:], metrics['test_eps_uncertainty'][-t:], f'test_episode_{episode_str}_cost_uncertainty', results_dir, xaxis='cost')
    if not args.symbolic_env:
      write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
      save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
    torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Set models to train mode
    transition_model.train()
    observation_model.train()
    reward_model.train()
    cost_model.train()
    encoder.train()
    one_step_ensemble.train()
    planner.train()


  # Checkpoint models
  if episode % args.checkpoint_interval == 0:
    planner_state = {k: v for k, v in planner.state_dict().items() if 'model' not in k}
    torch.save({'transition_model': transition_model.state_dict(), 'observation_model': observation_model.state_dict(),
                'reward_model': reward_model.state_dict(), 'cost_model': cost_model.state_dict(), 'encoder': encoder.state_dict(),
                'one_step_ensemble': one_step_ensemble.state_dict(), 'optimiser': optimiser.state_dict(), 'planner': planner_state},
                os.path.join(results_dir, 'models_%d.pth' % episode))
    if args.checkpoint_experience:
      torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()
