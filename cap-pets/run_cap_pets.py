import argparse
import os

import torch
import numpy as np
from tqdm import tqdm
import gym
from gym.wrappers import Monitor

from utils import EnvSampler, ReplayMemory
from ccem import ConstrainedCEM
from models import ProbEnsemble


def readParser():
    parser = argparse.ArgumentParser(description='CAP')

    parser.add_argument('--env', default="HalfCheetah-v3")
    parser.add_argument('--algo', default="cem")
    parser.add_argument('--monitor_gym', default=False, action='store_true')
    parser.add_argument('--penalize_uncertainty', action='store_true')
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--binary_cost', action='store_true')
    parser.add_argument('--learn_kappa', action='store_true')
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--cost_constrained', dest='cost_constrained', action='store_true')
    parser.add_argument('--cost_limit', type=float, default=0.,
        help='constraint threshold')
    parser.add_argument('--permissible_cost', type=float, default=0.,
        help='constraint threshold')
    parser.add_argument('--plan_hor', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0, metavar='N',
        help='random seed (default: 0)')
    parser.add_argument('--model_retain_epochs', type=int, default=20, metavar='A',
                    help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=1000, metavar='A',
                    help='frequency of training')
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                    help='steps per epoch')

    parser.add_argument('--num_epoch', type=int, default=100, metavar='A',
                    help='total number of epochs')
    parser.add_argument('--eval_n_episodes', type=int, default=10, metavar='A',
                    help='number of evaluation episodes')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                    help='batch size for training policy')

    parser.add_argument('--hidden_size', type=int, default=200, metavar='A',
                    help='ensemble model hidden dimension')
    parser.add_argument('--cuda', default=True, action="store_true",
                    help='run on CUDA (default: True)')

    args = parser.parse_args()
    if args.permissible_cost < args.cost_limit:
        args.permissible_cost = args.cost_limit
    args.learn_cost = True
    if args.binary_cost:    
        args.c_gamma = 1
    else:
        args.c_gamma = args.gamma
    if not torch.cuda.is_available():
        args.cuda = False
    return args


def train_env_model(args, env_pool, model):
    state, action, reward, next_state, done = env_pool.return_all()

    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)

    reward = np.reshape(reward, (reward.shape[0], -1))
    labels = {
        "state": delta_state,
        "reward": reward[:, :1],
        "cost": reward[:, 1:2],
    }

    model.train(inputs, labels, batch_size=256)

    # Save trained dynamics model
    if args.learn_cost:
        model_path = f'saved_models/{args.env}-ensemble-h{args.hidden_size}.pt'
    else:
        model_path = f'saved_models/{args.env}-ensemble-nocost-h{args.hidden_size}.pt'
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), model_path)

def train(args, env_sampler, env_model, cem_agent, env_pool):
    reward_sum = 0
    total_violation = 0
    environment_step = 0
    learner_update_step = 0
    eps_idx = 0
    env = env_sampler.env

    for epoch_step in tqdm(range(args.num_epoch)):
        # Record agent behaviour
        if args.monitor_gym:
            monitor = Monitor(env, f"videos/{args.run_name}", force=True)
            if epoch_step % 10 == 0:
                env.reset()
                env_sampler.env = monitor
                env_sampler.current_state = None
                monitor.render()

        epoch_rewards = [0]
        epoch_costs = [0]
        epoch_lens = [0]

        for i in range(args.epoch_length):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(cem_agent)
            epoch_rewards[-1] += reward[0]
            epoch_costs[-1] += args.c_gamma ** i * reward[1]
            epoch_lens[-1] += 1

            env_pool.push(cur_state, action, reward, next_state, done)
            
            environment_step += 1

            if done and i != args.epoch_length - 1:
                epoch_rewards.append(0)
                epoch_costs.append(0)
                epoch_lens.append(0)
                eps_idx += 1

            if (i + 1) % args.model_train_freq == 0:
                train_env_model(args, env_pool, env_model)
                if args.algo != "random":
                    cem_agent.set_model(env_model)

        epoch_reward = np.mean(epoch_rewards)
        epoch_cost = np.mean(epoch_costs)
        epoch_len = np.mean(epoch_lens)

        if args.monitor_gym:
            monitor.close()
        env_sampler.env = env

        # Track total number of violations
        if epoch_cost > args.cost_limit:
            total_violation += 1

        print("")
        print(f'Epoch {epoch_step} Reward {epoch_reward:.2f} Cost {epoch_cost:.2f} Total_Violations {total_violation}')

        if args.learn_kappa:
            cem_agent.optimize_kappa(epoch_cost, args.permissible_cost)

def main():
    args = readParser()
    spec = []
    if not args.cost_constrained:
        spec.append('NoConstraint')
    else:
        if args.penalize_uncertainty:
            spec.extend([f'P{args.kappa}', f'T{args.learn_kappa}'])
        if args.learn_kappa:
            spec.append('CAP')
        spec.append(f'C{args.cost_limit}') 

    spec = '-'.join(spec)

    run_name = f"{args.algo}-{spec}-{args.seed}"
    args.run_name = run_name

    print(f"Starting run {run_name}")

    if args.learn_kappa:
        args.penalize_uncertainty = True

    env = gym.make(args.env)
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Ensemble Dynamics Model
    env_model = ProbEnsemble(state_size, action_size, network_size=5, cuda=args.cuda,
                             cost=args.learn_cost, binary_cost=args.binary_cost, hidden_size=args.hidden_size)
    if args.cuda:
        env_model.to('cuda')

    # CEM Agent
    cem_agent = ConstrainedCEM(env,
                               plan_hor=args.plan_hor,
                               gamma=args.gamma,
                               cost_limit=args.cost_limit,
                               cost_constrained=args.cost_constrained,
                               penalize_uncertainty=args.penalize_uncertainty,
                               learn_kappa=args.learn_kappa,
                               kappa=args.kappa,
                               binary_cost=args.binary_cost,
                               cuda=args.cuda,
                               )

    # Sampler Environment
    env_sampler = EnvSampler(env, max_path_length=args.epoch_length)

    # Experience Buffer
    env_pool = ReplayMemory(args.epoch_length * args.num_epoch)

    # Train
    train(args, env_sampler, env_model, cem_agent, env_pool)


if __name__ == '__main__':
    main()