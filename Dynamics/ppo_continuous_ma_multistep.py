import numpy as np
import torch
import gymnasium as gym
from torch import nn
from torch.nn import functional as F
import argparse
from torch.utils import tensorboard
from tqdm import tqdm
import os
import random
import time
from dynamics_env.dynamics_ma_multistep import Dynamics
import pandas as pd

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def strtobool(x):
    if x.lower().strip() == 'true': return True
    else: return False

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

# Actor module
class Actor(nn.Module):
    def __init__(self, env, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, env.action_space.shape[0] * 2)
        )
        self.out_shape = env.action_space.shape[0]
    def forward(self, X):
        X = self.model(X)
        (means, log_stds) = torch.split(X, [self.out_shape, self.out_shape], dim=-1)
        return means, log_stds.exp()
    
# Critic module
class Critic(nn.Module):
    def __init__(self, env, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='Dynamics-v0',
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the LR of the optimizer(s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=8e3,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, cuda will be enabled when possible')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, records videos of the agent\'s performance')
    parser.add_argument('--log-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, training will be logged with Tensorboard')
    
    # Performance altering
    parser.add_argument('--num-steps', type=int, default=10,
                        help='number of steps per environment per rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, LR is annealed')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, gae will not be computed')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='the value of the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the value of the lambda parameter for gae')
    parser.add_argument('--num-minibatches', type=int, default=4,
                        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
                        help='number of iterations of policy updates')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, doesn\'t perform advantage normalization')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help='the surrogate ratios\' clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, doesn\'t perform value loss clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='the value of the entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='the coefficient of the value function in the agent\'s loss')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
                        help='if and the threshold kl-d value with which early stopping must be evaluated')
    parser.add_argument('--init-params', type=float, nargs='+',
                        help='initial parameters at each environment reset')
    parser.add_argument('--num-agents', type=int, required=True,
                        help='number of agents in the environment')
    args = parser.parse_args()
    if args.init_params == None:
        args.init_params = np.array([0. for _ in range(6 * args.num_agents)])
    return args    

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage.unsqueeze(-1)
    
    m = torch.min(ratio*advantage.unsqueeze(-1), clipped)

    with torch.no_grad():
        logratio = log_prob - old_log_prob
        # old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
    return -m, approx_kl, clipfracs

if __name__ == '__main__':
    args = parse_args()
    # run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    run_name = f'{args.gym_id}__{args.exp_name}'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    print(f'[AGENT] Seed set to {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')

    def make_env(seed):
        def thunk():
            env = Dynamics(hyperparameters={
                'stationary_potential_list':[],
                'stationary_potential_kwargs_list':[],
                # 'dynamic_potential_list':['point_source','point_source', 'point_source'],
                'dynamic_potential_list':['point_source','point_source'],
                'dynamic_potential_kwargs_list':[{'M':1e4}, {'M':1e4}, {'M':10}],
                'seed':seed,
                'num_steps':args.num_steps,
            })
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    env = make_env(seed=args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "must be a continuous action space"

    actors = {agent:Actor(env, activation=Mish).to(device) for agent in range(args.num_agents)}
    critics = {agent:Critic(env, activation=Mish).to(device) for agent in range(args.num_agents)}
    adam_actors = {agent:torch.optim.Adam(actors[agent].parameters(), lr=3e-4) for agent in range(args.num_agents)}
    adam_critics = {agent:torch.optim.Adam(critics[agent].parameters(), lr=1e-3) for agent in range(args.num_agents)}
    if args.log_train:
        writer = tensorboard.SummaryWriter(f'Dynamics/runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )
    start_time = time.time()
    update = 0
    episodic_reward = 0
    state_list = []
    print(f'[AGENT] Using {device}')
    with tqdm(range(int(args.total_timesteps)), desc=f'episodic_reward: {episodic_reward}') as progress:
        for i in range(int(args.total_timesteps)):
            update += 1
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.total_timesteps
                actor_lrnow = frac * args.learning_rate
                critic_lrnow = frac * args.learning_rate
                for agent in range(args.num_agents):
                    adam_actors[agent].param_groups[0]['lr'] = actor_lrnow
                    adam_critics[agent].param_groups[0]['lr'] = critic_lrnow
            dones = {False:agent for agent in range(args.num_agents)}
            states, _ = env.reset(args.init_params)
            states = {key:torch.tensor(value, dtype=torch.float32).to(device) for key,value in states.items()}

            observations = {agent:torch.zeros((args.num_steps,)+env.observation_space.shape, dtype=torch.float32).to(device) for agent in range(args.num_agents)}
            actions = {agent:torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device) for agent in range(args.num_agents)}
            logprobs = {agent:torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device) for agent in range(args.num_agents)}
            rewards = {agent:torch.zeros((args.num_steps,), dtype=torch.float32).to(device) for agent in range(args.num_agents)}
            dones = {agent:torch.zeros((args.num_steps,), dtype=torch.float32).to(device) for agent in range(args.num_agents)}
            values = {agent:torch.zeros((args.num_steps,), dtype=torch.float32).to(device) for agent in range(args.num_agents)}
            clip_fracs = {agent:[] for agent in range(args.num_agents)}
            j = 0
            while j < args.num_steps:
                # gathering rollout data
                step_actions = {}
                for agent in range(args.num_agents):
                    with torch.no_grad():
                        action_means, action_stds = actors[agent](states[agent])
                    value = critics[agent](states[agent]).flatten()
                    dist = torch.distributions.Normal(action_means, action_stds)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    
                    observations[agent][j] = states[agent]
                    actions[agent][j] = action
                    step_actions[agent] = action.numpy()
                    logprobs[agent][j] = logprob
                    values[agent][j] = value
                states, reward, done, _, info = env.step(step_actions)
                states = {key:torch.tensor(value, dtype=torch.float32).to(device) for key,value in states.items()}
                for agent in range(args.num_agents):
                    rewards[agent][j] = torch.tensor(reward[agent], dtype=torch.float32).to(device)
                    dones[agent][j] = torch.tensor(done[agent], dtype=torch.float32).to(device)
                if np.any(list(done.values())):
                    break
                j += 1
            
            # advantage calculation
            for agent in range(args.num_agents):
                advantages = torch.zeros_like(rewards[agent]).to(device)
                lastgaelam = 0
                done_index = dones[agent].nonzero().max().item() if dones[agent].any() else args.num_steps
                for t in reversed(range(done_index)):
                    # if t == args.num_steps - 1:
                    advantages = lastgaelam = rewards[agent] + (1- dones[agent]) * args.gamma * values[agent] + args.gamma * args.gae_lambda * (1-dones[agent]) * lastgaelam
                    # else:
                    #     advantages[t] = lastgaelam = rewards[agent][t] + (1-dones[agent][t+1]) * args.gamma * values[agent][t+1] - (1-dones[agent][t])*values[agent][t] + args.gamma * args.gae_lambda * (1-dones[agent][t+1]) * lastgaelam  

                action_means, action_stds = actors[agent](observations[agent])
                dist = torch.distributions.Normal(action_means, action_stds)
                new_logprobs = dist.log_prob(actions[agent])
                actor_loss, approx_kl, clipfracs = policy_loss(logprobs[agent], new_logprobs, advantages.detach(), args.clip_coef)
                actor_loss = actor_loss.mean()
                clip_fracs[agent] += clipfracs
                adam_actors[agent].zero_grad()
                actor_loss.backward()
                adam_actors[agent].step()

                critic_loss = advantages.pow(2).mean()
                adam_critics[agent].zero_grad()
                critic_loss.backward()
                adam_critics[agent].step()
                if args.log_train:
                    writer.add_scalar("loss/actor_loss_%d"%agent, actor_loss.detach(), global_step=i)
                    writer.add_scalar("reward/episode_reward_%d"%agent, rewards[agent].sum(dim=0).max().cpu().numpy(), global_step=i)
                    writer.add_scalar("loss/critic_loss %d"%agent, critic_loss.detach(), global_step=i)
                    writer.add_scalar('charts/approx_kl_%d'%agent, approx_kl.item(), global_step=i)
                    writer.add_scalar("charts/clipfrac_%d"%agent, np.mean(clip_fracs[agent]), global_step=i)

            episodic_reward = 0
            for agent in range(args.num_agents):
                episodic_reward += rewards[agent].sum(dim=0).max().cpu().item()
            if args.log_train:
                writer.add_scalar("reward/total_episode_reward", episodic_reward, global_step=i)
            state_list.append([episodic_reward]+info['params'].tolist())
            progress.set_description(f'episodic_reward: {episodic_reward}')
            progress.update()
    state_list = np.array(state_list)
    save_mask = state_list[:,0] > np.max(state_list[:,0]) / 2
    columns = ['reward']
    for agent in range(args.num_agents):
        columns += ['x', 'y', 'z', 'vx', 'vy', 'vz']
    save_df = pd.DataFrame(state_list[save_mask],columns=columns)
    save_df.to_csv(f'Dynamics/runs/{run_name}_best_performers.csv',index=False)