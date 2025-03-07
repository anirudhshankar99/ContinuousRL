import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import squash_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
        # self.log_std = nn.Parameter(torch.ones(env.action_space.shape[0])*0.5)  # Learnable log standard deviation
    
    def forward(self, X):
        X = self.model(X)
        (means, log_stds) = torch.split(X, [X.shape[-1] // 2, X.shape[-1] // 2], dim=-1)
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

class MultiAgentRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_returns = {}
        # self.episode_lengths = {}
        self.episode_counts = {}
        # self.contacts = 0
        self.latest_episode_stats = {}

    def reset(self, **kwargs):
        self.episode_returns = {}
        # self.episode_lengths = {}
        self.latest_episode_stats = {}
        # self.contacts = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rewards, dones, truncated, info = self.env.step(action)
        # self.contacts = info['contacts']
        info = {i:info[i] for i in info if info[i] != None}

        for agent, reward in rewards.items():
            self.episode_returns[agent] = reward
            # self.episode_lengths[agent] += 1

            if dones[agent]:
                # self.episode_counts[agent] += 1
                self.latest_episode_stats[agent] = {
                    'r': self.episode_returns[agent],
                    # 'l': self.contacts,
                }

        # Add episode statistics to `info` if available
        if self.latest_episode_stats != {}: info['episode'] = self.latest_episode_stats.copy()
        return obs, rewards, dones, truncated, info

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
        # self.log_std = nn.Parameter(torch.ones(env.action_space.shape[0])*0.5)  # Learnable log standard deviation
    
    def forward(self, X):
        X = self.model(X)
        (means, log_stds) = torch.split(X, [X.shape[-1] // 2, X.shape[-1] // 2], dim=-1)
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
    parser.add_argument('--gym-id', type=str, default='Squash-v0',
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the LR of the optimizer(s)')
    parser.add_argument('--seed', type=int, default=1,
                        help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=8e3,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, cuda will not be enabled when possible')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, records videos of the agent\'s performance')
    parser.add_argument('--log-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, training will be logged with Tensorboard')
    parser.add_argument('--render-mode', type=str, default='none',
                        help='one of the three ways to render the environment: \'human\', \'logs\' or \'none\'')
    
    # Performance altering
    parser.add_argument('--num-steps', type=int, default=1024,
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

    # Environment related
    parser.add_argument('--volley-frac', type=float, default=0.25,
                        help='fraction of the reward awarded for a succesful volley')
    parser.add_argument('--fps', type=int, default=30,
                        help='frames per second for all computations/visualizations: a value that is too low might miss collisions')
    
    args = parser.parse_args()
    if args.render_mode == 'demo':
        args.render_mode = 'none'
        args.demo = True
    print('[INFO] Training for %d steps.'%args.total_timesteps)
    return args

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    if ratio.shape != advantage.shape:
        advantage = advantage.unsqueeze(-1).repeat(repeats=(1,ratio.shape[-1]))
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage
    
    m = torch.min(ratio*advantage, clipped)

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
    if args.log_train:
        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )

    num_agents = 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')

    def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk():
            env = gym.make(gym_id, hyperparameters={
                'render_mode':'"%s"'%args.render_mode,
                'volley_reward_fraction':args.volley_frac,
                'fps':args.fps
                })
            env = MultiAgentRecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', record_video_trigger=lambda t: t%1000 == 0)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    env = make_env(args.gym_id, args.seed, 0, args.capture_video, run_name)()
    assert isinstance(env.action_space, gym.spaces.Box), "must be a continuous action space"

    actors, critics, optimizers_a, optimizers_c, observations, actions, logprobs, rewards, dones, values = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    agent_ids = ['agent_%d'%i for i in range(num_agents)]
    for agent_id in agent_ids:
        actors[agent_id] = Actor(env).to(device)
        critics[agent_id] = Critic(env).to(device)
        optimizers_a[agent_id] = optim.Adam(actors[agent_id].parameters(), lr=args.learning_rate, eps=3e-4)
        optimizers_c[agent_id] = optim.Adam(critics[agent_id].parameters(), lr=args.learning_rate, eps=1e-3)

    episodic_reward = 0
    update = 0
    start_time = time.time()
    with tqdm(range(int(args.total_timesteps)), desc=f'episodic_reward: {episodic_reward}') as progress:
        for i in range(int(args.total_timesteps)):
            update += 1
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.total_timesteps
                actor_lrnow = frac * args.learning_rate
                critic_lrnow = frac * args.learning_rate
                for agent_id in agent_ids:
                    optimizers_a[agent_id].param_groups[0]['lr'] = actor_lrnow
                    optimizers_c[agent_id].param_groups[0]['lr'] = critic_lrnow
            prev_logprob = {agent_id:None for agent_id in agent_ids}
            done = {False:None for agent_id in agent_ids}
            state, _ = env.reset()

            for agent_id in agent_ids:
                observations[agent_id]=torch.zeros((args.num_steps,) + env.observation_space.shape).to(device)
                actions[agent_id]=torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
                logprobs[agent_id]=torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
                rewards[agent_id]=torch.zeros((args.num_steps,)).to(device)
                dones[agent_id]=torch.zeros((args.num_steps,)).to(device)
                values[agent_id]=torch.zeros((args.num_steps,)).to(device)
                # state[agent_id] = torch.tensor(state[agent_id], dtype=torch.float32).to(device)
            clip_fracs = {agent_id:[] for agent_id in agent_ids}
            j = 0
            
            while j < args.num_steps:
                # gathering rollout data

                action_dict = {}
                for agent_id in agent_ids:
                    with torch.no_grad():
                        action_means, action_stds = actors[agent_id](state[agent_id])
                    value = critics[agent_id](state[agent_id]).flatten()
                    try:
                        dist = torch.distributions.Normal(action_means, action_stds)
                    except ValueError: print(i,j,action_means, action_stds, state[agent_id]) 
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    
                    observations[agent_id][j] = state[agent_id]
                    actions[agent_id][j] = action
                    logprobs[agent_id][j] = logprob
                    values[agent_id][j] = value
                    action_dict[agent_id] = action.cpu().numpy()
                    
                state, reward, done, _, info = env.step(action_dict)

                for agent_id in agent_ids:
                    rewards[agent_id][j] = torch.tensor(reward[agent_id], dtype=torch.float32).to(device) # nonzero after self done but not others
                    dones[agent_id][j] = torch.tensor(done[agent_id], dtype=torch.float32).to(device) # nonzero after self done but not others
                    # state[agent_id] = torch.from_numpy(state[agent_id]).float().to(device)

                if np.array(list(done.values())).all():
                    break
                j += 1

            # advantage calculation
            advantages = {}
            for agent_id in agent_ids:
                advantages[agent_id] = torch.zeros_like(rewards[agent_id]).to(device)
                lastgaelam = 0
                done_index = dones[agent_id].nonzero().max().item() if dones[agent_id].any() else args.num_steps
                for t in reversed(range(done_index)):
                    if t == args.num_steps - 1:
                        advantages[agent_id][t] = lastgaelam = rewards[agent_id][t] + (1-dones[agent_id][t]) * args.gamma * values[agent_id][t] + args.gamma * args.gae_lambda * (1-dones[agent_id][t]) * lastgaelam
                    else:
                        advantages[agent_id][t] = lastgaelam = rewards[agent_id][t] + (1-dones[agent_id][t+1]) * args.gamma * values[agent_id][t+1] - (1-dones[agent_id][t])*values[agent_id][t] + args.gamma * args.gae_lambda * (1-dones[agent_id][t+1]) * lastgaelam

            if args.log_train:
                for agent_id in agent_ids:
                    writer.add_scalar(f"loss/advantage/{agent_id}", advantages[agent_id].detach().cpu().numpy().mean(), global_step=i)
                    writer.add_scalar(f"reward/episode_reward/{agent_id}", rewards[agent_id].sum(dim=0).max().cpu().numpy(), global_step=i)

            approx_kls = {}
            for agent_id in agent_ids:
                action_means, action_stds = actors[agent_id](observations[agent_id])
                dist = torch.distributions.Normal(action_means, action_stds)
                new_logprobs = dist.log_prob(actions[agent_id])
                actor_loss, approx_kls[agent_id], clipfrac = policy_loss(logprobs[agent_id], new_logprobs, advantages[agent_id].detach(), args.clip_coef)
                actor_loss = actor_loss.mean()
                clip_fracs[agent_id] += clipfrac
                optimizers_a[agent_id].zero_grad()
                actor_loss.backward()
                optimizers_a[agent_id].step()

                critic_loss = advantages[agent_id].pow(2).mean()
                optimizers_c[agent_id].zero_grad()
                critic_loss.backward()
                optimizers_c[agent_id].step()

            if args.log_train:
                for agent_id in agent_ids:
                    writer.add_scalar(f"loss/actor_loss/{agent_id}", actor_loss.detach(), global_step=i)
                    writer.add_histogram(f"gradients/actor/{agent_id}",
                                    torch.cat([p.grad.view(-1) for p in actors[agent_id].parameters()]), global_step=i)
                    writer.add_scalar(f"loss/critic_loss/{agent_id}", critic_loss.detach(), global_step=i)
                    writer.add_histogram("gradients/critic",
                                    torch.cat([p.grad.view(-1) for p in critics[agent_id].parameters()]), global_step=i)
                    # writer.add_scalar('charts/learning_rate', optimizers_a[agent_id].param_groups[0]['lr'], global_step=i)
                    # writer.add_scalar('charts/entropy', entropy_loss.item(), global_step)
                    writer.add_scalar(f'charts/approx_kl/{agent_id}', approx_kls[agent_id].item(), global_step=i)
                    writer.add_scalar(f"charts/clipfrac/{agent_id}", np.mean(clip_fracs[agent_id]), global_step=i)
                    writer.add_scalar('charts/SPS', int(i/ (time.time() - start_time)), i)

            episodic_reward = torch.stack(list(rewards.values())).sum(dim=-1).mean().cpu().numpy()
            progress.set_description('episodic_reward: %2.2f'%episodic_reward)
            progress.update()

    env.close()
    if args.log_train:
        writer.close()
    while args.demo:
        prompt_to_start_demo = input("Enter anything to start: ")
        args.render_mode = 'human'
        env = make_env(args.gym_id, np.random.randint(0, 42), 0, args.capture_video, run_name)()
        done = {False:None for agent_id in agent_ids}
        state, _ = env.reset()
        while j < args.num_steps:
            # rolling out
            action_dict = {}
            for agent_id in agent_ids:
                with torch.no_grad():
                    action_means, action_stds = actors[agent_id](state[agent_id])
                try:
                    dist = torch.distributions.Normal(action_means, action_stds)
                except ValueError: print(i,j,action_means, action_stds, state[agent_id]) 
                action = dist.sample()
                action_dict[agent_id] = action.cpu().numpy()
                
            state, reward, done, _, info = env.step(action_dict)

            if np.array(list(done.values())).all():
                break
            j += 1
    