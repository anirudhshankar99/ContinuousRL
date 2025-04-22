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
from dynamics_env.dynamics_env import Dynamics

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def strtobool(x):
    if x.lower().strip() == 'true': return True
    else: return False

def mish(input):
    return input * torch.tanh(F.softplus(input))

class RecordEpisodeStatistics():
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_envs = 1

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations = self.env.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            terminateds = [terminateds]
            truncateds = [truncateds]
        terminateds = list(terminateds)
        truncateds = list(truncateds)

        for i in range(len(terminateds)):
            if terminateds[i] or truncateds[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "episode": {
                        "r": episode_return,
                        "l": episode_length,
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                }
                infos = {**infos, **episode_info}
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            terminateds if self.is_vector_env else terminateds[0],
            truncateds if self.is_vector_env else truncateds[0],
            infos,
        )

    def reset_2(self, **kwargs):
        observations = super().reset(**kwargs)
        self.orbit_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.orbit_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step_2(self, action):
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step_2(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.orbit_returns += rewards
        self.orbit_lengths += 1

        terminateds = [terminateds]
        truncateds = [truncateds]
        terminateds = list(terminateds)
        truncateds = list(truncateds)

        for i in range(len(terminateds)):
            if terminateds[i] or truncateds[i]:
                orbit_return = self.orbit_return[i]
                orbit_length = self.orbit_length[i]
                episode_info = {
                    "orbit": {
                        "r": orbit_return,
                        "l": orbit_length,
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                }
                infos = {**infos, **episode_info}
                self.return_queue.append(orbit_return)
                self.length_queue.append(orbit_length)
                self.orbit_count += 1
                self.orbit_return[i] = 0
                self.orbit_length[i] = 0
        return (
            observations,
            rewards,
            terminateds[0],
            truncateds[0],
            infos,
        )

class DynamicsModel(torch.nn.Module):
    def __init__(self, env, hidden_dim=64):
        super().__init__()
        input_dim = env.observation_space.shape[0] + env.action_space.shape[0]
        output_dim = env.observation_space.shape[0]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim * 2)
        )
        self.output_dim = output_dim
        self.trajectories = []
        self.rtgs = []
        self.stds = []
        self.confidence_network = torch.nn.Sequential(
            torch.nn.Linear((output_dim + 1) * args.lookahead_options, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, args.lookahead_options + 1),
        )
        self.softmax = torch.nn.Softmax(dim=0)
        
    def forward(self, x, y):
        x = self.model(x)
        means, logstds = torch.split(x, [self.output_dim, self.output_dim], dim=-1)
        loss = (((y - means) ** 2) / (2 * torch.exp(logstds) ** 2)) + logstds
        return loss.mean()
    
    def get_prediction(self, x):
        x = self.model(x)
        means, logstds = torch.split(x, [self.output_dim, self.output_dim], dim=-1)
        return means, logstds
    
    def reset(self):
        self.trajectories = []
        self.rtgs = []
        self.stds = []

    def select_action(self):
        rtgs = torch.stack(self.rtgs)
        stds = torch.stack(self.stds)
        x = torch.cat([rtgs, stds], dim=-1)
        scores = self.confidence_network(x.flatten())
        probabilities = self.softmax(scores)
        _, most_probable_action = torch.max(probabilities), torch.argmax(probabilities)
        return (most_probable_action, 0) if most_probable_action < args.lookahead_options else (np.random.randint(low=0, high=args.lookahead_options), 1)
    
    def confidence_forward(self, episode_rtgs, episode_stds):
        x = torch.cat([episode_rtgs.unsqueeze(-1), episode_stds], dim=-1)
        x = x.flatten(start_dim=1, end_dim=2)
        episode_scores = self.confidence_network(x)
        episode_preferred_actions, _ = torch.max(episode_scores, dim=-1) # (n_steps,)
        return episode_preferred_actions.flatten()

class Memory():
    def __init__(self, env):
        self.source_state_action = None
        self.dest_state = None
        self.rtgs = None
        self._observations_shape = env.observation_space.shape[0]
        self._actions_shape = env.action_space.shape[0]

    def get_mapped_pairs(self):
        return self.source_state_action, self.dest_state

class BestMemory(Memory):
    def __init__(self, env):
        super().__init__(env)

    def add(self, observations, actions, rtgs):
        if self.rtgs == None:
            self.source_state_action = torch.cat([observations[:-1], actions[:-1]],dim=1)
            self.dest_state = observations[1:]
            self.rtgs = rtgs[:-1]
            self.rtgs, sorted_indices = torch.sort(self.rtgs, descending=True)
            self.source_state_action = self.source_state_action[sorted_indices]
            self.dest_state = self.dest_state[sorted_indices]
            return
        
        # forget
        forget_mask = torch.ones_like(self.rtgs) * (torch.rand(self.rtgs.shape).to(device) > 0.01)
        self.rtgs *= forget_mask
        self.rtgs, sorted_indices = torch.sort(self.rtgs, descending=True)
        self.source_state_action = self.source_state_action[sorted_indices]
        self.dest_state = self.dest_state[sorted_indices]

        # already descending sorted
        source_state_action = torch.cat([observations[:-1], actions[:-1]],dim=1)
        dest_state = observations[1:]
        rtgs = rtgs[:-1]
        rtgs, sorted_indices = torch.sort(rtgs, descending=True)
        source_state_action = source_state_action[sorted_indices]
        dest_state = dest_state[sorted_indices]
        # merge sorting
        source_list_1_indices, source_list_2_indices = self.merge_sort(self.rtgs, rtgs)
        merged_source_state_action = torch.zeros(len(source_state_action)+len(self.source_state_action),self._observations_shape+self._actions_shape).to(device)
        merged_source_state_action[source_list_1_indices] = self.source_state_action
        merged_source_state_action[source_list_2_indices] = source_state_action

        merged_dest_state = torch.zeros(len(dest_state)+len(self.dest_state),self._observations_shape).to(device)
        merged_dest_state[source_list_1_indices] = self.dest_state
        merged_dest_state[source_list_2_indices] = dest_state

        merged_rtgs = torch.zeros(len(rtgs)+len(self.rtgs)).to(device)
        merged_rtgs[source_list_1_indices] = self.rtgs
        merged_rtgs[source_list_2_indices] = rtgs

        self.source_state_action = merged_source_state_action[:args.memory_size]
        self.dest_state = merged_dest_state[:args.memory_size]
        self.rtgs = merged_rtgs[:args.memory_size]

    def merge_sort(self, sorted_list_1, sorted_list_2):
        source_list_1_indices = []
        source_list_2_indices = []
        i, j, k = 0, 0, 0
        while(i < len(sorted_list_1) and j < len(sorted_list_2)):
            if sorted_list_1[i] > sorted_list_2[j]:
                source_list_1_indices.append(k)
                i += 1
                k += 1
            else:
                source_list_2_indices.append(k)
                j += 1
                k += 1
        while(i < len(sorted_list_1)):
            source_list_1_indices.append(k)
            i += 1
            k += 1
        while(j < len(sorted_list_2)):
            source_list_2_indices.append(k)
            j += 1
            k += 1
        return source_list_1_indices, source_list_2_indices
    
class RecentMemory(Memory):
    def __init__(self, env):
        super().__init__(env)

    def add(self, observations, actions):
        if self.source_state_action == None:
            self.source_state_action = torch.cat([observations[:-1], actions[:-1]],dim=1)
            self.dest_state = observations[1:]
            return
        source_state_action = torch.cat([observations[:-1], actions[:-1]],dim=1)
        dest_state = observations[1:]
        self.source_state_action = torch.cat([self.source_state_action, source_state_action], dim=0)
        self.dest_state = torch.cat([self.dest_state, dest_state], dim=0)
        if len(self.dest_state) > args.memory_size:
            self.dest_state = self.dest_state[len(self.dest_state) - args.memory_size:]
            self.source_state_action = self.source_state_action[len(self.source_state_action) - args.memory_size:]

class MemoryMixer():
    def __init__(self, env):
        self.best_memory = BestMemory(env)
        self.recent_memory = RecentMemory(env)
    
    def add(self, observations, actions, rtgs):
        self.best_memory.add(observations, actions, rtgs)
        self.recent_memory.add(observations, actions)
    
    def get_mapped_pairs(self):
        best_sa, best_d = self.best_memory.get_mapped_pairs()
        recent_sa, recent_d = self.recent_memory.get_mapped_pairs()
        return torch.cat([best_sa, recent_sa], dim=0), torch.cat([best_d, recent_d], dim=0)

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

# Actor module, categorical actions only
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
        self.split_shape = env.action_space.shape[0]
    def forward(self, X):
        X = self.model(X)
        (means, log_stds) = torch.split(X, [self.split_shape, self.split_shape], dim=-1)
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
                        help='if False, cuda will not be enabled when possible')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, records videos of the agent\'s performance')
    parser.add_argument('--log-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, training will be logged with Tensorboard')
    
    # Performance altering
    parser.add_argument('--num-steps', type=int, default=2048,
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
    parser.add_argument('--lookahead-steps', type=int, default=4,
                        help='number of steps looked ahead')
    parser.add_argument('--lookahead-options', type=int, default=4,
                        help='number of options looked ahead')
    parser.add_argument('--memory-size', type=int, default=1e4,
                        help='number of states stored in memory')
    parser.add_argument('--mlp-epochs', type=int, default=10,
                        help='number of epochs the MLP is trained for')
    parser.add_argument('--init-params', type=float, nargs='+', default=None,
                        help='initial (phase) parameters of the orbit')
    args = parser.parse_args()
    args.memory_size = int(args.memory_size)
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

def compute_rtgs(rewards, done_index):
    rewards_to_go = torch.zeros_like(rewards)
    for t in reversed(range(done_index)):
        if t == len(rewards) - 1:
            rewards_to_go[t] = rewards[t]
        else:
            rewards_to_go[t] = rewards_to_go[t+1] * args.gamma + rewards[t]
    # rtgs = torch.ceil(rewards_to_go)
    return rewards_to_go

if __name__ == '__main__':
    args = parse_args()
    # run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')

    def make_env(gym_id, seed):
        def thunk():
            # env = gym.make(gym_id, hyperparameters={
            #     'galaxy_model_kwargs_list':[{'M':10}]
            # })
            env = Dynamics(hyperparameters={
                'galaxy_model_kwargs_list':[{'M':10}],
                'seed':seed,
            })
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    env = make_env(args.gym_id, seed=args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "must be a continuous action space"

    actor = Actor(env, activation=Mish).to(device)
    critic = Critic(env, activation=Mish).to(device)
    theory = DynamicsModel(env).to(device)
    memory = MemoryMixer(env)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    adam_theory = torch.optim.Adam(theory.model.parameters(), lr=1e-4)
    adam_confidence = torch.optim.Adam(theory.confidence_network.parameters(), lr=5e-5)

    init_params = [0.48813504, 2.15189366, 1.02763376, 0.1, 0, 0]
    # init_params = [0.48813504, 2.15189366, 1.02763376, 2.69299098, 4.58071204, 8.75364678]
    if args.log_train:
        writer = tensorboard.SummaryWriter(f'Dynamics/runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )
        writer.add_text(
            'init_params',
            '|x|y|z|vx|vy|vz|\n|-|-|-|-|-|-|\n%s'%('\n'.join('%s'%i for i in init_params))
        )
    start_time = time.time()
    update = 0
    episodic_reward = 0
    best_episodic_reward = -np.inf
    best_weights = {'actor':None, 'critic':None, 'theory':None}
    env.reset(init_params=init_params)
    with tqdm(range(int(args.total_timesteps)), desc=f'episodic_reward: {episodic_reward}') as progress:
        for i in range(int(args.total_timesteps)):
            update += 1
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.total_timesteps
                actor_lrnow = frac * args.learning_rate
                critic_lrnow = frac * args.learning_rate
                adam_actor.param_groups[0]['lr'] = actor_lrnow
                adam_critic.param_groups[0]['lr'] = critic_lrnow
            prev_logprob = None
            done = False
            state, _ = env.reset_2()
            state = torch.tensor(state, dtype=torch.float32).to(device)

            observations = torch.zeros((args.num_steps,)+env.observation_space.shape, dtype=torch.float32).to(device)
            actions = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            logprobs = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            rewards = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            dynamics_choices = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            dones = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            values = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            episode_rtgs = torch.zeros((args.num_steps, args.lookahead_options), dtype=torch.float32).to(device) # all options not documented
            episode_stds = torch.zeros((args.num_steps, args.lookahead_options)+env.observation_space.shape, dtype=torch.float32).to(device) # all options not documented
            clip_fracs = []
            random_action_fracs = []
            j = 0
            while j < args.num_steps:
                # gathering rollout data
                with torch.no_grad():
                    action_means, action_stds = actor(state)
                value = critic(state).flatten()
                dist = torch.distributions.Normal(action_means, action_stds)

                # lookahead sequence:
                theory.reset()
                for l_o in range(args.lookahead_options):
                    for l_i in range(args.lookahead_steps):
                        if l_i == 0:
                            lookahead_action = dist.sample()
                            with torch.no_grad():
                                projected_state, previous_stds = theory.get_prediction(torch.cat([state, lookahead_action]))
                                previous_discounted_reward = critic(projected_state) - critic(state)
                            lookahead_actions = [lookahead_action]
                        else:
                            with torch.no_grad():
                                action_means, action_stds = actor(projected_state)
                            dist_ = torch.distributions.Normal(action_means, action_stds)
                            lookahead_action = dist_.sample()
                            with torch.no_grad():
                                projected_state, stds = theory.get_prediction(torch.cat([projected_state, lookahead_action]))
                                projected_discounted_reward = (args.gamma ** l_i) * (critic(projected_state) - critic(state))
                            previous_discounted_reward += projected_discounted_reward
                            previous_stds += stds
                            lookahead_actions.append(lookahead_action)
                    theory.trajectories.append(lookahead_actions)
                    theory.rtgs.append(previous_discounted_reward)
                    theory.stds.append(previous_stds)

                with torch.no_grad():
                    selected_action_index, random_selection_or_not = theory.select_action()
                action = theory.trajectories[selected_action_index][0]
                episode_rtgs[j] = torch.cat(theory.rtgs)
                episode_stds[j] = torch.stack(theory.stds)

                logprob = dist.log_prob(action)
                observations[j] = state
                actions[j] = action
                logprobs[j] = logprob
                state, reward, done, _, info = env.step_2(action.cpu().numpy())
                rewards[j] = torch.tensor(reward, dtype=torch.float32).to(device) # nonzero after self done but not others
                dones[j] = torch.tensor(done, dtype=torch.float32).to(device) # nonzero after self done but not others
                values[j] = value
                random_action_fracs.append(random_selection_or_not)
                state = torch.from_numpy(state).float().to(device)
                if done:
                    break
                j += 1
            
            # advantage calculation
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            done_index = dones.nonzero().max().item() if dones.any() else args.num_steps
            for t in reversed(range(done_index)):
                if t == args.num_steps - 1:
                    advantages[t] = lastgaelam = rewards[t] + (1- dones[t]) * args.gamma * values[t] + args.gamma * args.gae_lambda * (1-dones[t]) * lastgaelam
                else:
                    advantages[t] = lastgaelam = rewards[t] + (1-dones[t+1]) * args.gamma * values[t+1] - (1-dones[t])*values[t] + args.gamma * args.gae_lambda * (1-dones[t+1]) * lastgaelam
            
            if args.log_train:
                writer.add_scalar("loss/advantage", advantages.detach().cpu().numpy().mean(), global_step=i)
                writer.add_scalar("reward/episode_reward", rewards.sum(dim=0).max().cpu().numpy(), global_step=i)
                writer.add_scalar("reward/episode_length", done_index, global_step=i)

            action_means, action_stds = actor(observations)
            dist = torch.distributions.Normal(action_means, action_stds)
            new_logprobs = dist.log_prob(actions)
            actor_loss, approx_kl, clipfracs = policy_loss(logprobs, new_logprobs, advantages.detach(), args.clip_coef)
            actor_loss = actor_loss.mean()
            clip_fracs += clipfracs
            adam_actor.zero_grad()
            actor_loss.backward()
            if args.log_train:
                writer.add_scalar("loss/actor_loss", actor_loss.detach(), global_step=i)
                writer.add_histogram("gradients/actor",
                                torch.cat([p.grad.view(-1) for p in actor.parameters()]), global_step=i)
            adam_actor.step()

            critic_loss = advantages.pow(2).mean()
            adam_critic.zero_grad()
            critic_loss.backward()

            rtgs = compute_rtgs(rewards, done_index)
            memory.add(observations[:done_index], actions[:done_index], rtgs[:done_index])
            source_state_actions, dest_states = memory.get_mapped_pairs()
            for epoch in range(args.mlp_epochs):
                theory_loss = theory(source_state_actions, dest_states)
                adam_theory.zero_grad()
                theory_loss.backward()
                adam_theory.step()

            confidence_network_choices = theory.confidence_forward(episode_rtgs[:done_index], episode_stds[:done_index])
            confidence_network_choices = (confidence_network_choices / confidence_network_choices.detach() * advantages[:done_index].detach()).mean()
            adam_confidence.zero_grad()
            confidence_network_choices.backward()
            adam_confidence.step()

            if args.log_train:
                writer.add_scalar("loss/critic_loss", critic_loss.detach(), global_step=i)
                writer.add_scalar("loss/theory_loss", theory_loss.detach().item(), global_step=i)
                writer.add_histogram("gradients/critic",
                                torch.cat([p.grad.view(-1) for p in critic.parameters()]), global_step=i)
                writer.add_scalar('charts/learning_rate', adam_actor.param_groups[0]['lr'], global_step=i)
                writer.add_scalar('charts/random_action_frac', np.mean(random_action_fracs), global_step=i)
                writer.add_scalar('charts/approx_kl', approx_kl.item(), global_step=i)
                writer.add_scalar("charts/clipfrac", np.mean(clip_fracs), global_step=i)
                writer.add_scalar('charts/SPS', int(i/ (time.time() - start_time)), i)
            adam_critic.step()

            episodic_reward = rewards.sum(dim=0).max().cpu().numpy()
            if episodic_reward > best_episodic_reward:
                best_weights['actor'] = actor.state_dict()
                best_weights['critic'] = critic.state_dict()
                best_weights['theory'] = theory.state_dict()
                best_episodic_reward = episodic_reward
            progress.set_description(f'episodic_reward: {episodic_reward}')
            progress.update()
    try:
        os.mkdir(f'Dynamics/runs/{run_name}_weights/')
    except FileExistsError:
        pass
    torch.save(best_weights['actor'], f'Dynamics/runs/{run_name}_weights/actor_{args.seed}.pth')
    torch.save(best_weights['critic'], f'Dynamics/runs/{run_name}_weights/critic_{args.seed}.pth')
    torch.save(best_weights['theory'], f'Dynamics/runs/{run_name}_weights/theory_{args.seed}.pth')