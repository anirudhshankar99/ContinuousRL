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
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

import gymnasium as gym
from collections import defaultdict
import warnings


class MultiAgentRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_returns = defaultdict(float)
        self.episode_lengths = defaultdict(int)
        self.episode_counts = defaultdict(int)
        self.contacts = 0
        self.latest_episode_stats = {}

    def reset(self, **kwargs):
        self.episode_returns = defaultdict(float)
        self.episode_lengths = defaultdict(int)
        self.latest_episode_stats = {}
        self.contacts = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rewards, dones, truncated, info = self.env.step(action)
        self.contacts = info['contacts']
        info = {i:info[i] for i in info if info[i] != None}

        for agent, reward in rewards.items():
            self.episode_returns[agent] = reward
            self.episode_lengths[agent] += 1

            if dones[agent]:
                self.episode_counts[agent] += 1
                self.latest_episode_stats[agent] = {
                    'r': self.episode_returns[agent],
                    'l': self.contacts,
                }

        # Add episode statistics to `info` if available
        if self.latest_episode_stats != {}: info['episode'] = self.latest_episode_stats.copy()
        return obs, rewards, dones, truncated, info

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.shape[-1]), std=0.01),
        )
        self.cov_mat = torch.diag(torch.ones(env.action_space.shape[-1], device=device))*0.5
        self.action_scaling = [torch.tensor(env.action_space.low, dtype=torch.float32, device=device), torch.tensor(env.action_space.high, dtype=torch.float32, device=device)]

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_means = self.actor(x)
        dist = MultivariateNormal(action_means, torch.exp(self.cov_mat))
        if action is None:
            action = dist.sample()
        squashed_action = torch.tanh(action)
        scaled_action = self._scale_actions(squashed_action)
        log_prob = dist.log_prob(action) - torch.sum(torch.log(1 - squashed_action.pow(2) + 1e-6), dim=-1)
        return scaled_action, log_prob, dist.entropy(), self.critic(x)
    
    def _scale_actions(self, action): return self.action_scaling[0] + 0.5 * (action + 1.0) * (self.action_scaling[1] - self.action_scaling[0])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='Squash-v0',
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-5,
                        help='the LR of the optimizer(s)')
    parser.add_argument('--seed', type=int, default=1,
                        help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=5,
                        help='order of magnitude of the total timesteps of the experiment, i.e., 10^x')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, cuda will not be enabled when possible')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, the experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help='the wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='the entity of the wandb project')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, records videos of the agent\'s performance')
    parser.add_argument('--num-agents', type=int, default=2,
                        help='number of agents in the environment')
    parser.add_argument('--render-mode', type=str, default='none',
                        help='one of the three ways to render the environment: \'human\', \'logs\' or \'none\'')
    
    # Performance altering
    # parser.add_argument('--num-envs', type=int, default=4,
    #                     help='number of parallel environments trained in')
    parser.add_argument('--num-steps', type=int, default=1024,
                        help='number of steps per environment per rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, LR is not annealed')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, gae will not be computed')
    parser.add_argument('--gamma', type=float, default=0.99,
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
    # args.batch_size = int(args.num_envs * args.num_steps)

    # Environment related
    parser.add_argument('--volley-frac', type=float, default=0.25,
                        help='fraction of the reward awarded for a succesful volley')
    parser.add_argument('--scaling-volley', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, rewards for succesful volley increase linearly')
    parser.add_argument('--fps', type=int, default=30,
                        help='frames per second for all computations/visualizations: a value that is too low might miss collisions')
    
    args = parser.parse_args()
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_envs = 1
    args.total_timesteps = 10**args.total_timesteps
    print('[INFO] Training for %d steps.'%args.total_timesteps)
    return args

def _filter_gym_warnings():
    warnings.filterwarnings("ignore", message="WARN: Box low's precision lowered by casting to float32, current low.dtype=float64")
    warnings.filterwarnings("ignore", message="WARN: Box high's precision lowered by casting to float32, current high.dtype=float64")
    warnings.filterwarnings("ignore", message="WARN: The obs returned by the `reset()` method was expecting a numpy array, actual type: <class 'dict'>")
    warnings.filterwarnings("ignore", message="WARN: Casting input x to numpy array.")
    warnings.filterwarnings("ignore", message="WARN: The obs returned by the `reset()` method is not within the observation space.")
    warnings.filterwarnings("ignore", message="WARN: Expects `terminated` signal to be a boolean, actual type: <class 'dict'>")
    warnings.filterwarnings("ignore", message="WARN: Expects `truncated` signal to be a boolean, actual type: <class 'dict'>")
    warnings.filterwarnings("ignore", message="WARN: The obs returned by the `step()` method was expecting a numpy array, actual type: <class 'dict'>")
    warnings.filterwarnings("ignore", message="WARN: The obs returned by the `step()` method is not within the observation space.")
    warnings.filterwarnings("ignore", message="WARN: The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: <class 'dict'>")

if __name__ == '__main__':
    _filter_gym_warnings()
    args = parse_args()
    run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
    )

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
                'scaling_volley':args.scaling_volley,
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

    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
    #     for i in range(args.num_envs)])
    env = make_env(args.gym_id, args.seed, 0, args.capture_video, run_name)()
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "must be a discrete action space"

    agents, optimizers, obs, actions, logprobs, rewards, dones, values = {}, {}, {}, {}, {}, {}, {}, {}
    agent_ids = ['agent_%d'%i for i in range(args.num_agents)]
    for agent_id in agent_ids:
        agents[agent_id] = Agent(env).to(device)
        optimizers[agent_id] = optim.Adam(agents[agent_id].parameters(), lr=args.learning_rate, eps=1e-5)
        obs[agent_id]=torch.zeros((args.num_steps,) + env.observation_space.shape).to(device)
        actions[agent_id]=torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
        logprobs[agent_id]=torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
        rewards[agent_id]=torch.zeros((args.num_steps,)).to(device)
        dones[agent_id]=torch.zeros((args.num_steps,)).to(device)
        values[agent_id]=torch.zeros((args.num_steps,)).to(device)

    global_step = 0
    start_time = time.time()
    # next_obs = {agent_id: torch.Tensor(obs).to(device) for agent_id, obs in env.reset()[0].items()} # must be handled by envs.reset() (?)
    next_obs = env.reset()[0]

    next_done = {agent_id:torch.tensor(0, dtype=torch.float32) for agent_id in agent_ids}
    num_updates = args.total_timesteps // args.batch_size

    update = 0
    while global_step < args.total_timesteps:
        update += 1
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            for agent_id in agent_ids: optimizers[agent_id].param_groups[0]['lr'] = lrnow 

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            for agent_id in agent_ids:
                obs[agent_id][step] = next_obs[agent_id]
                dones[agent_id][step] = next_done[agent_id]

            action_dict = {}
            for agent_id in agent_ids:
                with torch.no_grad():
                    action, logprob, _, value = agents[agent_id].get_action_and_value(next_obs[agent_id])
                    values[agent_id][step] = value.flatten()
                actions[agent_id][step] = action
                logprobs[agent_id][step] = logprob
                
                action_dict[agent_id] = action.cpu().numpy()

            next_obs, reward, done, _, info = env.step(action_dict)
            for agent_id in agent_ids:
                rewards[agent_id][step] = torch.tensor(reward[agent_id]).to(device).view(-1)
                next_obs[agent_id], next_done[agent_id] = next_obs[agent_id], done[agent_id]

            if 'episode' in info.keys() and all(done.values()):
                for agent_id in agent_ids:
                    print(f'global_step = {global_step}, agent = {agent_id}, episodic_return={info["episode"][agent_id]["r"]}, episodic_length={info["episode"][agent_id]["l"]}')
                    writer.add_scalar('charts/episodic_return/{agent_id}', info["episode"][agent_id]["r"], global_step)
                    writer.add_scalar('charts/episodic_length/{agent_id}', info["episode"][agent_id]["l"], global_step)
                break
        next_obs = env.reset()[0]
        for agent_id in agent_ids:
            with torch.no_grad():
                next_value = agents[agent_id].get_value(next_obs[agent_id]).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards[agent_id]).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done[agent_id]
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[agent_id][t + 1]
                            nextvalues = values[agent_id][t + 1]
                        delta = rewards[agent_id][t] + args.gamma * nextvalues * nextnonterminal - values[agent_id][t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values[agent_id]
                else:
                    returns = torch.zeros_like(rewards[agent_id]).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done[agent_id]
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[agent_id][t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values[agent_id]

            b_obs = obs[agent_id].reshape((-1,) + env.observation_space.shape)
            b_logprobs = logprobs[agent_id].reshape(-1)
            b_actions = actions[agent_id].reshape((-1,) + env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values[agent_id].reshape(-1)
            
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agents[agent_id].get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else: v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[agent_id].parameters(), args.max_grad_norm)
                    optimizers[agent_id].step()
                
                # Need to better handle early stopping as this will only break out of agent loop ->
                # if args.target_kl is not None:
                #     if approx_kl > args.target_kl:
                #         break
        
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar('charts/learning_rate/{agent_id}', optimizers[agent_id].param_groups[0]['lr'], global_step)
            writer.add_scalar('losses/value_loss/{agent_id}', v_loss.item(), global_step)
            writer.add_scalar('losses/policy_loss/{agent_id}', pg_loss.item(), global_step)
            writer.add_scalar('losses/entropy/{agent_id}', entropy_loss.item(), global_step)
            writer.add_scalar('losses/approx_kl/{agent_id}', approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac/{agent_id}", np.mean(clipfracs), global_step)
            writer.add_scalar('losses/explained_variance/{agent_id}', explained_var, global_step)
            writer.add_scalar('charts/SPS', int(global_step/ (time.time() - start_time)), global_step)

    env.close()
    writer.close()
    