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
import continuouscartpole_env

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
        # self.log_std = nn.Parameter(torch.ones(env.action_space.shape[0])*0.5)  # Learnable log standard deviation
    
    def forward(self, X):
        X = self.model(X)
        (means, log_stds) = torch.split(X, [1,1], dim=-1)
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
    parser.add_argument('--gym-id', type=str, default='ContinuousCartPole-v0',
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
    args = parser.parse_args()
    return args    

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')

    def make_env(gym_id, seed):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # if capture_video:
            #     if idx == 0:
            #         env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', record_video_trigger=lambda t: t%1000 == 0)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    env = make_env(args.gym_id, seed=args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "must be a continuous action space"

    actor = Actor(env, activation=Mish).to(device)
    critic = Critic(env, activation=Mish).to(device)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    if args.log_train:
        writer = tensorboard.SummaryWriter(f'runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )
    start_time = time.time()
    update = 0
    episodic_reward = 0
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
            state, _ = env.reset()
            state = torch.tensor(state).to(device)

            observations = torch.zeros((args.num_steps,)+env.observation_space.shape, dtype=torch.float32).to(device)
            actions = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            logprobs = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            rewards = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            dones = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            values = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            clip_fracs = []
            j = 0
            while j < args.num_steps:
                # gathering rollout data
                with torch.no_grad():
                    action_means, action_stds = actor(state)
                value = critic(state).flatten()
                dist = torch.distributions.Normal(action_means, action_stds)
                action = dist.sample()
                logprob = dist.log_prob(action)
                
                observations[j] = state
                actions[j] = action
                logprobs[j] = logprob
                state, reward, done, _, info = env.step(action.cpu().numpy())
                rewards[j] = torch.tensor(reward, dtype=torch.float32).to(device) # nonzero after self done but not others
                dones[j] = torch.tensor(done, dtype=torch.float32).to(device) # nonzero after self done but not others
                values[j] = value
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
            returns = advantages.detach() + values.detach()
            
            if args.log_train:
                writer.add_scalar("loss/advantage", advantages.detach().cpu().numpy().mean(), global_step=i)
            
            if args.log_train:
                writer.add_scalar("reward/episode_reward", rewards.sum(dim=0).max().cpu().numpy(), global_step=i)

            for epoch in range(args.update_epochs):
                action_means, action_stds = actor(observations)
                dist = torch.distributions.Normal(action_means.flatten(), action_stds.flatten())
                new_logprobs = dist.log_prob(actions.flatten())
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

                epoch_values = critic(observations)
                critic_loss_unclipped = (epoch_values - returns).pow(2)
                value_clipped = values.detach() + torch.clamp(
                            epoch_values - values.detach(),
                            -args.clip_coef,
                            args.clip_coef
                        )
                critic_loss_clipped = (value_clipped - returns).pow(2)
                critic_loss = torch.max(critic_loss_clipped, critic_loss_unclipped).mean()
                adam_critic.zero_grad()
                critic_loss.backward()
                adam_critic.step()

            # y_pred, y_true = values.cpu().numpy(), rewards.cpu().numpy()
            # var_y = np.var(y_true)
            # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if args.log_train:
                writer.add_scalar("loss/critic_loss", critic_loss.detach(), global_step=i)
                writer.add_histogram("gradients/critic",
                                torch.cat([p.grad.view(-1) for p in critic.parameters()]), global_step=i)
                writer.add_scalar('charts/learning_rate', adam_actor.param_groups[0]['lr'], global_step=i)
                # writer.add_scalar('charts/entropy', entropy_loss.item(), global_step)
                writer.add_scalar('charts/approx_kl', approx_kl.item(), global_step=i)
                writer.add_scalar("charts/clipfrac", np.mean(clip_fracs), global_step=i)
                # writer.add_scalar('losses/explained_variance', explained_var, global_step)
                writer.add_scalar('charts/SPS', int(i/ (time.time() - start_time)), i)
            

            episodic_reward = rewards.sum(dim=0).max().cpu().numpy()
            progress.set_description(f'episodic_reward: {episodic_reward}')
            progress.update()