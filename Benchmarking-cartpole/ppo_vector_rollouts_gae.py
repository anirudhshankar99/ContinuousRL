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
    def __init__(self, envs, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, envs.single_action_space.n),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, X):
        return self.model(X)
    
# Critic module
class Critic(nn.Module):
    def __init__(self, envs, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
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
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
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
    parser.add_argument('--num-envs', type=int, default=4,
                        help='number of parallel environments trained in')
    parser.add_argument('--num-steps', type=int, default=2048,
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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args    

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage
    
    m = torch.min(ratio*advantage, clipped)

    with torch.no_grad():
        logratio = log_prob - old_log_prob
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
    return -m, approx_kl, clipfracs

def mask_values_and_rewards_(values, rewards, dones):
    steps, env_indices = torch.tensor_split(dones.nonzero(), 2, dim=1)
    masked_value_indices = torch.zeros_like(values).transpose(0, 1)
    max_done_index = 0
    for i in range(args.num_envs):
        done_index = args.num_steps
        try:
            done_index = steps[torch.nonzero((env_indices - i) == 0)[0][0].item()]
        except: # if the done never happened, let the done index remain at the max episode length
            pass
        max_done_index = done_index if done_index > max_done_index else max_done_index
        value_indices = torch.arange(0, args.num_steps, 1)
        masked_value_indices[i] = value_indices <= done_index
    masked_value_indices = masked_value_indices.transpose(0, 1)
    values *= masked_value_indices
    rewards *= masked_value_indices
    return max_done_index

if __name__ == '__main__':
    args = parse_args()
    # run_name = f'{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}'
    run_name = f'{args.gym_id}__{args.exp_name}'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')

    def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', record_video_trigger=lambda t: t%1000 == 0)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
        for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "must be a discrete action space"

    actor = Actor(envs, activation=Mish).to(device)
    critic = Critic(envs, activation=Mish).to(device)
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
    for i in tqdm(range(int(args.total_timesteps))):
        update += 1
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.total_timesteps
            actor_lrnow = frac * args.learning_rate
            critic_lrnow = frac * args.learning_rate
            adam_actor.param_groups[0]['lr'] = actor_lrnow
            adam_critic.param_groups[0]['lr'] = critic_lrnow
        # prev_logprob = None
        # done = False
        state, _ = envs.reset()
        state = torch.tensor(state).to(device)

        observations = torch.zeros((args.num_steps, args.num_envs,)+envs.single_observation_space.shape, dtype=torch.float32).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs,)+envs.single_action_space.shape, dtype=torch.float32).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs,), dtype=torch.float32).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs,), dtype=torch.float32).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs,), dtype=torch.float32).to(device)
        values = torch.zeros((args.num_steps, args.num_envs,), dtype=torch.float32).to(device)
        clip_fracs = []
        j = 0
        vectorenv_done = np.zeros((args.num_envs,), dtype=bool)
        episode_rewards = [None for _ in range(args.num_envs)]
        episode_lengths = [None for _ in range(args.num_envs)]
        while j < args.num_steps:
            # gathering rollout data
            with torch.no_grad():
                probs = actor(state)
            value = critic(state).flatten()
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            
            observations[j] = state
            actions[j] = action
            logprobs[j] = logprob
            next_state, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards[j] = torch.tensor(reward, dtype=torch.float32).to(device) # nonzero after self done but not others
            dones[j] = torch.tensor(done, dtype=torch.float32).to(device) # nonzero after self done but not others
            values[j] = value
            state = next_state
            state = torch.from_numpy(state).float().to(device)
            # use done.nonzero()
            if done.any():
                for idx in range(len(done)):
                    episode_rewards[idx] = rewards[idx].sum() if episode_rewards[idx] == None and done[idx] else None
                    episode_lengths[idx] = j if episode_lengths[idx] == None and done[idx] else None
            vectorenv_done = vectorenv_done + done
            if vectorenv_done.all():
                if i == args.num_steps // 200:
                    print(episode_rewards)
                    print(f'global_step = {i}, episodic_return={np.max(episode_rewards)}')
                if args.log_train:
                    writer.add_scalar('reward/episodic_return', np.max(episode_rewards), i)
                    writer.add_scalar('reward/episodic_length', np.max(episode_lengths), i)
                break
            j += 1
        
        # advantage calculation
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        
        max_done_index = mask_values_and_rewards_(values, rewards, dones) # inplace masking of values that also returns the max done index among all the environments (just for efficiency)
        for t in reversed(range(max_done_index)):
            if t == args.num_steps - 1:
                advantages[t] = lastgaelam = rewards[t] + (1- dones[t]) * args.gamma * values[t] + args.gamma * args.gae_lambda * (1-dones[t]) * lastgaelam
            else:
                advantages[t] = lastgaelam = rewards[t] + (1-dones[t+1]) * args.gamma * values[t+1] - (1-dones[t])*values[t] + args.gamma * args.gae_lambda * (1-dones[t+1]) * lastgaelam
        
        if args.log_train:
            writer.add_scalar("loss/advantage", advantages.detach().cpu().numpy().mean(), global_step=i)
        
        if args.log_train:
            writer.add_scalar("reward/episode_reward", rewards.sum(dim=0).max().cpu().numpy(), global_step=i)

        probs = actor(observations)
        dist = torch.distributions.Categorical(probs=probs)
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
        state = next_state

        critic_loss = advantages.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()

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
        adam_critic.step()