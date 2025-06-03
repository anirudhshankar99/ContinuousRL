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
from dynamics_env.dynamics_adversarial import Dynamics, Orbit
import transformer
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
    parser.add_argument('--num-steps', type=int, default=1,
                        help='number of steps per environment per rollout')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, gae will not be computed')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='the value of the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the value of the lambda parameter for gae')
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
    parser.add_argument('--num-bodies', type=int, required=True,
                        help='number of bodies tracked/integrated in the environment')
    parser.add_argument('--box-scaling', type=float, default=1.,
                        help='default box size is 10 pc, this is a scaling on that size')
    parser.add_argument('--n-predictions', type=int, default=10,
                        help='number of predictions made by the predictor')
    parser.add_argument('--orbit-timesteps', type=int, default=1000,
                        help='number of timesteps integrated')
    parser.add_argument('--orbit-duration', type=float, default=1000.,
                        help='number of Myrs integrated')
    args = parser.parse_args()
    if args.init_params == None:
        args.init_params = np.array([0. for _ in range(6 * args.num_bodies)])
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
                'stationary_potential_list':['bar'],
                'stationary_potential_kwargs_list':[{'M':1e10, 'a':5000, 'b':1500, 'c':1000, 'omega_p':0.0}],
                'dynamic_potential_list':['tracer'],
                'dynamic_potential_kwargs_list':[{'M':1e10}],
                'seed':seed,
                'box_scaling':args.box_scaling,
                'orbit_duration':args.orbit_duration,
                'orbit_timesteps':args.orbit_timesteps,
            })
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    env = make_env(seed=args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "must be a continuous action space"

    actor_proposer = Actor(env, activation=Mish).to(device)
    critic_proposer = Critic(env, activation=Mish).to(device)
    optim_actor_proposer = torch.optim.Adam(actor_proposer.parameters(), lr=3e-4)
    optim_critic_proposer = torch.optim.Adam(critic_proposer.parameters(), lr=1e-3)

    agent_predictor = transformer.Agent(input_dim=6 * args.num_bodies,
                                        output_dim=2 * 3 * args.num_bodies,
                                        condition_dim=6,
                                        hidden_dim=64,
                                        n_heads=4,
                                        num_layers=3,
                                        seq_len=args.n_predictions)
    optim_agent_predictor = torch.optim.Adam(agent_predictor.parameters(), lr=3e-4)
    if args.log_train:
        writer = tensorboard.SummaryWriter(f'Dynamics/runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )
    start_time = time.time()
    episodic_reward = 0
    state_list = []
    print(f'[AGENT] Using {device}')
    with tqdm(range(int(args.total_timesteps)), desc=f'episodic_reward: {episodic_reward}') as progress:
        for i in range(int(args.total_timesteps)):
            dones = False
            state, info = env.reset(args.init_params)
            conditioning = info['model_type_list']
            state = torch.tensor(state, dtype=torch.float32, device=device)

            observations_proposer = torch.zeros((args.num_steps,)+env.observation_space.shape, dtype=torch.float32).to(device)
            observations_predictor = Orbit(None, None, None)
            actions_proposer = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            actions_predictor = torch.zeros((args.num_steps, args.n_predictions, 3 * args.num_bodies), dtype=torch.float32).to(device)
            logprobs_proposer = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            logprobs_predictor = torch.zeros((args.num_steps, args.n_predictions, 3 * args.num_bodies), dtype=torch.float32).to(device)
            rewards_proposer = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            rewards_predictor = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            dones = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            values_proposer = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            values_predictor = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            clip_fracs_proposer = []
            clip_fracs_predictor = []
            j = 0
            while j < args.num_steps:
                # gathering rollout data
                with torch.no_grad():
                    action_means, action_stds = actor_proposer(state)
                value = critic_proposer(state)
                dist = torch.distributions.Normal(action_means, action_stds)
                action = dist.sample()
                logprob = dist.log_prob(action)
                
                observations_proposer[j] = state
                actions_proposer[j] = action
                logprobs_proposer[j] = logprob
                values_proposer[j] = value
                orbit, _, done, _, info = env.step(action.numpy())
                predictor_timesteps = np.floor(np.linspace(0, args.orbit_duration-1, args.n_predictions + 1)).astype(int)
                phase_coords = np.transpose(orbit.y[:, predictor_timesteps][:,:-1])
                accs = np.transpose(orbit.a[:, predictor_timesteps][:,:-1])

                predictor_means, predictor_stds, predictor_value = agent_predictor(torch.tensor(phase_coords, dtype=torch.float32, device=device), 
                                                                                   torch.tensor(accs, dtype=torch.float32, device=device), 
                                                                                   torch.tensor(conditioning, dtype=torch.float32, device=device))
                predictor_means, predictor_stds = predictor_means.detach(), predictor_stds.detach() # (T, action_dim,)
                dist = torch.distributions.Normal(predictor_means, predictor_stds)
                predictor_action = dist.sample() # (T, action_dim,)
                predictor_logprob = dist.log_prob(predictor_action) # (T, action_dim,)
                observations_predictor = orbit
                actions_predictor[j] = predictor_action
                logprobs_predictor[j] = predictor_logprob
                values_predictor[j] = predictor_value
                prediction_distance = nn.functional.mse_loss(predictor_action, torch.tensor(np.transpose(orbit.y[:3 * args.num_bodies, predictor_timesteps][:,1:]), dtype=torch.float32, device=device) / env.high[0])
                # print(phase_coords)
                # print(predictor_action.mean(), torch.mean(torch.tensor(np.transpose(orbit.y[:3 * args.num_bodies, predictor_timesteps][:,1:]), dtype=torch.float32, device=device) / env.high[0]))
                rewards_proposer[j] = prediction_distance
                rewards_predictor[j] = -prediction_distance
                j += 1
            
            # advantage calculation
            advantages_proposer = torch.zeros_like(rewards_proposer).to(device)
            advantages_predictor = torch.zeros_like(rewards_predictor).to(device)
            lastgaelam = 0
            done_index = dones.nonzero().max().item() if dones.any() else args.num_steps
            for t in reversed(range(done_index)):
                advantages_proposer = lastgaelam = rewards_proposer + (1- dones) * args.gamma * values_proposer + args.gamma * args.gae_lambda * (1-dones) * lastgaelam
            lastgaelam = 0
            for t in reversed(range(done_index)):
                advantages_predictor = lastgaelam = rewards_predictor + (1- dones) * args.gamma * values_predictor + args.gamma * args.gae_lambda * (1-dones) * lastgaelam
            action_means, action_stds = actor_proposer(observations_proposer)
            dist = torch.distributions.Normal(action_means, action_stds)
            new_logprob_proposer = dist.log_prob(actions_proposer)
            actor_proposer_loss, approx_kl_proposer, clipfracs = policy_loss(logprobs_proposer, new_logprob_proposer, advantages_proposer.detach(), args.clip_coef)
            actor_proposer_loss = actor_proposer_loss.mean()
            clip_fracs_proposer += clipfracs
            optim_actor_proposer.zero_grad()
            actor_proposer_loss.backward()
            optim_actor_proposer.step()
            critic_loss_proposer = advantages_proposer.pow(2).mean()
            optim_critic_proposer.zero_grad()
            critic_loss_proposer.backward()
            optim_critic_proposer.step()

            orbit = observations_predictor
            predictor_timesteps = np.floor(np.linspace(0, args.orbit_duration, args.n_predictions + 1))
            predictor_means, predictor_stds, _ = agent_predictor(torch.tensor(phase_coords, dtype=torch.float32, device=device), 
                                                                torch.tensor(accs, dtype=torch.float32, device=device), 
                                                                torch.tensor(conditioning, dtype=torch.float32, device=device))
            dist = torch.distributions.Normal(predictor_means, predictor_stds)
            new_logprob_predictor = dist.log_prob(actions_predictor) # (T, action_dim,)
            actor_predictor_loss, approx_kl_predictor, clipfracs = policy_loss(logprobs_predictor, new_logprob_predictor, advantages_predictor.detach(), args.clip_coef)
            actor_predictor_loss = actor_predictor_loss.mean()
            clip_fracs_predictor += clipfracs
            critic_loss_predictor = advantages_predictor.pow(2).mean()
            predictor_loss = actor_predictor_loss + critic_loss_predictor
            optim_agent_predictor.zero_grad()
            predictor_loss.backward()
            optim_agent_predictor.step()

            if args.log_train:
                writer.add_scalar("loss/actor_loss_proposer", actor_proposer_loss.detach(), global_step=i)
                writer.add_scalar("loss/actor_loss_predictor", actor_predictor_loss.detach(), global_step=i)
                writer.add_scalar("reward/episode_reward_proposer", rewards_proposer.sum(dim=0).max().cpu().numpy(), global_step=i)
                writer.add_scalar("reward/episode_reward_predictor", rewards_predictor.sum(dim=0).max().cpu().numpy(), global_step=i)
                writer.add_scalar("loss/critic_loss_proposer", critic_loss_proposer.detach(), global_step=i)
                writer.add_scalar("loss/critic_loss_predictor", critic_loss_predictor.detach(), global_step=i)
                writer.add_scalar('charts/approx_kl_proposer', approx_kl_proposer.item(), global_step=i)
                writer.add_scalar('charts/approx_kl_predictor', approx_kl_predictor.item(), global_step=i)
                writer.add_scalar("charts/clipfrac_proposer", np.mean(clip_fracs_proposer), global_step=i)
                writer.add_scalar("charts/clipfrac_predictor", np.mean(clip_fracs_predictor), global_step=i)

            episodic_reward += rewards_proposer.sum(dim=0).max().cpu().item()
            state_list.append([episodic_reward]+env._denormalise_state(actions_proposer[0]).tolist())
            progress.set_description(f'episodic_reward: {episodic_reward}')
            progress.update()
    state_list = np.array(state_list)
    save_mask = state_list[:,0] > (np.max(state_list[:,0]) / 2 + np.mean(state_list[:,0]))
    columns = ['reward']
    for agent in range(args.num_bodies):
        columns += ['x', 'y', 'z', 'vx', 'vy', 'vz']
    save_df = pd.DataFrame(state_list[save_mask],columns=columns)
    save_df.to_csv(f'Dynamics/runs/{run_name}_best_performers.csv',index=False)