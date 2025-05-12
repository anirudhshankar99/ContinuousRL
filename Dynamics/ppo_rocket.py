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
from dynamics_env.dynamics_rocket import Dynamics
from scipy.integrate import solve_ivp

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
    # parser.add_argument('--num-steps', type=int, default=2048,
    #                     help='number of steps per environment per rollout')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, gae will not be computed')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='the value of the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the value of the lambda parameter for gae')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help='the surrogate ratios\' clipping coefficient')
    parser.add_argument('--v-e', type=float, default=3000,
                        help='exhaust velocity of the rocket in m/s')
    parser.add_argument('--fuel-frac', type=float, default=0.9,
                        help='fraction of the rocket\'s takeoff mass comprised of fuel')
    parser.add_argument('--orbit-timesteps', type=int, default=1000,
                        help='number of timesteps over which the orbit is integrated')
    parser.add_argument('--orbit-duration', type=float, default=100,
                        help='orbit time in years')
    parser.add_argument('--max-engine-thrust', type=float, default=7500e3, # supposed to be 7500e3
                        help='maximum possible engine thrust in N')
    parser.add_argument('--rocket-mass', type=float, default=433100)
    parser.add_argument('--destination_type', type=str, default='radius',
                        help='whether the destination is a radius limit or a location')
    parser.add_argument('--start', type=float, nargs='+', default=None,
                        help='coordinates of the starting point')
    parser.add_argument('--destination-params', type=float, nargs='+', default=None,
                        help='parameters of the chosen destination type')
    parser.add_argument('--capture-radius', type=float, default = 6371e2,
                        help='radius at which rocket is deemed captured by a planet')
    args = parser.parse_args()
    args.dry_mass = args.rocket_mass * (1 - args.fuel_frac)
    destination_types = ['radius', 'destination', 'destination_planet']
    args.num_steps = int(args.orbit_timesteps)
    assert args.destination_type in destination_types
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

def reward_function(pos, prev_pos, mass, prev_mass):
    if args.destination_type == 'radius':
        current_radius = np.linalg.norm(pos - start_center, axis=-1)
        prev_radius = np.linalg.norm(prev_pos - start_center, axis=-1)
        consistent_reward = (current_radius - prev_radius) / (destination_radius - start_radius) + (mass - prev_mass) / args.rocket_mass
        completion_reward = current_radius >= destination_radius
    elif args.destination_type == 'destination_planet':
        planet_position = env.planetary_models[destination_planet_index].get_position()
        consistent_reward = (start_planet_distance - np.linalg.norm(planet_position - pos, axis=-1)) / start_planet_distance + (mass - prev_mass) / args.rocket_mass
        completion_reward = np.linalg.norm(pos - planet_position, axis=-1) < destination_planet_radius
    elif args.destination_type == 'destination':
        consistent_reward = (start_destination_distance - np.linalg.norm(destination_coords - pos, axis=-1)) / start_destination_distance + (mass - prev_mass) / args.rocket_mass
        completion_reward = np.linalg.norm(pos - destination_coords, axis=-1) < destination_radius
    # print(consistent_reward)
    # print(completion_reward)
    return consistent_reward + completion_reward

def done_function(pos):
    if args.destination_type == 'radius':
        current_radius = np.linalg.norm(pos - start_center)
        completion_reward = current_radius >= destination_radius
    elif args.destination_type == 'destination_planet':
        planet_position = env.planetary_models[destination_planet_index].get_position()
        completion_reward = np.linalg.norm(pos - planet_position) < destination_planet_radius
    elif args.destination_type == 'destination':
        completion_reward = np.linalg.norm(pos - destination_coords) < destination_radius
    return completion_reward

def event_dest_reached(t, y):
    pos = y[:2]
    if args.destination_type == 'radius':
        current_radius = np.linalg.norm(pos - start_center)
        print(current_radius)
        completion = 0 if current_radius >= destination_radius else 1
    elif args.destination_type == 'destination_planet':
        planet_position = env.planetary_models[destination_planet_index].get_position()
        completion = 0 if np.linalg.norm(pos - planet_position) < destination_planet_radius else 1
    elif args.destination_type == 'destination':
        completion = 0 if np.linalg.norm(pos - destination_coords) < destination_radius else 1
    return completion
event_dest_reached.terminal = True
def rocket_captured(t, y):
    pos = y[:2]
    min_dist = np.inf
    for model in env.planetary_models:
        dist = np.linalg.norm(pos - model.get_position(t))
        min_dist = min(min_dist, dist)
    return min_dist - args.capture_radius
rocket_captured.terminal = True

def rocket_function(t, y):
    # state packaging
    episode_step = int(t // (args.orbit_duration / args.orbit_timesteps))
    pos = y[:2]
    vel = y[2:4]
    mass = y[-1:]
    a_gravity = env.get_acceleration(np.concat([pos, np.array([t])]))
    state = env._normalise_state(torch.tensor(np.concat([pos, vel, a_gravity, mass]), dtype=torch.float32, device=device)).float()
    # action sampling
    with torch.no_grad():
        action_means, action_stds = actor(state)
    value = critic(state).flatten()
    dist = torch.distributions.Normal(action_means, action_stds)
    action = dist.sample()
    logprob = dist.log_prob(action)
    action = env._process_actions(action)
    # rocket science
    thrust = action.float().numpy() # in N (kg m/s^2)
    mdot = -np.linalg.norm(thrust, axis=-1) / args.v_e # in m/s^2
    delta_m = mdot * args.orbit_duration / args.orbit_timesteps
    if (mass - delta_m) < args.dry_mass:
        a_thrust = np.zeros_like(thrust)
        mdot = 0
    else:
        a_thrust = thrust / mass

    # PPO
    observations[episode_step] = state
    actions[episode_step] = action
    logprobs[episode_step] = logprob
    done = done_function(pos)
    dones[episode_step] = torch.tensor(done, dtype=torch.float32).to(device)
    values[episode_step] = value

    # derivatives for integrator
    dydt = np.zeros_like(y)
    dydt[:2] = vel
    dydt[2:4] = a_gravity + a_thrust
    dydt[4] = mdot
    return dydt

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
                'planetary_model_list':['point_source', 'point_source', 'point_source'],
                'planetary_model_kwargs_list':[{'M':2e30, 'period':1e10, 'orbit_radius':0, 'phase':0}, # sun
                                                {'M':1.898e27, 'period':11.86, 'orbit_radius':7.7866e11, 'phase':0.785}, # jupiter
                                                {'M':5.972e24, 'period':1, 'orbit_radius':1.496e11, 'phase':3.945}], # earth
                'seed':seed,
                'max_engine_thrust':args.max_engine_thrust,
            })
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    env = make_env(seed=args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "must be a continuous action space"

    actor = Actor(env, activation=Mish).to(device)
    critic = Critic(env, activation=Mish).to(device)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    if args.log_train:
        writer = tensorboard.SummaryWriter(f'Dynamics/runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )
    start_time = time.time()
    episodic_reward = 0
    print(f'[AGENT] Using {device}')
    if args.start == None:
        leo_distance = 6371e3 + 300e3 # 300e3 m
        leo_speed = 7.7e3 # 7.7e3 m/s
        earth_orbital_speed = 2.977e4 # m/s
        launch_theta = np.pi / 2 # np.random.rand() * 2 * np.pi * 0
        earth_phase, earth_orbit_radius = env.planetary_models[-1].phase, env.planetary_models[-1].orbit_radius
        launch_position = np.array([np.cos(launch_theta), np.sin(launch_theta)]) * leo_distance + np.array([np.cos(earth_phase), np.sin(earth_phase)]) * earth_orbit_radius
        launch_velocity = np.array([np.cos(launch_theta), np.sin(launch_theta)]) * leo_speed + env.planetary_models[-1].get_velocity(0, earth_orbital_speed)
        launch_mass = np.array([args.rocket_mass])
        init_params = np.concat([launch_position, launch_velocity, launch_mass])
    else: init_params = np.array(args.start)
    if args.destination_type == 'radius':
        if args.destination_params == None:
            destination_radius = 35768e3 # geocentric orbit radius
            start_center = np.array([np.cos(earth_phase), np.sin(earth_phase)]) * earth_orbit_radius
            start_radius = 6371e3 + 300e3
        else:
            destination_radius = args.destination_params[0]
            start_center = args.destination_params[1]
            start_radius = args.destination_params[2]
    elif args.destination_type == 'destination_planet':
        if args.destination_params == None:
            destination_planet_index = len(env.planetary_models)-1
            destination_planet_radius = 6051.8e3 # m (venus)
        else:
            destination_planet_index = args.destination_params[0]
            destination_planet_radius = args.destination_params[1]
        start_planet_distance = np.linalg.norm(init_params[:2] - env.planetary_models[destination_planet_index].get_position())
    elif args.destination_type == 'destination':
        if args.destination_params == None:
            destination_distance = 3 * 1.496e11 # 3au in m
            destination_theta = np.random.rand() * 2 * np.pi # in rad
            destination_radius = 6371e3 # earth radius in m
        else:
            destination_distance = args.destination_params[0]
            destination_theta = args.destination_params[1]
            destination_radius = args.destination_params[2]
        start_destination_distance = np.linalg.norm(init_params[:2] - np.linalg.norm(np.array([np.cos(destination_theta), np.sin(destination_theta)]) * destination_distance))
        destination_coords = np.array([np.cos(destination_theta), np.sin(destination_theta)]) * destination_distance + np.array([np.cos(earth_phase), np.sin(earth_phase)]) * earth_orbit_radius

    with tqdm(range(int(args.total_timesteps)), desc=f'episodic_reward: {episodic_reward}') as progress:
        for i in range(int(args.total_timesteps)):
            prev_logprob = None
            done = False
            state = torch.tensor(init_params, dtype=torch.float32).to(device)

            observations = torch.zeros((args.num_steps,)+env.observation_space.shape, dtype=torch.float32).to(device)
            actions = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            logprobs = torch.zeros((args.num_steps,)+env.action_space.shape, dtype=torch.float32).to(device)
            rewards = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            dones = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            values = torch.zeros((args.num_steps,), dtype=torch.float32).to(device)
            clip_fracs = []
            orbit = solve_ivp(rocket_function, t_span=(0, args.orbit_duration), y0=init_params, t_eval=np.linspace(0, args.orbit_duration, args.orbit_timesteps), events=[event_dest_reached, rocket_captured])
            denorm_observations = env._denormalise_state(observations).numpy()
            done_index = dones.nonzero().max().item() if dones.any() else args.num_steps
            rewards[1:done_index] = torch.tensor(reward_function(denorm_observations[1:,:2], denorm_observations[:-1,:2], denorm_observations[1:,-1], denorm_observations[:-1,-1]), dtype=torch.float32, device=device)[:done_index+1]
            break
            # advantage calculation
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(done_index)):
                if t == args.num_steps - 1:
                    advantages[t] = lastgaelam = rewards[t] + (1- dones[t]) * args.gamma * values[t] + args.gamma * args.gae_lambda * (1-dones[t]) * lastgaelam
                else:
                    advantages[t] = lastgaelam = rewards[t] + (1-dones[t+1]) * args.gamma * values[t+1] - (1-dones[t])*values[t] + args.gamma * args.gae_lambda * (1-dones[t+1]) * lastgaelam  

            action_means, action_stds = actor(observations)
            dist = torch.distributions.Normal(action_means, action_stds)
            new_logprobs = dist.log_prob(actions)
            actor_loss, approx_kl, clipfracs = policy_loss(logprobs, new_logprobs, advantages.detach(), args.clip_coef)
            actor_loss = actor_loss.mean()
            clip_fracs += clipfracs
            adam_actor.zero_grad()
            actor_loss.backward()
            adam_actor.step()

            critic_loss = advantages.pow(2).mean()
            adam_critic.zero_grad()
            critic_loss.backward()

            if args.log_train:
                writer.add_scalar("loss/actor_loss", actor_loss.detach(), global_step=i)
                writer.add_histogram("gradients/actor",
                                torch.cat([p.grad.view(-1) for p in actor.parameters()]), global_step=i)
                writer.add_scalar("loss/advantage", advantages.detach().cpu().numpy().mean(), global_step=i)
                writer.add_scalar("reward/episode_reward", rewards.sum(dim=0).max().cpu().numpy(), global_step=i)
                writer.add_scalar("loss/critic_loss", critic_loss.detach(), global_step=i)
                writer.add_histogram("gradients/critic",
                                torch.cat([p.grad.view(-1) for p in critic.parameters()]), global_step=i)
                writer.add_scalar('charts/learning_rate', adam_actor.param_groups[0]['lr'], global_step=i)
                writer.add_scalar('charts/approx_kl', approx_kl.item(), global_step=i)
                writer.add_scalar("charts/clipfrac", np.mean(clip_fracs), global_step=i)
                writer.add_scalar('charts/SPS', int(i/ (time.time() - start_time)), i)
            adam_critic.step()

            episodic_reward = rewards.sum(dim=0).max().cpu().numpy()
            progress.set_description(f'episodic_reward: {episodic_reward}')
            progress.update()