import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the LR of the optimizer(s)')
    parser.add_argument('--seed', type=int, default=1,
                        help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2.5e4,
                        help='total timesteps of the experiment')
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
    
    # Performance altering
    parser.add_argument('--num-envs', type=int, default=4,
                        help='number of parallel environments trained in')
    parser.add_argument('--num-steps', type=int, default=256,
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

if __name__ == '__main__':
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
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
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
    assert isinstance(env.action_space, gym.spaces.Discrete), "must be a discrete action space"

    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps,) + env.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps,) + env.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps,)).to(device)
    rewards = torch.zeros((args.num_steps,)).to(device)
    dones = torch.zeros((args.num_steps,)).to(device)
    values = torch.zeros((args.num_steps,)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(env.reset()[0]).to(device)
    next_done = torch.zeros(1).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # for update in range(1, int(num_updates + 1)): #change to while less than total updates (?)
    update = 0
    while global_step < args.total_timesteps:
        update += 1
        # if args.anneal_lr:
        #     frac = 1.0 - (update - 1.0) / num_updates
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]['lr'] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, _, info = env.step(action.cpu().numpy())

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)

            if 'episode' in info.keys():
                print(f'global_step = {global_step}, episodic_return={info["episode"]["r"]}')
                writer.add_scalar('charts/episodic_return', info["episode"]["r"], global_step)
                writer.add_scalar('charts/episodic_length', info["episode"]["l"], global_step)
                next_obs = torch.Tensor(env.reset()[0]).to(device)
                break

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # if args.gae:
            #     advantages = torch.zeros_like(rewards).to(device)
            #     lastgaelam = 0
            #     for t in reversed(range(args.num_steps)):
            #         if t == args.num_steps - 1:
            #             nextnonterminal = 1.0 - next_done
            #             nextvalues = next_value
            #         else:
            #             nextnonterminal = 1.0 - dones[t + 1]
            #             nextvalues = values[t + 1]
            #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            #         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            #     returns = advantages + values
            # else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

        # b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)
        
        # b_inds = np.arange(args.batch_size)
        # clipfracs = []
        for epoch in range(args.update_epochs):
            # np.random.shuffle(b_inds)
            # for start in range(0, args.batch_size, args.minibatch_size):
                # end = start + args.minibatch_size
                # mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                obs, actions
            )
            logratio = newlogprob - logprobs
            ratio = logratio.exp()
            # print(ratio.mean())

            # with torch.no_grad():
            #     old_approx_kl = (-logratio).mean()
            #     approx_kl = ((ratio - 1) - logratio).mean()
            #     clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            # mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            # pg_loss = (-advantages * ratio).mean()

            newvalue = newvalue.view(-1)
            # if args.clip_vloss:
            #     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            #     v_clipped = b_values[mb_inds] + torch.clamp(
            #         newvalue - b_values[mb_inds],
            #         -args.clip_coef,
            #         args.clip_coef
            #     )
            #     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            #     v_loss = 0.5 * v_loss_max.mean()
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean() / 10

            entropy_loss = entropy.mean()
            # print(pg_loss, 'pg_loss')
            # print(v_loss, 'v_loss')
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            
            # if args.target_kl is not None:
            #     if approx_kl > args.target_kl:
            #         break
        
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
        # writer.add_scalar('losses/approx_kl', approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar('losses/explained_variance', explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar('charts/SPS', int(global_step/ (time.time() - start_time)), global_step)
    env.close()
    writer.close()