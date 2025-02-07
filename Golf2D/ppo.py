import torch
import gym
import time
import numpy as np

class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.seq(x)
    
class PPO(torch.nn.Module):
    def __init__(self, env, hyperparameters:dict):
        super(PPO, self).__init__()
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.actor = FeedForward(self.state_dim, self.action_dim, self.hidden_dim)
        self.actor_log_std = torch.nn.Parameter(torch.eye(self.action_dim, self.action_dim)*0.5)
        self.critic = FeedForward(self.state_dim, 1, self.hidden_dim)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            'timesteps': 0,          # timesteps so far
            'iterations': 0,          # iterations so far
            'epoch_lengths': [],       # episodic lengths in batch
            'epoch_rewards': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

    def learn(self, total_timesteps:int):
        timesteps = 0
        iterations = 0
        while timesteps < total_timesteps:
            epoch_observations, epoch_actions, epoch_log_probs, epoch_rewards_to_go, epoch_lengths = self.rollout()
            timesteps += np.sum(epoch_lengths)
            iterations += 1

            self.logger['timesteps'] = timesteps
            self.logger['iterations'] = iterations

            V, _ = self.evaluate(epoch_observations, epoch_actions)
            A_k = epoch_rewards_to_go - V.detach()

            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, current_log_probs = self.evaluate(epoch_observations, epoch_actions)
                ratio = torch.exp(current_log_probs - epoch_log_probs)

                min_ratio = torch.min(ratio, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon))*A_k
                actor_loss = -torch.mean(min_ratio)
                critic_loss = torch.nn.functional.mse_loss(V, epoch_rewards_to_go)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach().numpy())
            
            self._log_summary()
            if iterations % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './Golf2D/weights/actor.pth')
                torch.save(self.actor.state_dict(), './Golf2D/weights/critic.pth')
        
    def rollout(self):
        epoch_observations = []
        epoch_actions = []
        epoch_log_probs = []
        epoch_rewards_to_go = []
        epoch_rewards = []
        epoch_lengths = []

        episode_rewards = []
        timesteps = 0

        while timesteps < self.timesteps_per_batch:
            episode_rewards = []

            observations, _ = self.env.reset()
            observations = torch.tensor(observations, dtype=torch.float32)
            done = False

            for episode_timestep in range(self.max_timesteps_per_episode):
                # if self.render ->
                epoch_observations.append(observations)
                action, log_prob = self.get_action(observations)
                observations, rewards, terminated, truncated, _ = self.env.step(action)
                observations = torch.tensor(observations, dtype=torch.float32)

                done = terminated | truncated

                episode_rewards.append(rewards)
                epoch_actions.append(torch.tensor(action, dtype=torch.float32))
                epoch_log_probs.append(log_prob)

                timesteps += 1
                if done: break
                
            epoch_lengths.append(episode_timestep + 1)
            epoch_rewards.append(episode_rewards)

        epoch_observations = torch.stack(epoch_observations, dim=0)
        epoch_actions = torch.stack(epoch_actions, dim=0)
        epoch_log_probs = torch.stack(epoch_log_probs, dim=0)
        epoch_rewards_to_go = self.compute_rewards_to_go(epoch_rewards)

        self.logger['epoch_rewards'] = epoch_rewards
        self.logger['epoch_lengths'] = epoch_lengths
                
        return epoch_observations, epoch_actions, epoch_log_probs, epoch_rewards_to_go, epoch_lengths
    
    def compute_rewards_to_go(self, epoch_rewards):
        epoch_rewards_to_go = []

        for episode_rewards in reversed(epoch_rewards):
            discounted_reward = 0

            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                epoch_rewards_to_go.insert(0, discounted_reward)
        
        epoch_rewards_to_go = torch.tensor(epoch_rewards_to_go, dtype=torch.float32)
        return epoch_rewards_to_go

    def get_action(self, observations):
        mean_action = self.actor(observations)
        cov_mat = self.actor_log_std
        dist = torch.distributions.MultivariateNormal(mean_action, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, epoch_observations, epoch_actions):
        V = self.critic(epoch_observations).squeeze(dim=-1)
        mean_action = self.actor(epoch_observations)
        dist = torch.distributions.MultivariateNormal(mean_action, self.actor_log_std)
        log_probs = dist.log_prob(epoch_actions)
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.lr = 5e-4
        self.gamma = 0.99
        self.epsilon = 0.2
        self.hidden_dim = 64

        self.render = True
        self.seed = None
        self.save_freq = 10

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Seed set to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        timesteps = self.logger['timesteps']
        iterations = self.logger['iterations']
        avg_ep_lens = np.mean(self.logger['epoch_lengths'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['epoch_rewards']])
        avg_actor_loss = np.mean([losses.mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{iterations} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {timesteps}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []