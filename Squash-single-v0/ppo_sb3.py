import random
import gymnasium as gym
import squash_env
import numpy as np
import torch
from stable_baselines3 import PPO

seed = 0
if __name__ == '__main__':

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available and False else 'cpu')

    def make_env(gym_id, seed):
        def thunk():
            env = gym.make(gym_id)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
    #     for i in range(args.num_envs)])
    env = make_env('CartPole-v1', 0)()
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "must be a discrete action space"
    agent = PPO("MlpPolicy", env, verbose=2, n_steps = 1024, device=device)
    agent.policy.optimizer.step = lambda: None
    agent.learn(total_timesteps=2.5e5)
    env.close()