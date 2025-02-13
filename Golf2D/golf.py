from torch import tensor, float32
import sys

TIMESTEPS = 1e5

try: model_choice = sys.argv[1]
except: model_choice = 'ppo'

if model_choice == 'ppo':
    import environment
    from Golf2D.ppo_vanilla import PPO
    env = environment.GolfEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hyperparameters = {
                    'timesteps_per_batch': 256, 
                    'max_timesteps_per_episode': 64, 
                    'gamma': 0.99, 
                    'n_updates_per_iteration': 10,
                    'lr': 3e-7, 
                    'clip': 0.2,
                    'render': True,
                    'render_every_i': 10,
                    'hidden_dim': 128,
                    'lambda_':0.95
                    }
    model = PPO(env, hyperparameters)
    print('Learning...')
    model.learn(total_timesteps=TIMESTEPS)
    input_choice = 0
    while True:
        input_choice = input("Enter 0 to run the trial from default start, any other number to run the trial from random start, anything else to exit.\nChoice: ")
        try:
            int(input_choice)
        except: break
        print('Testing...')
        if int(input_choice) == 0:
            obs = env.reset(start=True)
        else: obs = env.reset(start=False)
        obs = tensor(obs, dtype=float32)
        done = False
        steps = 0
        while not done:
            action, _log_probs = model.get_action(obs)
            obs, reward, done, _ = env.step(action, True)
            obs = tensor(obs, dtype=float32)
            steps += 1
        print('That took %d steps.'%steps)
    env.close()
elif model_choice == 'sb3':
    # import env_2 as environment
    import environment
    env = environment.GolfEnv()
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=2, n_steps = 256, device='cpu')
    print('Learning...')
    model.learn(total_timesteps=TIMESTEPS)
    input_choice = 0
    while True:
        input_choice = input("Enter 0 to run the trial from default start, any other number to run the trial from random start, anything else to exit.\nChoice: ")
        try:
            int(input_choice)
        except: break
        print('Testing...')
        if int(input_choice) == 0:
            obs= env.reset(start=True)
        else: obs = env.reset(start=False)
        done = False
        steps = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _= env.step(action, True)
            steps += 1
        print('That took %d steps.'%steps)
    env.close()
else:
    import environment
    from ppo_2 import PPO
    env = environment.GolfEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hyperparameters = {
                    'timesteps_per_batch': 256, 
                    'max_timesteps_per_episode': 128, 
                    'gamma': 0.99, 
                    'n_updates_per_iteration': 10,
                    'lr': 3e-8, 
                    'clip': 0.2,
                    'render': True,
                    'render_every_i': 10,
                    'hidden_dim': 128
                    }
    model = PPO(env, hyperparameters)
    print('Learning...')
    model.learn(total_timesteps=TIMESTEPS)
    input_choice = 0
    while True:
        input_choice = input("Enter 0 to run the trial from default start, any other number to run the trial from random start, anything else to exit.\nChoice: ")
        try:
            int(input_choice)
        except: break
        print('Testing...')
        if int(input_choice) == 0:
            obs = env.reset(start=True)
        else: obs = env.reset(start=False)
        obs = tensor(obs, dtype=float32)
        done = False
        steps = 0
        while not done:
            action, _log_probs = model.get_action(obs)
            obs, reward, done, _= env.step(action, True)
            obs = tensor(obs, dtype=float32)
            steps += 1
        print('That took %d steps.'%steps)
    env.close()