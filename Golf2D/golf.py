from stable_baselines3 import PPO
from environment import GolfEnv
# from stable_baselines3.common.callbacks import BaseCallback

# class EpisodeEndCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(EpisodeEndCallback, self).__init__(verbose)

#     def _on_step(self):
#         if self.locals["dones"][0]:  # Check if episode ended
#             self.model.learn(total_timesteps=self.model.n_steps, reset_num_timesteps=False)
#         return True  # Continue training

# callback = EpisodeEndCallback()

env = GolfEnv()
model = PPO("MlpPolicy", env, verbose=2, n_steps = 256, device='cpu')
print('Learning...')
model.learn(total_timesteps=100000)

input_choice = 0
while int(input_choice) == 0 or int(input_choice) == 1:
    input_choice = input("Enter 0 to run the trial from default start, 1 to run the trial from random start, anything else to exit.\nChoice: ")
    try:
        int(input_choice)
    except: break
    print('Testing...')
    if int(input_choice) == 0:
        obs = env.reset(start=True)
    else: obs = env.reset(start=False)
    done = False
    steps = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action, True)
        steps += 1
        # env.render()
    print('That took %d steps.'%steps)
env.close()