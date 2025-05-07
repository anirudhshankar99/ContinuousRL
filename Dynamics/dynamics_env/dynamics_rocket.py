import gymnasium as gym
import numpy as np
try:
    from . import planetary_models
except:
    import planetary_models
from torch import log10 as ln

class Dynamics(gym.Env):
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.high = np.array([self.box_scale, self.box_scale, 1e8, 1e8, 1e12, 1e12, self.rocket_mass]) # pos x, y, vel x, y, acc x, y, m
        self.low = np.array([-self.box_scale, -self.box_scale, -1e8, -1e8, -1e12, -1e12, 0.0])
        self.action_bounds = np.array([self.max_thrust, self.max_thrust]) # thrust~mdot*ve x, y, on/off
        self.action_space = gym.spaces.Box(
            low=-self.action_bounds,
            high=self.action_bounds,
            dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=self.low, 
            high=self.high, 
            dtype=np.float64
        )
        self.planetary_models = []
        for model_name, model_kwargs_dict in zip(self.planetary_model_list, self.planetary_model_kwargs_list):
            self.planetary_models.append(planetary_models.add_planetary_model(model_name, **{'%s'%key:value for key,value in model_kwargs_dict.items()}))

    def render(self, mode='human'):
        return
        
    def get_acceleration(self, pos):
        ax, ay = 0, 0
        for planetary_model in self.planetary_models:
            dax, day = planetary_model.get_acceleration(pos)
            ax, ay = ax + dax, ay + day
        return np.reshape(np.array([ax, ay]), shape=(-1))

    def _init_hyperparameters(self, hyperparameters):
        self.planetary_model_list = ['point_source']
        self.planetary_model_kwargs_list = [{}]
        self.seed = 0
        self.rocket_mass = 433100
        self.box_scale = 4.578e12 # in km
        self.max_thrust = 981000 # in N
        self.fuel_frac = 0.9
        self.v_e = 3500 # in m/s

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

    def _get_info(self):
        return {}
    
    def _process_actions(self, action):
        return np.clip(action * self.action_bounds, -self.action_bounds, self.action_bounds)
    
    def _normalise_state(self, state):
        return state / self.high
    
    def _denormalise_state(self, state):
        return state * self.high