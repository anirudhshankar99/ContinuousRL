import gymnasium as gym
import numpy as np
try:
    from . import planetary_models
except:
    import planetary_models
from torch import log10 as ln

EARTH_MASS_IN_SI = 5.972e24
SOLAR_MASS_IN_SI = 2e30

class Dynamics(gym.Env):
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.planetary_models = []
        for model_name, model_kwargs_dict in zip(self.planetary_model_list, self.planetary_model_kwargs_list):
            self.planetary_models.append(planetary_models.add_planetary_model(model_name, **{'%s'%key:value for key,value in model_kwargs_dict.items()}))
        planet_info_scale_high = [np.array([self.box_scale, self.box_scale, 1e8, 1e8, SOLAR_MASS_IN_SI]) for _ in range(len(self.planetary_models))]
        planet_info_scale_high = [item for sublist in planet_info_scale_high for item in sublist]
        planet_info_scale_low = [np.array([self.box_scale, self.box_scale, 1e8, 1e8, 0]) for _ in range(len(self.planetary_models))]
        planet_info_scale_low = [item for sublist in planet_info_scale_low for item in sublist]
        self.high = np.array([self.box_scale, self.box_scale, 1e8, 1e8, 1e12, 1e12, self.rocket_mass]+planet_info_scale_high+[self.box_scale]) # pos x, y, vel x, y, acc x, y, m, (pos x,y, vel x,y, mass) x planets, distance_to_dest x,y
        self.mass_position_in_state = [7,4]
        self.low = np.array([-self.box_scale, -self.box_scale, -1e8, -1e8, -1e12, -1e12, 0.0]+planet_info_scale_low+[self.box_scale])
        self.action_bounds = np.array([self.max_engine_thrust, self.max_engine_thrust]) # thrust~mdot*ve x, y, on/off
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
        self.max_engine_thrust = 7500e3 # in N
        self.fuel_frac = 0.9
        self.v_e = 3500 # in m/s

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

    def _get_info(self):
        return {}
    
    def _process_actions(self, action):
        unit_action = action / np.linalg.norm(action)
        return unit_action * self.action_bounds
    
    def _normalise_state(self, state):
        # planet_mass_mask = [True if (i-self.mass_position_in_state[0])%self.mass_position_in_state[1]==0 and i-self.mass_position_in_state[0]>0 else False for i in range(len(self.high))]
        state = state / self.high
        # log_state = np.log10(state)
        # state[planet_mass_mask] = log_state[planet_mass_mask]
        return state
    
    def _denormalise_state(self, state):
        return state * self.high