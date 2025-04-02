import gymnasium as gym
import numpy as np
try:
    from . import galaxy_models
except:
    import galaxy_models
import torch
from scipy.integrate import solve_ivp

class Dynamics(gym.Env):
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.high = np.array([50.0, 50.0, 50.0, 300.0, 300.0, 300.0, 45.9114146119, 45.9114146119, 45.9114146119])
        self.low = np.array([-50.0, -50.0, -50.0, -300.0, -300.0, -300.0, 0., 0., 0.])
        self.action_space = gym.spaces.Box(
            low=self.low/10,
            high=self.high/10,
            dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=self.low, 
            high=self.high, 
            dtype=np.float64
        )
        self.galaxy_models = []
        for model_name, model_kwargs_dict in zip(self.galaxy_model_list, self.galaxy_model_kwargs_list):
            self.galaxy_models.append(galaxy_models.add_galaxy_model(model_name, **{'%s'%key:value for key,value in model_kwargs_dict.items()}))

    def step(self, action):
        action = self._process_actions(action)
        self.init_params += action
        self.orbit = self._calculate_orbit()
        self.orbit_timesteps = len(self.orbit.y[0])
        self.orbit_accelerations = self.get_acceleration(self.orbit.y[:,self.orbit_timesteps])
        return self.init_params, -self.orbit_rewards, False, False, self._get_info() # rewards, dones must be calculated externally
    
    def reset_2(self):
        self.timestep = 0
        self.phase_coords = np.array(self.orbit.y[:,self.timestep])
        # cat accelerations
        return self._normalise_state(self.phase_coords), self._get_info_2()
    
    def step_2(self, action):
        self.timestep += 1
        action = self._process_actions_2(action)
        self.phase_coords = np.clip(self.phase_coords + action, self.low[:3], self.high[:3])
        orbit_phase_coords = np.array(self.orbit.y[:,self.timestep])
        reward = -self.chi_sq(self.phase_coords, orbit_phase_coords)
        done = self.timestep == self.orbit_timesteps - 1 or (np.abs(orbit_phase_coords) >= self.high[:3]).any()
        self.orbit_rewards += reward
        # cat accelerations
        return self._normalise_state(self.phase_coords), reward, done, False, self._get_info_2()

    def reset(self, seed=None, options=None, init_params=None):
        if init_params == None:
            self.init_params = self._process_actions((np.random.rand(6,)*2-1) * self.high[:3] / 10)
            print('[ENV]: Initial parameters not specified, using the following, selected randomly:',self.init_params)
        else:
            assert (np.abs(np.array(init_params)) <= self.high[:3]).all(), "If initial parameters are specified, they must be within the observation space"
            self.init_params = self._process_actions(np.array(init_params))

        self.phase_coords = self.init_params
        self.orbit = self._calculate_orbit()
        self.orbit_timesteps = len(self.orbit.y[0])
        self.orbit_ax, self.orbit_ay, self.orbit_az = self.get_acceleration(self.orbit.y[:3,:self.orbit_timesteps])
        self.orbit_rewards = 0
        return self.init_params, self._get_info()

    def render(self, mode='human'):
        return
    
    def _calculate_orbit(self):
        return solve_ivp(self.get_equations, t_span=(0, self.orbit_duration), y0=self.init_params, t_eval=np.linspace(0, self.orbit_duration, self.orbit_timesteps))
        
    def get_acceleration(self, pos):
        ax, ay, az = 0, 0, 0
        for galaxy_model in self.galaxy_models:
            dax, day, daz = galaxy_model.get_acceleration(pos)
            ax, ay, az = ax + dax, ay + day, az + daz
        return ax, ay, az
    
    def get_equations(self, t, w):
        x, y, z, vx, vy, vz = w
        ax, ay, az = self.get_acceleration(np.array([x, y, z]))
        return [vx, vy, vz, ax.item(), ay.item(), az.item()]

    def _init_hyperparameters(self, hyperparameters):
        self.galaxy_model_list = ['point_source']
        self.galaxy_model_kwargs_list = [{}]
        self.seed = 0
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.orbit_timesteps = 1000
        self.orbit_duration = 100 # Myr
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(f"[ENV] Seed set to {self.seed}")

    def _get_info(self):
        return {}
    
    def _get_info_2(self):
        return {}
    
    def _process_actions(self, action):
        return np.clip(action, self.low / 10, self.high[:3] / 10)

    def _process_actions_2(self, action):
        return np.clip(self._denormalise_state(action), self.low, self.high[:3])
    
    def _normalise_state(self, state):
        return state / self.high
    
    def _denormalise_state(self, state):
        return state * self.high

    def chi_sq(self, true_val, pred_val, sigma=1):
        return np.sum((true_val - pred_val)**2 / sigma**2)