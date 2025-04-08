import gymnasium as gym
import numpy as np
try:
    from . import galaxy_models
except:
    import galaxy_models
import torch
from scipy.integrate import solve_ivp
import torch.nn.functional as F

class Dynamics(gym.Env):
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        # self.high = np.array([50.0, 50.0, 50.0, 300.0, 300.0, 300.0, 45.9114146119, 45.9114146119, 45.9114146119])
        self.high = np.array([50.0, 50.0, 50.0, 300.0, 300.0, 300.0])
        # self.low = np.array([-50.0, -50.0, -50.0, -300.0, -300.0, -300.0, 0., 0., 0.])
        self.low = np.array([-50.0, -50.0, -50.0, -300.0, -300.0, -300.0])
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

    def step(self, action, delta=1e-5):
        action = self._process_actions(action)
        self.init_params += action
        orbit = self._calculate_orbit()
        init_params_delta = self.init_params + np.random.normal(scale=delta, size=len(self.init_params))
        orbit_delta = self._calculate_orbit(init_params_delta)
        orbit_dists = np.linalg.norm(orbit.y - orbit_delta.y, axis=0)
        log_orbit_dists = np.log(orbit_dists + 1e-8)
        return self.init_params, orbit_rewards, False, False, info # rewards, dones must be calculated externally

    def reset(self, init_params=None):
        if init_params == None:
            self.init_params = self._process_actions((np.random.rand(6,)*2-1) * self.high / 50)
            print('[ENV]: Initial parameters not specified, using the following, selected randomly:', self.init_params)
        else:
            assert (np.abs(np.array(init_params)) <= self.high).all(), "If initial parameters are specified, they must be within the observation space"
            self.init_params = self._process_actions(np.array(init_params))
        # self.phase_coords = self.init_params
        # self.orbit = self._calculate_orbit()
        # self.orbit_timesteps = len(self.orbit.y[0])
        # self.orbit_ax, self.orbit_ay, self.orbit_az = self.get_acceleration(self.orbit.y[:3,:self.orbit_timesteps])
        # self.orbit_rewards = 0
        return self.init_params, self._get_info()

    def render(self, mode='human'):
        return
    
    def _calculate_orbit(self, init_params=None):
        if init_params == None: init_params = self.init_params
        return solve_ivp(self.get_equations, t_span=(0, self.orbit_duration), y0=init_params, t_eval=np.linspace(0, self.orbit_duration, self.orbit_timesteps))
        
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
        self.cuda = False
        self.predicted_orbit_delta_list = [10]
        self.orbit_model_epochs = int(1e4)
        self.orbit_timesteps = 1000
        self.orbit_duration = 100 # Myr
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available else 'cpu')
        print(f'[ENV] Using {self.device}')

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(f"[ENV] Seed set to {self.seed}")

    def _get_info(self):
        return {}
    
    def _process_actions(self, action):
        return np.clip(action, self.low / 10, self.high / 10)
    
    def _normalise_state(self, state):
        return state / self.high
    
    def _denormalise_state(self, state):
        return state * self.high

    def error(self, true_val, pred_val):
        return torch.sum(((true_val - pred_val)/pred_val)**2)
    
    def create_orbit_model(self, hidden_dim=64):
        return torch.nn.Sequential(
            torch.nn.Linear(self.observation_space.shape[0], hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.observation_space.shape[0]),
        )