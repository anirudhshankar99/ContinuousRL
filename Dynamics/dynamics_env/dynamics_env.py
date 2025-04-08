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

    def step(self, action):
        action = self._process_actions(action)
        self.init_params += action
        orbit = self._calculate_orbit()
        # out_of_box_mask = orbit[:3] > np.expand_dims(self.high[:3], axis=-1)
        # first_index = np.argmax(out_of_box_mask, axis=1, keepdims=True).min()
        # orbit = orbit[:,:first_index]
        orbit_model = self.create_orbit_model()
        orbit_model_opt = torch.optim.Adam(orbit_model.parameters(), lr=1e-3)
        prev_states = torch.tensor(orbit[:,:-1], dtype=torch.float32, device=self.device).transpose(dim0=-1, dim1=-2)
        dest_states = torch.tensor(orbit[:,1:], dtype=torch.float32, device=self.device).transpose(dim0=-1, dim1=-2)
        for _ in range(self.orbit_model_epochs):
            x = orbit_model(prev_states)
            loss = F.mse_loss(x, dest_states)
            orbit_model_opt.zero_grad()
            loss.backward()
            orbit_model_opt.step()
        orbit_rewards = 0
        info = {}
        for delta in self.predicted_orbit_delta_list:
            delta_orbit = prev_states[:-delta]
            delta_dest_orbit = dest_states[delta:]
            for _ in range(delta):
                with torch.no_grad():
                    delta_orbit = orbit_model(delta_orbit)
            orbit_rewards += self.error(delta_dest_orbit, delta_orbit).item() / len(prev_states)
            info['%d_y'%delta] = delta_dest_orbit
            info['%d_x'%delta] = delta_orbit
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
    
    def _calculate_orbit(self):
        return solve_ivp(self.get_equations, t_span=(0, self.orbit_duration), y0=self.init_params, t_eval=np.linspace(0, self.orbit_duration, self.orbit_timesteps)).y[:]
        
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