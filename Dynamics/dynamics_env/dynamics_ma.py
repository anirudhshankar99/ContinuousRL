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
        high = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        low = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0])
        self.n_agents = len(self.dynamic_potential_list)
        self.high = np.repeat(np.expand_dims(high, 0), self.n_agents, 0).flatten()
        self.low = np.repeat(np.expand_dims(low, 0), self.n_agents, 0).flatten()
        self.action_space = gym.spaces.Box(
            low=self.low,
            high=self.high,
            dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=self.low, 
            high=self.high, 
            dtype=np.float64
        )

    def step(self, action, delta=1e-8):
        action = self._process_actions(action)
        self.init_params = np.clip(self.init_params + action, self.low, self.high)
        init_params_delta = self.init_params + np.random.normal(scale=delta, size=len(self.init_params))
        orbit = self._calculate_orbit()
        orbit_delta = self._calculate_orbit(init_params=init_params_delta)

        orbit_dists = np.linalg.norm(orbit.y - orbit_delta.y, axis=0)
        log_orbit_dists = np.log(orbit_dists + 1e-8)
        fit_coeffs = np.polyfit(orbit.t, log_orbit_dists, 1)
        max_r = np.max(np.linalg.norm(orbit.y[:3], axis=0))
        reward = fit_coeffs[0] * self.out_of_bounds_damping(max_r)

        return self._get_ma(self.init_params), self._get_ma(reward), self._get_ma(False), self._get_ma(False), {}

    def reset(self, init_params=[]):
        if len(init_params) == 0:
            self.init_params = self._process_actions((np.random.rand(6,)*2-1) * self.high / 50)
        else:
            assert (np.abs(np.array(init_params)) <= self.high).all(), "If initial parameters are specified, they must be within the observation space"
            self.init_params = self._process_actions(np.array(init_params))
        self.stationary_potentials = []
        for model_name, model_kwargs_dict in zip(self.stationary_potential_list, self.stationary_potential_kwargs_list):
            self.stationary_potentials.append(galaxy_models.add_galaxy_model(model_name, **{'%s'%key:value for key,value in model_kwargs_dict.items()}))
        self.dynamic_potentials = []
        for model_name, model_kwargs_dict in zip(self.dynamic_potential_list, self.dynamic_potential_kwargs_list):
            self.dynamic_potentials.append(galaxy_models.add_galaxy_model(model_name, **{'%s'%key:value for key,value in model_kwargs_dict.items()}))
        return self._get_ma(self.init_params), self._get_info()

    def render(self, mode='human'):
        return
    
    def _calculate_orbit(self, init_params=[]):
        if len(init_params) == 0: init_params = self.init_params
        return solve_ivp(self.get_equations, t_span=(0, self.orbit_duration), y0=init_params, t_eval=np.linspace(0, self.orbit_duration, self.orbit_timesteps))
        
    def get_acceleration(self, pos):
        a = []
        for agent in range(self.n_agents):
            agent_ax, agent_ay, agent_az = 0, 0, 0
            for galaxy_model in self.stationary_potentials:
                dax, day, daz = galaxy_model.get_acceleration(pos[agent])
                agent_ax, agent_ay, agent_az = agent_ax + dax, agent_ay + day, agent_az + daz
            for other_agent in range(self.n_agents):
                if agent == other_agent: continue
                dax, day, daz = self.dynamic_potentials[other_agent].get_acceleration(pos[agent], selfpos=pos[other_agent])
                agent_ax, agent_ay, agent_az = agent_ax + dax, agent_ay + day, agent_az + daz
            a.append([agent_ax, agent_ay, agent_az])
        return a
    
    def get_equations(self, t, w):
        phasecoords = w.reshape(-1, 6)
        pos, vel = np.split(phasecoords, [3,3], axis=1)
        a = self.get_acceleration(pos)
        agent_phase_coords = np.array([vel[agent].tolist()+a[agent] for agent in range(self.n_agents)])
        return agent_phase_coords

    def _init_hyperparameters(self, hyperparameters):
        self.stationary_potential_list = ['point_source']
        self.stationary_potential_kwargs_list = [{'M':10}]
        self.dynamic_potential_list = []
        self.dynamic_potential_kwargs_list = [{}]
        self.seed = 0
        self.cuda = False
        self.orbit_timesteps = 1000
        self.orbit_duration = 1000 # Myr
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available else 'cpu')
        print(f'[ENV] Using {self.device}')

        if self.seed == None:
            self.seed = np.random.randint(0, 100)
        assert(type(self.seed) == int)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(f"[ENV] Seed set to {self.seed}")

    def _get_info(self):
        return {}
    
    def _get_ma(self, obs):
        return {'%d'%agent_id:obs for agent_id in range(self.n_agents)}
    
    def _process_actions(self, action):
        return np.clip(self._denormalise_state(action), self.low, self.high)
    
    def _normalise_state(self, state):
        return state / self.high
    
    def _denormalise_state(self, state):
        return state * self.high

    def error(self, true_val, pred_val):
        return torch.sum(((true_val - pred_val)/pred_val)**2)
    
    def out_of_bounds_damping(self, r_max):
        return np.e**(-1/self.high[0] / 1e3) if r_max < self.high[0] else np.e**(-r_max / self.high[0] / 1e3)