import gymnasium as gym
import numpy as np
try:
    from . import galaxy_models
except:
    import galaxy_models
import torch

class Orbit():
    def __init__(self, y, t):
        self.y = y
        self.t = t

class Dynamics(gym.Env):
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.high = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])*self.box_scaling
        self.low = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0])*self.box_scaling
        self.num_agents = len(self.dynamic_potential_list)
        high = np.repeat(np.expand_dims(self.high, 0), self.num_agents, 0).flatten()
        low = np.repeat(np.expand_dims(self.low, 0), self.num_agents, 0).flatten()
        self.high_cat = high
        self.low_cat = low
        self.action_space = gym.spaces.Box(
            low=self.low,
            high=self.high,
            dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=low, 
            high=high,
            dtype=np.float64
        )

    def step(self, action, delta=1e-8):
        action = self._process_actions(action)
        self.init_params = np.clip(self.init_params + action, self.low_cat, self.high_cat)
        init_params_delta = self.init_params + np.random.normal(scale=delta, size=len(self.init_params))
        orbit = self._calculate_orbit()
        orbit_delta = self._calculate_orbit(init_params=init_params_delta)
        if orbit == None or orbit_delta == None:
            return self._get_ma(self.init_params), {agent:0 for agent in range(self.num_agents)}, self._get_ma(False), self._get_ma(False), {'orbit':orbit, 'orbit_delta':orbit_delta}
        agent_rewards = {}
        for agent in range(self.num_agents):
            orbit_dists = np.linalg.norm(orbit.y[agent * 6: (agent+1) * 6] - orbit_delta.y[agent * 6: (agent+1) * 6], axis=0)
            log_orbit_dists = np.log(orbit_dists + 1e-8)
            fit_coeffs = np.polyfit(orbit.t, log_orbit_dists, 1)
            max_r = np.max(np.linalg.norm(orbit.y[agent * 6: (agent+1) * 6][:3], axis=0))
            agent_rewards[agent] = fit_coeffs[0] * self.out_of_bounds_damping(max_r)
        return self._get_ma(self.init_params), agent_rewards, self._get_ma(False), self._get_ma(False), {'orbit':orbit, 'orbit_delta':orbit_delta}

    def reset(self, init_params=[]):
        if len(init_params) == 0:
            self.init_params = np.zeros((6 * self.num_agents,))
        else:
            assert (np.abs(np.array(init_params)) <= self.high_cat).all(), "If initial parameters are specified, they must be within the observation space"
            self.init_params = self._clip_state(np.array(init_params))
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
        return self.leapfrog_verlet(self.get_acceleration, t_span=(0, self.orbit_duration), y0=init_params, delta_t=self.orbit_duration / self.orbit_timesteps)
        
    def leapfrog_verlet(self, ode_function, t_span, y0, delta_t):
        n_steps = int((t_span[1]-t_span[0])//delta_t)
        pos = np.zeros((n_steps, self.num_agents, 3))
        vel = np.zeros((n_steps, self.num_agents, 3))
        phasecoords = y0.reshape(-1, 6)
        pos[0], vel[0] = np.split(phasecoords, 2, axis=1) # shape (2,3,)
        t = np.linspace(t_span[0], t_span[1], n_steps)
        acc = np.array(ode_function(pos[0], t[0]))
        for i in range(1, n_steps):
            v_half = vel[i-1] + 0.5 * delta_t * acc
            pos[i] = pos[i-1] + delta_t * v_half
            new_acc = np.array(ode_function(pos[i], t[i]))
            vel[i] = v_half + 0.5 * delta_t * new_acc
            acc = new_acc
        orbit_y = np.reshape(np.concat([pos, vel], axis=-1), (n_steps, -1)).transpose()
        return Orbit(orbit_y, np.linspace(t_span[0], t_span[1], n_steps))

    def reverse_leapfrog_verlet(self, ode_function, t_span, y0, delta_t):
        n_steps = int((t_span[1]-t_span[0])//delta_t)
        pos = np.zeros((n_steps, self.num_agents, 3))
        vel = np.zeros((n_steps, self.num_agents, 3))
        phasecoords = y0.reshape(-1, 6)
        pos[n_steps-1], vel[n_steps-1] = np.split(phasecoords, 2, axis=1) # shape (2,3,)
        t = np.linspace(t_span[0], t_span[1], n_steps)
        acc = np.array(ode_function(pos[n_steps-1], t[n_steps-1]))
        for i in range(n_steps-1,0,-1):
            v_half = vel[i] - 0.5*acc*delta_t
            pos[i-1] = pos[i] - v_half*delta_t
            prev_acc = np.array(ode_function(pos[i-1], t[i-1]))
            vel[i-1] = v_half - 0.5*prev_acc*delta_t
        orbit_y = np.reshape(np.concat([pos, vel], axis=-1), (n_steps, -1)).transpose()
        return Orbit(orbit_y, np.linspace(t_span[0], t_span[1], n_steps))
    def get_acceleration(self, pos, t=None):
        """
        pos is the list of positions of all the agents
        """
        a = []
        for agent in range(self.num_agents):
            agent_ax, agent_ay, agent_az = 0, 0, 0
            for galaxy_model in self.stationary_potentials:
                if galaxy_model.sign == 'bar':
                    dax, day, daz = galaxy_model.get_acceleration(np.concat([pos[agent], np.array([t])], axis=-1))
                else:
                    dax, day, daz = galaxy_model.get_acceleration(pos[agent])
                agent_ax, agent_ay, agent_az = agent_ax + dax, agent_ay + day, agent_az + daz
            for other_agent in range(self.num_agents):
                if agent == other_agent: continue
                if self.dynamic_potentials[other_agent].sign == 'bar':
                    dax, day, daz = self.dynamic_potentials[other_agent].get_acceleration(np.concat([pos[agent], np.array([t])], axis=-1), selfpos=pos[other_agent])
                else:
                    dax, day, daz = self.dynamic_potentials[other_agent].get_acceleration(pos[agent], selfpos=pos[other_agent])
                agent_ax, agent_ay, agent_az = agent_ax + dax, agent_ay + day, agent_az + daz
            a.append([agent_ax.item(), agent_ay.item(), agent_az.item()])
        return a
    
    def get_equations(self, t, w):
        # of the form (x y z vx vy vz), (x y z vx vy vz),...
        phasecoords = w.reshape(-1, 6)
        pos, vel = np.split(phasecoords, 2, axis=1)
        a = self.get_acceleration(pos)
        phasecoords_dot = []
        for agent in range(self.num_agents):
            phasecoords_dot += vel[agent].tolist() + a[agent]
        # of the form (x. y. z. vx. vy. vz.), ...
        return np.array(phasecoords_dot)

    def _init_hyperparameters(self, hyperparameters):
        self.stationary_potential_list = ['point_source']
        self.stationary_potential_kwargs_list = [{'M':10}]
        self.dynamic_potential_list = []
        self.dynamic_potential_kwargs_list = [{}]
        self.seed = 0
        self.cuda = False
        self.orbit_timesteps = 1000
        self.orbit_duration = 10000 # Myr
        self.box_scaling = 1
        self.damping_delta = 0.1
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
        return {agent_id:obs for agent_id in range(self.num_agents)}
    
    def _process_actions(self, action):
        action = np.concat(list(action.values()))
        return np.clip(self._denormalise_state(action), self.low_cat, self.high_cat)
    
    def _clip_state(self, state):
        return np.clip(self._denormalise_state(state), self.low_cat, self.high_cat)

    def _normalise_state(self, state):
        return state / self.high_cat
    
    def _denormalise_state(self, state):
        return state * self.high_cat
    
    def out_of_bounds_damping(self, r_max):
        return (1 + self.damping_delta)*np.e**(-1) if r_max < self.high[0] else np.e**(-r_max / self.high[0])+self.damping_delta*np.e**(-1)