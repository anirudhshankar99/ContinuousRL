import gymnasium as gym
import numpy as np
try:
    from . import galaxy_models
except:
    import galaxy_models
import torch

class Orbit():
    def __init__(self, y, t, a):
        self.y = y
        self.t = t
        self.a = a

class Dynamics(gym.Env):
    def __init__(self, hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.high = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])*self.box_scaling
        self.low = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0])*self.box_scaling
        self.num_dynamic_potentials = len(self.dynamic_potential_list)
        high = np.repeat(np.expand_dims(self.high, 0), self.num_dynamic_potentials, 0).flatten()
        low = np.repeat(np.expand_dims(self.low, 0), self.num_dynamic_potentials, 0).flatten()
        self.high_cat = high
        self.low_cat = low
        self.km_to_pc = 3.241e-14
        self.Myr_to_sec = 86400 * 365 * 1e6
        self.orbit_duration *= self.Myr_to_sec
        self.action_space = gym.spaces.Box(
            low=self.low_cat,
            high=self.high_cat,
            dtype=np.float64
        )
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float64
        )

    def step(self, action):
        action = self._process_actions(action)
        init_params = np.clip(self.init_params + action, self.low_cat, self.high_cat)
        orbit = self._calculate_orbit(init_params)
        return orbit, None, False, False, {}

    def reset(self, init_params=[]):
        model_type_list = {key:0 for key,_ in galaxy_models.model_mapping.items()}
        if len(init_params) == 0:
            self.init_params = np.zeros((6 * self.num_dynamic_potentials,))
        else:
            assert (np.abs(np.array(init_params)) <= self.high_cat).all(), "If initial parameters are specified, they must be within the observation space"
            self.init_params = self._clip_state(np.array(init_params))
        self.stationary_potentials = []
        for model_name, model_kwargs_dict in zip(self.stationary_potential_list, self.stationary_potential_kwargs_list):
            self.stationary_potentials.append(galaxy_models.add_galaxy_model(model_name, **{'%s'%key:value for key,value in model_kwargs_dict.items()}))
            model_type_list[model_name] += 1
        self.dynamic_potentials = []
        for model_name, model_kwargs_dict in zip(self.dynamic_potential_list, self.dynamic_potential_kwargs_list):
            self.dynamic_potentials.append(galaxy_models.add_galaxy_model(model_name, **{'%s'%key:value for key,value in model_kwargs_dict.items()}))
            model_type_list[model_name] += 1
        return self.init_params, {'model_type_list': list(model_type_list.values())}

    def render(self, mode='human'):
        return
    
    def _calculate_orbit(self, init_params=[]):
        if len(init_params) == 0: init_params = self.init_params
        return self.leapfrog_verlet(self.get_acceleration, t_span=(0, self.orbit_duration), y0=init_params, delta_t=self.orbit_duration / self.orbit_timesteps)
        
    def leapfrog_verlet(self, ode_function, t_span, y0, delta_t):
        n_steps = int((t_span[1]-t_span[0])//delta_t)
        pos = np.zeros((n_steps, self.num_dynamic_potentials, 3))
        vel = np.zeros((n_steps, self.num_dynamic_potentials, 3))
        accs = np.zeros((n_steps, self.num_dynamic_potentials, 3))
        phasecoords = y0.reshape(-1, 6)
        pos[0], vel[0] = np.split(phasecoords, 2, axis=1) # shape (2,3,)
        t = np.linspace(t_span[0], t_span[1], n_steps)
        accs[0] = acc = np.array(ode_function(pos[0], t[0]))
        for i in range(1, n_steps):
            v_half = vel[i-1] + 0.5 * delta_t * acc * self.km_to_pc
            pos[i] = pos[i-1] + delta_t * v_half * self.km_to_pc
            # origin capture
            w = pos[i] - pos[i-1]
            origin_capture = vel[i] != vel[i]
            for galaxy_model in self.stationary_potentials:
                v = galaxy_model.pos - pos[i-1]
                w_dot_v = self.dot_product(w, v)
                v_dot_v = self.dot_product(v, v)
                t_ = w_dot_v / v_dot_v
                T = pos[i-1] + np.reshape(t_, (-1,1)) * v
                origin_capture =  np.linalg.norm((T - galaxy_model.pos), axis=-1) < self.origin_capture_delta * self.box_scaling
                vel[i][origin_capture] = np.zeros_like(vel[i][origin_capture])
                pos[i][origin_capture] = T[origin_capture]
                accs[i][origin_capture] = np.zeros_like(accs[i][origin_capture])
                if np.any(origin_capture): break
            accs[i][~origin_capture] = new_acc = np.array(ode_function(pos[i], t[i]))[~origin_capture]
            vel[i][~origin_capture] = (v_half + 0.5 * delta_t * new_acc * self.km_to_pc)[~origin_capture]
            acc = new_acc
            if np.all(origin_capture):
                pos[i:] = pos[i]
                vel[i:] = vel[i]
                accs[i:] = accs[i]
                break
        orbit_y = np.reshape(np.concat([pos, vel], axis=-1), (n_steps, -1)).transpose()
        accs = np.reshape(accs, (n_steps, -1)).transpose()
        return Orbit(orbit_y, np.linspace(t_span[0], t_span[1], n_steps), accs)
    
    def get_acceleration(self, pos, t=None):
        """
        pos is the list of positions of all the agents
        """
        a = []
        for agent in range(self.num_dynamic_potentials):
            agent_ax, agent_ay, agent_az = 0, 0, 0
            for galaxy_model in self.stationary_potentials:
                if galaxy_model.sign == 'bar':
                    dax, day, daz = galaxy_model.get_acceleration(np.concat([pos[agent], np.array([t])], axis=-1))
                else:
                    dax, day, daz = galaxy_model.get_acceleration(pos[agent])
                agent_ax, agent_ay, agent_az = agent_ax + dax, agent_ay + day, agent_az + daz
            for other_agent in range(self.num_dynamic_potentials):
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
        for agent in range(self.num_dynamic_potentials):
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
        self.orbit_timesteps = 100
        self.orbit_duration = 100 # Myr
        self.box_scaling = 1
        self.origin_capture_delta = 1e-1
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
    
    def _process_actions(self, action):
        return np.clip(self._denormalise_state(action), self.low_cat, self.high_cat)
    
    def _clip_state(self, state):
        return np.clip(self._denormalise_state(state), self.low_cat, self.high_cat)

    def _normalise_state(self, state):
        return state / self.high_cat
    
    def _denormalise_state(self, state):
        return state * self.high_cat
    
    def dot_product(self, a, b):
        return np.sum(np.einsum('...i,...i->...i',a,b), axis=-1)