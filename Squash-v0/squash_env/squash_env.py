import gymnasium as gym
from gymnasium import spaces
import pygame
import torch
import numpy as np
from matplotlib.colors import hsv_to_rgb

class SquashEnv(gym.Env):

    def __init__(self, hyperparameters:dict={}):
        self._init_hyperparameters(hyperparameters)
        self.render_modes = ['human', 'logs', 'none']
        if self.render_mode.lower() not in self.render_modes: 
            print('The render_mode must be either \'human\', \'logs\' or \'none\'. Defaulting to none.')
            self.render_mode = 'none'
        if self.size_y_to_x_ratio > 1:
            self.size_x = 1/self.size_y_to_x_ratio
            self.size_y = 1
        else:
            self.size_y = 1/self.size_y_to_x_ratio
            self.size_x = 1
        self.agent_speed = 0.4
        self._ball_size = self.size_x / 10 # 20 balls can fill the width of the field
        self._agent_size = self._ball_size # agent takes the same size
        self._wall_width = self._ball_size / 5 # 20% of the ball's size
        self.max_impulse = self.size_y # covers the length of the field in 1 second
        self.min_impulse = 0.25 * self.size_y # covers the length of the field in 4 seconds
        self._window_size_scaling = 800
        self._agent_colours = (np.stack(self._get_agent_colours(), axis=0)*255).astype(int)
        self.initial_volley_reward_fraction = self.volley_reward_fraction

        # Observations: agent positions (x,y) x n_agents, ball(x,y), distance_to_ball (x,y), ball_velocity(v_r,v_theta), my_turn_in, agent_after_me, num_agents_remaining
        observation_low_positions = []
        observation_high_positions = []
        for _ in range(self.num_agents):
            observation_low_positions += [0., 0.]
            observation_high_positions += [self.size_x, self.size_y]
        self.observation_space = spaces.Box(
            low=np.array(observation_low_positions+[0., 0., 0., 0., 0., -np.pi / 2, 0, 0, 1]),
            high=np.array(observation_high_positions+[self.size_x, self.size_y, self.size_x, self.size_y, self.max_impulse / 0.1, np.pi / 2, self.num_agents - 1, self.num_agents - 1, self.num_agents]), # eventually remove velocity limit
            dtype=np.float32
        )

        # Actions: agent move (x,y), impulse_to_ball, impulse_angle
        self.action_space = spaces.Box(
            low=np.array([-1., -1., self.min_impulse, -np.pi / 2,]),
            high=np.array([1., 1., self.max_impulse, np.pi / 2,]), # eventually remove impulse limit
            dtype=np.float32
        )

        self._wall_positions = [[np.array([0,0]), np.array([0, self.size_y])], 
                               [np.array([self.size_x,0]), np.array([self.size_x, self.size_y])], 
                               [np.array([0,self.size_y]), np.array([self.size_x, self.size_y])], 
                               [np.array([0,0]), np.array([self.size_x, 0])]]

        self.display = None
        self.clock = None
        if self.render_mode == 'logs':
            self._logs = []

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._agent_locations = np.stack([np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64) for _ in range(self.num_agents)], dtype=np.float64, axis=0) # (2*n_agents,)
        self._ball_location = np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64)
        self._ball_velocity = np.array([0., 0.], dtype=np.float64)
        self._agent_turn = 0
        self._agent_turns = {i:i for i in range(self.num_agents)} # { agent: my_turn_in, ...}
        self.reward, self.terminated = {'agent_%d'%i:0. for i in range(self.num_agents)}, {'agent_%d'%i:False for i in range(self.num_agents)}
        self.reward['agent_0'] = -1*self.volley_reward_fraction
        self.contacts = 0
        self.struck = False # whether or not the player has struck the ball
        if self.scaling_volley: self.volley_reward_fraction = self.initial_volley_reward_fraction

        observation = self._get_obs()
        info = self._get_info() # to include number of volleys

        if self.render_mode == 'human':
            self.render()
        if self.render_mode == 'logs':
            self._log_entry('[RESET] ')

        return observation, info

    def step(self, action):
        """
        Action is of the form { 'agent_x': (action_dim, ), ...}
        """
        processed_actions = self._process_actions(action)
        if self._ball_out_of_bounds():
            # eliminate player
            current_players = list(self._agent_turns.keys())
            eliminated_agent = self._agent_turn
            current_players.pop(current_players.index(eliminated_agent))
            self.terminated['agent_%d'%eliminated_agent] = True

            # reward eliminator (the one with highest turns left), survivors, and penalize loser
            eliminator = list(self._agent_turns.keys())[list(self._agent_turns.values()).index(np.max(list(self._agent_turns.values())))]
            
            # winner
            if len(current_players) == 1: 
                self.terminated['agent_%d'%eliminator] = True
                # self.reward['agent_%d'%eliminator] += 1 # an extra reward for being the last one standing
                self.reward['agent_%d'%eliminated_agent] -= 1*self.volley_reward_fraction*0.5
                # return dummy observations and all dones
                return {'agent_%d'%i:torch.zeros(2 * self.num_agents + 7, device=self.device) for i in range(self.num_agents)}, self.reward, {'agent_%d'%i:True for i in range(self.num_agents)}, {'agent_%d'%i:False for i in range(self.num_agents)}, self._get_info()

            for standing_agent in list(self._agent_turns.keys()):
                if standing_agent == eliminator: self.reward['agent_%d'%standing_agent] += 1
                else: self.reward['agent_%d'%standing_agent] += 1*self.surviving_reward_fraction

            # turns for remaining players
            self._agent_turns = {
                current_players[i]:i for i in range(len(current_players))
            }
            self._agent_turn = current_players[0]

            # reset players and ball
            self._ball_location = np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64)
            self._agent_locations = np.stack([np.array([self.size_x / 2, self.size_y / 2], dtype=np.float64) for _ in range(self.num_agents)], dtype=np.float64, axis=0) # (2*n_agents,)
            self._ball_velocity = np.array([0., 0.], dtype=np.float64)

        self._ball_velocity[1] = self._ball_rebound()

        if self.struck == False:
            for i in list(self._agent_turns.keys()):
                # check for ball contact if in turn
                if i == self._agent_turn:
                    if self._euclidian_dist(self._agent_locations[i], self._ball_location) < self._ball_size:
                        # contact
                        if self.render_mode == 'logs': self._log_entry('[CONTACT] by agent %d. '%self._agent_turn)
                        self.contacts += 1
                        self._ball_velocity = self._2d_collision_elastic(self._ball_velocity, action['agent_%d'%i][-2:])
                        self.reward['agent_%d'%self._agent_turn] += 1*self.volley_reward_fraction # rewards receiver
                        self.struck = True

        for i in list(self._agent_turns.keys()):
            # perform movement if agent is still alive
            if i in self._agent_turns:
                # still alive
                self._agent_locations[i] = np.clip(
                    self._agent_locations[i] + processed_actions['agent_%d'%i][:-2],
                    [self._ball_size, self._ball_size],
                    [self.size_x - self._ball_size, self.size_y - self._ball_size]
                    )
        # ball movement
        self._ball_location = self._ball_location + np.stack([self._ball_velocity[0]*np.cos(self._ball_velocity[1]), self._ball_velocity[0]*np.sin(self._ball_velocity[1])]) / self.fps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, self.reward, self.terminated, {'agent_%d'%i:False for i in range(self.num_agents)}, info

    def render(self):
        if self.display is None:
            pygame.init()
            pygame.display.init()
            self.font = pygame.font.Font('freesansbold.ttf', 32)
            self.display = pygame.display.set_mode((self._window_size_scaling*self.size_x, self._window_size_scaling*self.size_y))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.display.fill((255, 255, 255))

        turn_text = self.font.render('Turn: %d'%self._agent_turn, True, (0, 0, 0), (255, 255, 255))
        turn_text_box = turn_text.get_rect()
        turn_text_box.center = (int(self.size_x * 0.8 * self._window_size_scaling), int(self.size_y * 0.2 * self._window_size_scaling))
        self.display.blit(turn_text, turn_text_box)
        pygame.draw.circle(self.display, (255, 0, 0), (int(self._ball_location[0]*self._window_size_scaling), int(self._ball_location[1]*self._window_size_scaling)), self._ball_size*self._window_size_scaling)

        for standing_agent in list(self._agent_turns.keys()):
            agent_text = self.font.render('%d'%standing_agent, True, (0, 0, 0), self._agent_colours[standing_agent])
            agent_text_box = agent_text.get_rect()
            agent_text_box.center = ((int(self._agent_locations[standing_agent][0]*self._window_size_scaling), int(self._agent_locations[standing_agent][1]*self._window_size_scaling)))
            pygame.draw.circle(self.display, self._agent_colours[standing_agent], (int(self._agent_locations[standing_agent][0]*self._window_size_scaling), int(self._agent_locations[standing_agent][1]*self._window_size_scaling)), self._agent_size*self._window_size_scaling)
            self.display.blit(agent_text, agent_text_box)

        for position_pair in self._wall_positions:
            pygame.draw.line(self.display, (0, 0, 0), position_pair[0].astype(int).tolist(), position_pair[1].astype(int).tolist(), int(self._wall_width * self._window_size_scaling))
        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _init_hyperparameters(self, hyperparameters):
        self.size_y_to_x_ratio = 1.5
        self.seed = 0
        self.num_agents = 2
        self.fps = 30
        self.volley_reward_fraction = 0.25
        self.surviving_reward_fraction = 0.25
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.render_mode = 'none'
        self.scaling_volley = False

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + '%s'%val)

        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Seed set to {self.seed}")

    def _get_obs(self):
        obs = {}
        for i in range(self.num_agents):
            if i not in self._agent_turns.keys(): obs['agent_%d'%i] = torch.tensor(self._agent_locations.flatten().tolist()+self._ball_location.tolist()+(self._ball_location - self._agent_locations[i]).tolist()+self._ball_velocity.tolist()+
                          [np.inf, np.inf, len(self._agent_turns)], dtype=torch.float32).to(self.device)
            else: obs['agent_%d'%i] = torch.tensor(self._agent_locations.flatten().tolist()+self._ball_location.tolist()+(self._ball_location - self._agent_locations[i]).tolist()+self._ball_velocity.tolist()+
                          [self._agent_turns[i], list(self._agent_turns.keys())[list(self._agent_turns.values()).index((self._agent_turns[i]+1)%(self.num_agents))],
                          len(self._agent_turns)], dtype=torch.float32).to(self.device)
        return obs

    def _get_info(self):
        return {'contacts':self.contacts}
    
    def _process_actions(self, action):
        return {
            agent_id:(action[agent_id] / np.concat([np.abs(action[agent_id][:-2]), action[agent_id][-2:]]) * self.agent_speed / self.fps) for agent_id in action.keys()
        }
    
    def _euclidian_dist(self, position_1, position_2):
        return np.linalg.norm(position_1 - position_2)
    
    def _ball_rebound(self):
        def _reflect(theta, phi): return 2 * phi - theta
        left, right, up = self._ball_location[0] < self._ball_size, self._ball_location[0] > self.size_x - self._ball_size, self._ball_location[1] < self._ball_size
        if up:
            # change turn
            self.reward['agent_%d'%self._agent_turn] += 1*self.volley_reward_fraction # rewards setter
            self._agent_turn = list(self._agent_turns.keys())[list(self._agent_turns.values()).index(1)]
            self.struck = False
            if self.scaling_volley: self._increase_volley_reward()
            for key in self._agent_turns.keys():
                if self._agent_turns[key] > 0: 
                    self._agent_turns[key] -= 1
                else: self._agent_turns[key] = len(self._agent_turns) - 1
            if left: 
                if self.render_mode == 'logs': self._log_entry('[REBOUND] off the top-left. agent %d\'s turn now. '%list(self._agent_turns.keys())[list(self._agent_turns.values()).index(np.min(list(self._agent_turns.values())))])
                return _reflect(_reflect(self._ball_velocity[1], 3 * np.pi / 2), 0)
            elif right:
                if self.render_mode == 'logs': self._log_entry('[REBOUND] off the top-right. agent %d\'s turn now. '%list(self._agent_turns.keys())[list(self._agent_turns.values()).index(np.min(list(self._agent_turns.values())))])
                return _reflect(_reflect(self._ball_velocity[1], np.pi / 2), 0)
            else: 
                if self.render_mode == 'logs': self._log_entry('[REBOUND] off the top. agent %d\'s turn now. '%list(self._agent_turns.keys())[list(self._agent_turns.values()).index(np.min(list(self._agent_turns.values())))])
                return _reflect(self._ball_velocity[1], np.pi)

        elif left: 
            if self.render_mode == 'logs': self._log_entry('[REBOUND] off the left wall. ')
            return _reflect(self._ball_velocity[1], 3 * np.pi / 2)
        elif right: 
            if self.render_mode == 'logs': self._log_entry('[REBOUND] off the right wall. ')
            return _reflect(self._ball_velocity[1], np.pi / 2)
        return self._ball_velocity[1]
    
    def _ball_out_of_bounds(self):
        return self._ball_location[1] > self.size_y - self._ball_size
    
    def _2d_collision_elastic(self, initial_velocity, impulse):
        ball_velocity_x, ball_velocity_y = initial_velocity[0] * np.cos(initial_velocity[1]), initial_velocity[0] * np.sin(initial_velocity[1])
        impulse_x, impulse_y = impulse[0] * np.cos(impulse[1]), impulse[0] * np.sin(impulse[1])
        final_ball_velocity_x, final_ball_velocity_y = ball_velocity_x + impulse_x, ball_velocity_y + impulse_y
        return np.stack([np.sqrt(final_ball_velocity_x ** 2 + final_ball_velocity_y ** 2), np.arctan2(final_ball_velocity_y, final_ball_velocity_x)])
    
    def _get_agent_colours(self):
        colors = []
        for i in np.arange(0., 360., 360. / self.num_agents):
            h = i / 360.
            l = (50 + np.random.rand() * 10) / 100.
            s = (90 + np.random.rand() * 10) / 100.
            colors.append(hsv_to_rgb([h, l, s]))
        return colors
    
    def _log_entry(self, state_description):
        _log = state_description
        for standing_agent in list(self._agent_turns.keys()):
            _log += 'agent_%d'%standing_agent + ' is at ('
            for coordinate in self._agent_locations[standing_agent]:
                _log += '%.2f,'%coordinate
            _log += '), '
        _log += 'the ball is at ('
        for coordinate in self._ball_location:
            _log += '%.2f,'%coordinate
        _log += ').'
        print(_log)
        self._logs.append(_log)

    def _increase_volley_reward(self):
        self.volley_reward_fraction += self.initial_volley_reward_fraction / 4